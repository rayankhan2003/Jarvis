"""Builds and runs the voice pipeline.

    mic -> wake word gate -> speech-to-text -> instant commands -> brain (+ tools)
        -> voice -> speakers
"""

from __future__ import annotations

import subprocess
import sys

from loguru import logger
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import LLMRunFrame, TTSSpeakFrame
from pipecat.pipeline.llm_switcher import LLMSwitcher
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker, ProcessorUnusablePolicy
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.services.kokoro.tts import KokoroTTSService
from pipecat.transcriptions.language import Language
from pipecat.transports.local.audio import LocalAudioTransport, LocalAudioTransportParams
from pipecat.turns.user_mute import AlwaysUserMuteStrategy
from pipecat.workers.runner import WorkerRunner

from jarvis.config import Config
from jarvis.failover import FreeTierFailover, keep_only_messages_for
from jarvis.fastpath import FastPath
from jarvis.persona import system_prompt
from jarvis.tools import ALL_TOOLS
from jarvis.wake import WakeWordGate, load_wake_model

# Tools slow enough that silence would feel broken; Jarvis says so first.
SLOW_TOOLS = {"web_search", "get_weather", "create_reminder"}
WAKE_SOUND = "/System/Library/Sounds/Tink.aiff"


class SetupError(RuntimeError):
    pass


def make_llms(config: Config, prompt: str) -> list:
    llms = []
    for name in config.brains():
        if name == "gemini":
            from pipecat.services.google.llm import GoogleLLMService

            llms.append(GoogleLLMService(
                api_key=config.gemini_api_key,
                settings=GoogleLLMService.Settings(model=config.gemini_model, system_instruction=prompt),
            ))
        elif name == "groq":
            from pipecat.services.groq.llm import GroqLLMService

            llms.append(GroqLLMService(
                api_key=config.groq_api_key,
                settings=GroqLLMService.Settings(model=config.groq_llm_model, system_instruction=prompt),
            ))
        elif name == "ollama":
            from pipecat.services.ollama.llm import OLLamaLLMService

            llms.append(OLLamaLLMService(
                base_url=config.ollama_url,
                settings=OLLamaLLMService.Settings(model=config.ollama_model, system_instruction=prompt),
            ))
    if not llms:
        raise SetupError("No brain configured. Set GEMINI_API_KEY and/or GROQ_API_KEY in .env "
                         "(both are free), or OLLAMA_MODEL for offline use.")
    return llms


def make_stt(config: Config):
    if config.local_stt:
        from pipecat.services.whisper.stt import MLXModel, WhisperSTTServiceMLX

        return WhisperSTTServiceMLX(settings=WhisperSTTServiceMLX.Settings(
            model=MLXModel.LARGE_V3_TURBO_Q4.value, language=Language.EN))
    if not config.groq_api_key:
        raise SetupError("Speech-to-text uses Groq by default: set GROQ_API_KEY (free), "
                         "or JARVIS_LOCAL_STT=1 to transcribe on the Mac instead.")
    from pipecat.services.groq.stt import GroqSTTService

    return GroqSTTService(api_key=config.groq_api_key, settings=GroqSTTService.Settings(
        model=config.groq_stt_model, language=Language.EN))


def play_wake_sound():
    if sys.platform == "darwin":
        subprocess.Popen(["afplay", WAKE_SOUND], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


async def run_jarvis(config: Config):
    prompt = system_prompt(config)

    transport = LocalAudioTransport(LocalAudioTransportParams(audio_in_enabled=True, audio_out_enabled=True))
    stt = make_stt(config)
    llms = make_llms(config, prompt)
    tts = KokoroTTSService(settings=KokoroTTSService.Settings(voice=config.voice, language=Language.EN_GB))

    switcher = LLMSwitcher(llms=llms, strategy_type=FreeTierFailover) if len(llms) > 1 else None
    brain = switcher or llms[0]
    logger.info(f"Brain providers, in order: {', '.join(config.brains())}")

    context = LLMContext(tools=ALL_TOOLS)
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            vad_analyzer=SileroVADAnalyzer(),
            empty_user_turn=None,
            # Without headphones the mic hears Jarvis and would cut it off after a word.
            user_mute_strategies=[] if config.interruptions else [AlwaysUserMuteStrategy()],
        ),
    )

    gate = None
    if not config.always_listen:
        gate = WakeWordGate(
            load_wake_model(),
            threshold=config.wake_threshold,
            awake_secs=config.awake_secs,
            on_wake=play_wake_sound,
            mute_while_speaking=not config.interruptions,
        )

    async def dismiss():
        if gate:
            gate.sleep()

    async def new_user_turn():
        if switcher:
            await switcher.strategy.restore_preferred()

    fastpath = FastPath(context, on_dismiss=dismiss, on_user_turn=new_user_turn)

    pipeline = Pipeline([
        transport.input(),
        *([gate] if gate else []),
        stt,
        fastpath,
        user_aggregator,
        brain,
        tts,
        transport.output(),
        assistant_aggregator,
    ])

    worker = PipelineWorker(
        pipeline,
        params=PipelineParams(enable_metrics=True, enable_usage_metrics=True),
        processor_unusable_policy=ProcessorUnusablePolicy.END,
    )

    for llm in llms:
        @llm.event_handler("on_function_calls_started")
        async def on_function_calls_started(service, function_calls):
            if any(call.function_name in SLOW_TOOLS for call in function_calls):
                await tts.queue_frame(TTSSpeakFrame("One moment.", append_to_context=False))

    if switcher:
        def adopt(service):
            dropped = keep_only_messages_for(context, service.get_llm_adapter().id_for_llm_specific_messages)
            if dropped:
                logger.debug(f"Dropped {dropped} provider-specific messages for {service.name}")

        @switcher.strategy.event_handler("on_service_switched")
        async def on_service_switched(strategy, service):
            adopt(service)

        @switcher.strategy.event_handler("on_failover")
        async def on_failover(strategy, failed, new, category):
            # The failed provider never answered; ask the new one the same thing.
            adopt(new)
            await worker.queue_frames([LLMRunFrame()])

    greeting = f"Jarvis online, {config.honorific}."
    if gate:
        greeting += " Say 'Hey Jarvis' when you need me."
    await worker.queue_frames([TTSSpeakFrame(greeting, append_to_context=False)])

    runner = WorkerRunner()
    await runner.add_workers(worker)
    await runner.run()
