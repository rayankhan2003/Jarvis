"""Builds and runs the voice pipeline.

    mic -> wake word gate -> speech-to-text -> instant commands -> brain (+ tools)
        -> voice -> speakers
"""

from __future__ import annotations

import asyncio
import subprocess
import sys

from loguru import logger
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import EndFrame, LLMRunFrame, TTSSpeakFrame
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

from jarvis import conversation, screen
from jarvis.config import Config
from jarvis.failover import FreeTierFailover, keep_only_messages_for
from jarvis.fastpath import FastPath
from jarvis.memory import MEMORY
from jarvis.overlay import OVERLAY, OverlayStatus
from jarvis.persona import system_prompt
from jarvis.ptt import PushToTalk
from jarvis.router import BrainRouter, ModelRouter
from jarvis.timers import TIMERS
from jarvis.tools import ALL_TOOLS, info, system
from jarvis.usage import Usage, UsageMeter
from jarvis.voices import language_for
from jarvis.wake import WakeWordGate, load_wake_model

REASONING = {"low", "medium", "high"}
# Spoken replies are a sentence or two; capping them also caps free-tier tokens.
MAX_REPLY_TOKENS = 300
# Tools slow enough that silence would feel broken; Jarvis says so first.
SLOW_TOOLS = {"web_search", "get_weather", "create_reminder"}
WAKE_SOUND = "/System/Library/Sounds/Tink.aiff"


class SetupError(RuntimeError):
    pass


def make_llms(config: Config, prompt: str) -> list:
    llms = []
    for name in config.brains():
        if name == "mistral":
            from pipecat.services.mistral.llm import MistralLLMService

            llms.append(MistralLLMService(
                api_key=config.mistral_api_key,
                settings=MistralLLMService.Settings(
                    model=config.mistral_model, system_instruction=prompt, max_tokens=MAX_REPLY_TOKENS),
            ))
        elif name == "gemini":
            from pipecat.services.google.llm import GoogleLLMService

            llms.append(GoogleLLMService(
                api_key=config.gemini_api_key,
                settings=GoogleLLMService.Settings(model=config.gemini_model, system_instruction=prompt),
            ))
        elif name == "groq":
            from pipecat.services.groq.llm import GroqLLMService

            llms.append(GroqLLMService(
                api_key=config.groq_api_key,
                settings=GroqLLMService.Settings(
                    model=config.groq_llm_model, system_instruction=prompt,
                    reasoning_effort=config.groq_reasoning if config.groq_reasoning in REASONING else "low"),
            ))
        elif name == "ollama":
            from pipecat.services.ollama.llm import OLLamaLLMService

            llms.append(OLLamaLLMService(
                base_url=config.ollama_url,
                settings=OLLamaLLMService.Settings(model=config.ollama_model, system_instruction=prompt),
            ))
    if not llms:
        raise SetupError("No brain configured. Set MISTRAL_API_KEY, GROQ_API_KEY and/or GEMINI_API_KEY "
                         "in .env (all free), or OLLAMA_MODEL for offline use.")
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
    # The user's profile: defaults the tools fall back on.
    system.DEFAULT_BROWSER = config.browser
    system.MUSIC_TASTE = config.music_taste
    info.HOME_CITY = config.city
    prompt = system_prompt(config, memories=MEMORY.prompt_block())

    transport = LocalAudioTransport(LocalAudioTransportParams(audio_in_enabled=True, audio_out_enabled=True))
    stt = make_stt(config)
    llms = make_llms(config, prompt)
    tts = KokoroTTSService(settings=KokoroTTSService.Settings(voice=config.voice, language=language_for(config.voice)))

    switcher = LLMSwitcher(llms=llms, strategy_type=FreeTierFailover) if len(llms) > 1 else None
    brain = switcher or llms[0]
    logger.info(f"Brain providers, in order: {', '.join(config.brains())}")

    names = config.brains()
    router = None
    if "mistral" in names and config.mistral_complex_model != config.mistral_model:
        mistral = llms[names.index("mistral")]
        router = ModelRouter(mistral, type(mistral).Settings, config.mistral_model, config.mistral_complex_model)
    # The big brain: complex requests go to Groq's large model when it isn't first already.
    big_brain = None
    if switcher and config.smart_brain and "groq" in names and names[0] != "groq":
        big_brain = BrainRouter(switcher.strategy, llms[names.index("groq")])
        logger.info(f"Big brain for complex requests: {config.groq_llm_model}")

    async def refresh_prompt():
        # Memory changed: every brain gets the new instructions for the next request.
        new_prompt = system_prompt(config, memories=MEMORY.prompt_block())
        for llm in llms:
            await llm._update_settings(type(llm).Settings(system_instruction=new_prompt))

    MEMORY.on_change = refresh_prompt
    usage = Usage()

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

    use_wake_word = config.trigger in {"wake", "both"}
    use_key = config.trigger in {"key", "both"}
    screen.CONFIG = config
    if config.buddy:
        OVERLAY.start()

    def woke():
        play_wake_sound()
        OVERLAY.status("listening")

    def slept():
        if use_wake_word:
            # Each "Hey Jarvis" starts a fresh, small conversation. With push-to-talk only,
            # the gate sleeps after every release, so keep the (already capped) conversation.
            conversation.clear(context)
        OVERLAY.status("idle")

    gate = None
    if not config.always_listen:
        gate = WakeWordGate(
            load_wake_model() if use_wake_word else None,
            threshold=config.wake_threshold,
            awake_secs=config.awake_secs,
            on_wake=woke,
            mute_while_speaking=not config.interruptions,
            on_sleep=slept,
        )

    ptt = None
    if gate and use_key:
        ptt = PushToTalk(config.ptt_key, on_press=gate.hold, on_release=gate.release)
        if not ptt.start(asyncio.get_running_loop()):
            ptt = None
            if not use_wake_word:
                # Key-only, but the key can't be heard: fall back to the wake word rather than go deaf.
                logger.warning("Push-to-talk isn't available; listening for 'Hey Jarvis' instead")
                gate._model = load_wake_model()
                use_wake_word = True

    async def dismiss():
        if gate:
            gate.sleep()

    async def new_user_turn():
        OVERLAY.status("thinking")
        if switcher:
            await switcher.strategy.restore_preferred()

    def route(text: str):
        if big_brain and (frames := big_brain.frames_for(text)):
            logger.info(f"Using the big brain ({config.groq_llm_model}) for this request")
            return frames
        if router and (frames := router.frames_for(text)):
            logger.info(f"Using {router.current} for this request")
            return frames
        return []

    async def shutdown():
        # EndFrame lets the goodbye finish playing before the pipeline stops.
        await TIMERS.cancel()
        await worker.queue_frames([EndFrame()])
        OVERLAY.stop()

    fastpath = FastPath(context, on_dismiss=dismiss, on_shutdown=shutdown, on_user_turn=new_user_turn,
                        before_brain=route, usage=usage)

    pipeline = Pipeline([
        transport.input(),
        *([gate] if gate else []),
        stt,
        fastpath,
        user_aggregator,
        brain,
        tts,
        transport.output(),
        OverlayStatus(OVERLAY),
        assistant_aggregator,
        UsageMeter(usage),
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

    async def announce(message: str):
        await worker.queue_frames([TTSSpeakFrame(f"{config.honorific.capitalize()}, {message[0].lower()}{message[1:]}",
                                                 append_to_context=False)])

    TIMERS.on_fire = announce

    key = config.ptt_key.replace("_", " ")
    greeting = f"Jarvis online, {config.honorific}."
    if gate and ptt and use_wake_word:
        greeting += f" Hold {key}, or say 'Hey Jarvis'."
    elif gate and ptt:
        greeting += f" Hold {key} to talk."
    elif gate:
        greeting += " Say 'Hey Jarvis' when you need me."
    await worker.queue_frames([TTSSpeakFrame(greeting, append_to_context=False)])

    runner = WorkerRunner()
    await runner.add_workers(worker)
    try:
        await runner.run()
    finally:
        if ptt:
            ptt.stop()
        OVERLAY.stop()
