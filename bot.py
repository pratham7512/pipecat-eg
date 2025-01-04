#
# Copyright (c) 2024, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import os
import sys
from pathlib import Path
from sys import platform

from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import LLMMessagesFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.services.cartesia import CartesiaTTSService
from pipecat.services.deepgram import DeepgramSTTService
from pipecat.services.groq import GroqLLMService
from pipecat.transports.network.websocket_server import (
    WebsocketServerParams,
    WebsocketServerTransport,
)

def read_secret(secret_name):
    secret_path = Path("/etc/secrets") / secret_name
    if secret_path.exists():
        return secret_path.read_text().strip()
    return None

# Read secrets directly
GROQ_API_KEY = read_secret("GROQ_API_KEY")
DEEPGRAM_API_KEY = read_secret("DEEPGRAM_API_KEY")
CARTESIA_API_KEY = read_secret("CARTESIA_API_KEY")

# Set environment variables
if GROQ_API_KEY:
    os.environ["GROQ_API_KEY"] = GROQ_API_KEY
if DEEPGRAM_API_KEY:
    os.environ["DEEPGRAM_API_KEY"] = DEEPGRAM_API_KEY
if CARTESIA_API_KEY:
    os.environ["CARTESIA_API_KEY"] = CARTESIA_API_KEY

def check_required_secrets():
    missing_secrets = []
    for key in ["GROQ_API_KEY", "DEEPGRAM_API_KEY", "CARTESIA_API_KEY"]:
        if not os.getenv(key):
            missing_secrets.append(key)
    
    if missing_secrets:
        raise RuntimeError(f"Missing required secrets: {', '.join(missing_secrets)}")

logger.remove(0)
logger.add(sys.stderr, level="DEBUG")

logger.debug("Starting application...")
# Debug log the environment variables (masking sensitive data)
for key in ["GROQ_API_KEY", "DEEPGRAM_API_KEY", "CARTESIA_API_KEY"]:
    value = os.getenv(key)
    if value:
        masked_value = '*' * (len(value) - 4) + value[-4:]
        logger.debug(f"{key}: {masked_value}")
    else:
        logger.warning(f"{key} not found in environment variables")

async def main():
    # Check for required secrets before starting
    check_required_secrets()

    transport = WebsocketServerTransport(
        host="0.0.0.0",
        port=10000,
        params=WebsocketServerParams(
            audio_out_sample_rate=16000,
            audio_out_enabled=True,
            add_wav_header=True,
            vad_enabled=True,
            vad_analyzer=SileroVADAnalyzer(),
            vad_audio_passthrough=True,
        )
    )

    llm = GroqLLMService(api_key=GROQ_API_KEY, model="llama-3.1-70b-versatile")
    stt = DeepgramSTTService(api_key=DEEPGRAM_API_KEY)
    tts = CartesiaTTSService(
        api_key=CARTESIA_API_KEY,
        voice_id="79a125e8-cd45-4c13-8a67-188112f4dd22",  # British Lady
        sample_rate=16000,
    )

    messages = [
        {
            "role": "system",
            "content": "You are a helpful LLM in a WebRTC call. Your goal is to demonstrate your capabilities in a succinct way. Your output will be converted to audio so don't include special characters in your answers. Respond to what the user said in a creative and helpful way.",
        },
    ]

    context = OpenAILLMContext(messages)
    context_aggregator = llm.create_context_aggregator(context)

    pipeline = Pipeline(
        [
            transport.input(),  # Websocket input from client
            stt,  # Speech-To-Text
            context_aggregator.user(),
            llm,  # LLM
            tts,  # Text-To-Speech
            transport.output(),  # Websocket output to client
            context_aggregator.assistant(),
        ]
    )

    task = PipelineTask(pipeline, params=PipelineParams(allow_interruptions=True))

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        # Kick off the conversation.
        messages.append({"role": "system", "content": "Please introduce yourself to the user."})
        await task.queue_frames([LLMMessagesFrame(messages)])

    runner = PipelineRunner()

    # Check if the platform is Windows and skip signal handler setup if it is
    if platform == "win32":
        logger.warning("Signal handling is not supported on Windows. Skipping signal handler setup.")
    else:
        runner._setup_sigint()

    await runner.run(task)

if __name__ == "__main__":
    asyncio.run(main())
