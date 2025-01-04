#
# Copyright (c) 2024, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import os
import sys
from sys import platform  # Import platform to check OS

from dotenv import load_dotenv
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




# Load environment variables
load_dotenv("/etc/secrets/.env")

# Debug log the environment variables (masking sensitive data)
for key in ["GROQ_API_KEY", "DEEPGRAM_API_KEY", "CARTESIA_API_KEY"]:
    value = os.getenv(key)
    if value:
        masked_value = '*' * (len(value) - 4) + value[-4:]
        logger.debug(f"{key}: {masked_value}")
    else:
        logger.warning(f"{key} not found in environment variables")

logger.remove(0)
logger.add(sys.stderr, level="DEBUG")

logger.debug("Starting application...")
logger.debug(f"GROQ API Key: {'*' * 4 + os.environ.get('GROQ_API_KEY')[-4:] if os.environ.get('GROQ_API_KEY') else 'Not found'}")
logger.debug(f"DEEPGRAM API Key: {'*' * 4 + os.environ.get('DEEPGRAM_API_KEY')[-4:] if os.environ.get('DEEPGRAM_API_KEY') else 'Not found'}")
logger.debug(f"CARTESIA API Key: {'*' * 4 + os.environ.get('CARTESIA_API_KEY')[-4:] if os.environ.get('CARTESIA_API_KEY') else 'Not found'}")

async def main():
    transport = WebsocketServerTransport(
        host="0.0.0.0",  # Add this line to bind to all interfaces
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
    print(f"this is groq api key = {os.getenv('GROQ_API_KEY')}")
    llm = GroqLLMService(api_key=os.getenv("GROQ_API_KEY"), model="llama-3.1-70b-versatile")

    stt = DeepgramSTTService(api_key=os.getenv("DEEPGRAM_API_KEY"))

    tts = CartesiaTTSService(
        api_key=os.getenv("CARTESIA_API_KEY"),
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
        runner._setup_sigint()  # Call this only if not on Windows

    await runner.run(task)


if __name__ == "__main__":
    asyncio.run(main())
