# app.py
import asyncio
import os
import sys
from sys import platform
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

# Configure environment variables
def setup_environment():
    # Try loading from /etc/secrets/.env first (Render production)
    if os.path.exists('/etc/secrets/.env'):
        load_dotenv('/etc/secrets/.env', override=True)
    else:
        # Fall back to local .env file (development)
        load_dotenv(override=True)
    
    # Verify required environment variables
    required_vars = ['GROQ_API_KEY', 'DEEPGRAM_API_KEY', 'CARTESIA_API_KEY']
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        logger.error(f"Missing required environment variables: {', '.join(missing_vars)}")
        sys.exit(1)

# Configure logging
logger.remove(0)
logger.add(sys.stderr, level="DEBUG")

async def main():
    # Set up environment variables
    setup_environment()
    
    transport = WebsocketServerTransport(
        host="0.0.0.0",
        port=int(os.getenv('PORT', 10000)),  # Use PORT from environment or default to 10000
        params=WebsocketServerParams(
            audio_out_sample_rate=16000,
            audio_out_enabled=True,
            add_wav_header=True,
            vad_enabled=True,
            vad_analyzer=SileroVADAnalyzer(),
            vad_audio_passthrough=True,
        )
    )

    llm = GroqLLMService(
        api_key=os.getenv("GROQ_API_KEY"),
        model="llama-3.1-70b-versatile"
    )
    
    stt = DeepgramSTTService(
        api_key=os.getenv("DEEPGRAM_API_KEY")
    )
    
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
            transport.input(),
            stt,
            context_aggregator.user(),
            llm,
            tts,
            transport.output(),
            context_aggregator.assistant(),
        ]
    )

    task = PipelineTask(pipeline, params=PipelineParams(allow_interruptions=True))

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        messages.append({"role": "system", "content": "Please introduce yourself to the user."})
        await task.queue_frames([LLMMessagesFrame(messages)])

    runner = PipelineRunner()
    
    if platform == "win32":
        logger.warning("Signal handling is not supported on Windows. Skipping signal handler setup.")
    else:
        runner._setup_sigint()

    await runner.run(task)

if __name__ == "__main__":
    asyncio.run(main())