import asyncio
import os
import sys
from sys import platform
from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import LLMMessagesFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext

from pipecat.services.elevenlabs import ElevenLabsTTSService
from pipecat.services.deepgram import DeepgramSTTService
from pipecat.services.groq import GroqLLMService

from pipecat.transports.network.websocket_server import (
    WebsocketServerParams,
    WebsocketServerTransport,
)

# Configure logging
logger.remove(0)
logger.add(sys.stderr, level="DEBUG")

def verify_environment():
    """Verify all required environment variables are set"""
    required_vars = ['GROQ_API_KEY', 'DEEPGRAM_API_KEY', 'ELEVENLABS_API_KEY']
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        logger.error(f"Missing required environment variables: {', '.join(missing_vars)}")
        sys.exit(1)
    logger.info("All required environment variables are set")

async def main():
    # Verify environment variables
    verify_environment()
    
    # [1] Configure transport with recommended sample rate (16 kHz) and WAV header
    # to ensure consistent data handling across the pipeline.
    transport = WebsocketServerTransport(
        host="0.0.0.0",
        port=int(os.getenv('PORT', 10000)),
        params=WebsocketServerParams(
            audio_out_sample_rate=16000,
            audio_out_enabled=True,  # Enable outbound audio
            add_wav_header=True,     # Helpful for proper playback
            vad_enabled=True,
            vad_analyzer=SileroVADAnalyzer(),
            vad_audio_passthrough=True,
        )
    )

    # [2] Initialize your services
    llm = GroqLLMService(
        api_key=os.getenv("GROQ_API_KEY"),
        model="llama-3.1-70b-versatile"  # Adjust model as needed
    )
    stt = DeepgramSTTService(api_key=os.getenv("DEEPGRAM_API_KEY"))
    tts = ElevenLabsTTSService(
        api_key=os.getenv("ELEVENLABS_API_KEY"),
        voice_id="cgSgspJ2msm6clMCkdW9",
        output_format="pcm_16000",
        model="eleven_turbo_v2_5"
    )

    # Define initial system message for LLM context
    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful LLM in a WebRTC call. "
                "Respond succinctly and clearly so the voice output is easy to hear."
            ),
        },
    ]

    # Create an aggregator for the LLM context
    context = OpenAILLMContext(messages)
    context_aggregator = llm.create_context_aggregator(context)

    # [3] Build the pipeline
    pipeline = Pipeline(
        [
            transport.input(),             # Receive audio from clients
            stt,                           # Speech-to-text
            context_aggregator.user(),     # Aggregate user messages
            llm,                           # Get LLM response
            tts,                           # Text-to-speech
            transport.output(),            # Send audio back to client
            context_aggregator.assistant() # Aggregate assistant messages
        ]
    )

    # Configure a pipeline task with interruption allowed
    task = PipelineTask(pipeline, params=PipelineParams(allow_interruptions=True))

    # Example event handler: On new client, introduce yourself
    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        intro_msg = [
            {"role": "system", "content": "Please introduce yourself to the user."}
        ]
        await task.queue_frames([LLMMessagesFrame(intro_msg)])

    # Run the pipeline
    runner = PipelineRunner()

    if platform == "win32":
        logger.warning("Signal handling is not supported on Windows.")
    else:
        runner._setup_sigint()

    await runner.run(task)

if __name__ == "__main__":
    asyncio.run(main())
