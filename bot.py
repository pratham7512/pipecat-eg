import os
import asyncio
import uuid
import json
from loguru import logger
from google.protobuf import message
import dotenv
from pathlib import Path

# aiohttp for HTTP endpoints
from aiohttp import web

# aiortc for self-hosted WebRTC
from aiortc import RTCPeerConnection, RTCSessionDescription, MediaStreamTrack
from aiortc.contrib.media import MediaBlackhole  # optional
from aiortc.contrib.signaling import BYE

# Pipecat imports
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.task import PipelineTask
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams
from pipecat.services.deepgram import DeepgramSTTService
from pipecat.services.elevenlabs import ElevenLabsTTSService
from pipecat.services.groq import GroqLLMService
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.frames.frames import AudioFrame, LLMMessagesFrame

# -----------
# 1. CUSTOM TRANSPORT TO REPLACE pipecat.transports.memory.memory_transport
#    If pipecat.transports.memory.memory_transport is missing, we define it ourselves:
# -----------

from typing import List, Optional, Callable
import threading
import queue

class MemoryTransport:
    """
    Minimal custom transport that Pipecat can call for audio input/output.
    
    - Input: We store inbound PCM frames from WebRTC. Pipecat pulls them in pipeline.
    - Output: Pipecat calls `on_audio_out` with PCM data, 
      and we store or forward them to the WebRTC side.

    This is not production-ready: it just demonstrates the concept of an "in-memory" 
    transport bridging aiortc with Pipecat.
    """
    def __init__(self):
        # Audio input queue (PCM data from WebRTC side => pipeline)
        self.input_audio_queue = queue.Queue()
        # Handler to be called when Pipecat has TTS audio (PCM) for output
        self._audio_out_callback: Optional[Callable[[bytes], None]] = None

    # Pipecat pipeline calls input(). We produce frames from input_audio_queue.
    async def input(self, frame_receiver):
        while True:
            # Wait for next PCM chunk
            pcm_bytes = self.input_audio_queue.get()
            if pcm_bytes is None:
                break  # Means we want to end
            # Wrap in Pipecat AudioFrame
            frame = AudioFrame(audio=pcm_bytes, sample_rate=16000, num_channels=1)
            await frame_receiver(frame)

    # Pipecat pipeline calls output() with TTS audio frames for the user
    async def output(self, frame, frame_sender):
        if frame.type == "audio":
            # If we have a callback set, forward the TTS audio to it
            if self._audio_out_callback:
                self._audio_out_callback(frame.audio)
        await frame_sender(frame)

    # Called from aiortc->server side when we have newly decoded PCM to supply
    def feed_audio_in(self, pcm_bytes: bytes):
        """Push PCM data into the pipeline."""
        self.input_audio_queue.put(pcm_bytes)

    # Called from pipeline->aiortc side to set callback for TTS PCM
    def register_audio_out_callback(self, cb: Callable[[bytes], None]):
        self._audio_out_callback = cb

    def close(self):
        """Signal to pipeline that it should stop pulling input."""
        self.input_audio_queue.put(None)


# -----------
# 2. ROOM / CONNECTION MANAGEMENT
# -----------
rooms = {}  # room_id -> { "peers": set(RTCPeerConnection), "transport": MemoryTransport, "pipeline_task": PipelineTask }

# Load environment variables
dotenv.load_dotenv()

# Add error handling for missing environment variables
def get_required_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise ValueError(f"Missing required environment variable: {name}")
    return value

def build_pipeline() -> tuple[PipelineTask, MemoryTransport]:
    """
    Create a Pipecat pipeline that does:
      - (MemoryTransport) => STT => LLM => TTS => (MemoryTransport)
    """
    # Environment variables for your external services (set these before running!)
    # e.g. export GROQ_API_KEY=xxxx
    #      export DEEPGRAM_API_KEY=xxxx
    #      export ELEVENLABS_API_KEY=xxxx
    groq_api_key = get_required_env("GROQ_API_KEY")
    deepgram_api_key = get_required_env("DEEPGRAM_API_KEY")
    elevenlabs_api_key = get_required_env("ELEVENLABS_API_KEY")

    # Services
    llm = GroqLLMService(api_key=groq_api_key, model="llama-3.1-70b-versatile")
    stt = DeepgramSTTService(api_key=deepgram_api_key)
    tts = ElevenLabsTTSService(
        api_key=elevenlabs_api_key,
        voice_id="cgSgspJ2msm6clMCkdW9",
        output_format="pcm_16000",
        model="eleven_turbo_v2_5"
    )

    # Context aggregator for conversation
    init_messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful AI assistant, interacting with the user via voice. "
                "Keep responses concise and clear. Avoid special characters or code blocks."
            )
        },
    ]
    context = OpenAILLMContext(init_messages)
    context_aggregator = llm.create_context_aggregator(context)

    # Our custom in-memory transport
    transport = MemoryTransport()

    pipeline = Pipeline([
        transport.input,
        stt,
        context_aggregator.user(),
        llm,
        tts,
        transport.output,
        context_aggregator.assistant(),
    ])

    # Create the pipeline task
    task = PipelineTask(
        pipeline,
        params=PipelineParams(allow_interruptions=True)
    )

    return task, transport


# -----------
# 3. AIORTC AUDIO TRACKS
# -----------
# We need to handle inbound Opus audio from the browser, decode to PCM, feed it to 
# MemoryTransport, then re-encode TTS PCM as Opus to send back. 
#
# For brevity, we are not showing the actual PCM <-> Opus conversion here. 
# Instead, we just represent how you'd route it.
# In real production code, you can extend aiortc to decode audio frames to raw PCM 
# with a custom MediaRecorder or manually bridging the raw frames.

class InboundAudioTrack(MediaStreamTrack):
    kind = "audio"

    def __init__(self, transport: MemoryTransport):
        super().__init__()
        self._transport = transport

    async def recv(self):
        # The parent call to super().recv() returns an AudioFrame in raw 
        # or partially-decoded form, depending on aiortc’s chain.  
        frame = await super().recv()

        # You can attempt .to_ndarray() if it's supported, or you'd do your own
        # PCM decoding. For demonstration, we use stub “fake” PCM.
        # A real pipeline would convert `frame` from Opus to PCM properly.
        fake_pcm_16khz_mono = b"\x00" * 3200  # e.g. 100ms of silence as dummy

        # Feed into MemoryTransport
        self._transport.feed_audio_in(fake_pcm_16khz_mono)

        return frame  # Just pass the original up the chain (for further mixing, etc.)


# -----------
# 4. WEB HANDLERS / SIGNALING
# -----------
async def index(request):
    """Serve index.html for WebRTC client."""
    return web.FileResponse("./static/index.html")


async def create_or_join_room(request):
    data = await request.json()
    room_id = data.get("roomId")
    if not room_id:
        room_id = str(uuid.uuid4())  # create new room if blank

    if room_id not in rooms:
        # Create new pipeline
        logger.info(f"Creating new room {room_id}")
        pipeline_task, memory_transport = build_pipeline()
        rooms[room_id] = {
            "peers": set(),
            "transport": memory_transport,
            "pipeline_task": pipeline_task
        }
        # Start Pipecat pipeline in background
        runner = PipelineRunner()
        asyncio.ensure_future(runner.run(pipeline_task))

    return web.json_response({"roomId": room_id})


async def offer(request):
    """
    Receive { roomId, sdp, type } from the client. 
    Create PeerConnection, set remote desc, produce an answer.
    """
    body = await request.json()
    room_id = body["roomId"]
    sdp = body["sdp"]
    sdp_type = body["type"]

    if room_id not in rooms:
        return web.json_response({"error": "Room not found"}, status=404)

    pc = RTCPeerConnection()
    rooms[room_id]["peers"].add(pc)

    # For demonstration, we route inbound audio via InboundAudioTrack
    @pc.on("track")
    def on_track(track):
        if track.kind == "audio":
            logger.info("Got inbound audio track; bridging to MemoryTransport")
            inbound_track = InboundAudioTrack(rooms[room_id]["transport"])
            # If you strictly need to “play” the inbound track somewhere (like a blackhole),
            # you can attach a recorder or MediaBlackhole:
            # blackhole = MediaBlackhole()
            # blackhole.addTrack(track)
            # or pass track to inbound_track if you want to keep it in pipeline
            # for simplicity we won't forward the track’s frames back to the user

    @pc.on("iceconnectionstatechange")
    def on_ice_state_change():
        if pc.iceConnectionState == "disconnected":
            logger.info("Peer disconnected from room %s", room_id)

    # 1) Set remote description
    await pc.setRemoteDescription(RTCSessionDescription(sdp, sdp_type))
    # 2) Create answer
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    # ### Sending TTS Audio Back to the Browser
    # We need an outbound track that we feed from Pipecat TTS PCM -> Opus frames
    # For demonstration, let's create a dummy audio track that does nothing:
    # Real code: you'd implement something like a "TTSOutboundAudioTrack" with 
    # a custom `recv()` that yields frames from pipeline TTS. Then,
    # pc.addTrack(MyTTSOutboundTrack(...)).
    # 
    # We'll handle the bridging with a callback from Pipecat:
    async def on_tts_audio_out(pcm_bytes: bytes):
        """
        This callback is triggered by MemoryTransport when TTS audio appears. 
        We must convert PCM -> Opus-coded frames for WebRTC.
        
        In production, implement an aiortc MediaStreamTrack that pulls from a buffer 
        and yields frames in real-time. 
        """
        # Stub: do nothing
        pass

    # Tie the callback in
    ttransport = rooms[room_id]["transport"]
    ttransport.register_audio_out_callback(lambda pcm: asyncio.ensure_future(on_tts_audio_out(pcm)))

    return web.json_response({
        "sdp": pc.localDescription.sdp,
        "type": pc.localDescription.type,
    })


async def add_ice_candidate(request):
    """
    Add ICE candidate from client to the peer connection.
    For simplicity, we only track one PC per room in this example.
    """
    body = await request.json()
    room_id = body["roomId"]
    candidate = body["candidate"]
    sdpMid = body["sdpMid"]
    sdpMLineIndex = body["sdpMLineIndex"]
    if room_id not in rooms or not rooms[room_id]["peers"]:
        return web.json_response({"error": "Room not found or no peers"}, status=404)

    # In a real multi-peer environment, you'd store PCs by user
    pc = next(iter(rooms[room_id]["peers"]))  # first PC in this POC
    ice_candidate = {
        "candidate": candidate,
        "sdpMid": sdpMid,
        "sdpMLineIndex": sdpMLineIndex,
    }
    if candidate:
        await pc.addIceCandidate(ice_candidate)
    return web.json_response({"status": "ok"})


async def cleanup(request):
    """
    Graceful room cleanup
    """
    data = await request.json()
    room_id = data["roomId"]
    if room_id in rooms:
        # close peer connections
        for pc in rooms[room_id]["peers"]:
            await pc.close()
        # close pipeline transport
        rooms[room_id]["transport"].close()
        del rooms[room_id]
        return web.json_response({"status": f"Room {room_id} cleaned up."})
    else:
        return web.json_response({"error": "Room not found"}, status=404)


# Add health check endpoint
async def healthcheck(request):
    return web.Response(text="OK")


def main():
    logger.add("server.log", level="DEBUG")

    # Ensure static directory exists
    Path("./static").mkdir(exist_ok=True)
    
    app = web.Application()
    app.router.add_get("/health", healthcheck)
    app.router.add_get("/", index)
    app.router.add_post("/createOrJoinRoom", create_or_join_room)
    app.router.add_post("/offer", offer)
    app.router.add_post("/candidate", add_ice_candidate)
    app.router.add_post("/cleanup", cleanup)

    # Serve static/ as our root dir
    app.router.add_static("/", "./static")

    port = int(os.getenv("PORT", "8080"))
    host = os.getenv("HOST", "0.0.0.0")
    
    web.run_app(app, host=host, port=port)


if __name__ == "__main__":
    main()
