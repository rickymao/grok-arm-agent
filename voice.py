#!/usr/bin/env python3
"""
Streaming Text-to-Speech Example

This example demonstrates how to use XAI's streaming TTS API to convert text to speech
in real-time using WebSocket connections. The audio is played as it's received and
optionally saved to a file.

API: wss://api.x.ai/v1/realtime/audio/speech
Audio format: PCM linear16, 24kHz, mono
"""

import asyncio
import base64
import json
import os
import sys
import contextlib
import time
import wave
from pathlib import Path
import termios
import tty
import dotenv
import websockets
from dotenv import load_dotenv

# PyAudio is optional - only needed for playback
try:
    import pyaudio
    PYAUDIO_AVAILABLE = True
except ImportError:
    PYAUDIO_AVAILABLE = False
    pyaudio = None


async def streaming_tts(
    text: str,
    voice: str = "ara",
    output_file: str = None,
    play_audio: bool = True,
    sample_rate: int = 24000,
    channels: int = 1,
    sample_width: int = 2,
):
    """
    Stream text-to-speech from XAI API.

    Args:
        text: Text to convert to speech
        voice: Voice ID (ara, rex, sal, eve, una, leo)
        output_file: Optional path to save audio
        play_audio: Whether to play audio in real-time
        sample_rate: Audio sample rate (24000 Hz)
        channels: Number of audio channels (1 for mono)
        sample_width: Sample width in bytes (2 for 16-bit)
    """
    # Get API key
    api_key = os.getenv("XAI_API_KEY")
    if not api_key:
        raise ValueError("XAI_API_KEY not found in environment variables")

    # Get base URL
    base_url = os.getenv("BASE_URL", "https://api.x.ai/v1")
    ws_url = base_url.replace("https://", "wss://").replace("http://", "ws://")
    uri = f"{ws_url}/realtime/audio/speech"

    print(f"🎤 Connecting to {uri}")
    print(f"📝 Voice: {voice}")
    print(f"📄 Text: {text[:50]}{'...' if len(text) > 50 else ''}")

    # Set up headers
    headers = {"Authorization": f"Bearer {api_key}"}

    # Initialize audio playback if needed
    audio_stream = None
    p = None
    if play_audio:
        if not PYAUDIO_AVAILABLE:
            print("⚠️  PyAudio not available - skipping playback")
            print("   Install with: pip install pyaudio")
            play_audio = False
        else:
            p = pyaudio.PyAudio()
            audio_stream = p.open(
                format=pyaudio.paInt16 if sample_width == 2 else pyaudio.paInt32,
                channels=channels,
                rate=sample_rate,
                output=True,
            )

    audio_bytes = b""
    chunk_count = 0
    
    # Timing metrics
    start_time = time.time()
    first_chunk_time = None
    last_chunk_time = None

    try:
        async with websockets.connect(uri, additional_headers=headers) as websocket:
            print("✅ Connected to XAI streaming TTS API")

            # Send config message
            config_message = {"type": "config", "data": {"voice_id": voice}}
            await websocket.send(json.dumps(config_message))
            print(f"📤 Sent config: {config_message}")

            # Send text chunk
            text_message = {
                "type": "text_chunk",
                "data": {"text": text, "is_last": True},
            }
            await websocket.send(json.dumps(text_message))
            request_sent_time = time.time()
            print(f"📤 Sent text chunk")

            # Receive audio chunks
            print("🎵 Receiving and playing audio in real-time...")
            while True:
                try:
                    response = await websocket.recv()
                    data = json.loads(response)

                    # Extract audio data
                    audio_b64 = data["data"]["data"]["audio"]
                    is_last = data["data"]["data"].get("is_last", False)

                    # Decode audio
                    chunk_bytes = base64.b64decode(audio_b64)
                    audio_bytes += chunk_bytes
                    chunk_count += 1
                    
                    # Track timing
                    current_time = time.time()
                    if first_chunk_time is None and len(chunk_bytes) > 0:
                        first_chunk_time = current_time
                        time_to_first_audio = (first_chunk_time - request_sent_time) * 1000
                        print(f"  ⚡ First audio chunk received in {time_to_first_audio:.0f}ms")

                    # Play audio in real-time (streaming playback!)
                    if play_audio and audio_stream and len(chunk_bytes) > 0:
                        await asyncio.to_thread(audio_stream.write, chunk_bytes)

                    print(f"  📦 Chunk {chunk_count}: {len(chunk_bytes)} bytes", end="")
                    if is_last:
                        last_chunk_time = current_time
                        print(" (last)")
                        break
                    else:
                        print()

                except websockets.exceptions.ConnectionClosedOK:
                    print("✅ Connection closed normally")
                    break
                except websockets.exceptions.ConnectionClosedError as e:
                    print(f"❌ Connection closed with error: {e}")
                    break

    finally:
        # Clean up audio playback
        if audio_stream:
            audio_stream.stop_stream()
            audio_stream.close()
        if p:
            p.terminate()

    # Calculate and display metrics
    total_time = time.time() - start_time
    audio_duration = len(audio_bytes) / (sample_rate * channels * sample_width)
    
    print(f"\n✅ Received {chunk_count} audio chunks ({len(audio_bytes)} bytes total)")
    print(f"\n📊 Performance Metrics:")
    if first_chunk_time:
        print(f"   ⚡ Time to first audio: {(first_chunk_time - request_sent_time) * 1000:.0f}ms")
    if last_chunk_time:
        print(f"   ⏱️  Time to last byte: {(last_chunk_time - request_sent_time) * 1000:.0f}ms")
    print(f"   📏 Audio duration: {audio_duration:.2f}s")
    print(f"   🎯 Total time: {total_time:.2f}s")
    if last_chunk_time and audio_duration > 0:
        streaming_ratio = ((last_chunk_time - request_sent_time) / audio_duration) * 100
        print(f"   💡 Streaming efficiency: Generated in {streaming_ratio:.0f}% of playback time")
        if streaming_ratio < 100:
            print(f"   🚀 Audio finished playing BEFORE generation completed (streaming advantage!)")

    # Save to file if requested
    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with wave.open(str(output_path), "wb") as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(sample_width)
            wf.setframerate(sample_rate)
            wf.writeframes(audio_bytes)

        print(f"💾 Saved audio to {output_file}")

    return audio_bytes


def text_to_speech(text: str, voice: str = "ara"):
    """Main entry point."""
    try:
        asyncio.run(
            streaming_tts(
                text=text,
                voice=voice,
            )
        )
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


# PyAudio is optional - only needed for microphone input
try:
    import pyaudio
    PYAUDIO_AVAILABLE = True
except ImportError:
    PYAUDIO_AVAILABLE = False
    pyaudio = None


class StreamingSTT:
    """Streaming Speech-to-Text handler."""

    def __init__(
        self,
        sample_rate: int = 16000,
        channels: int = 1,
        chunk_size: int = 1024,
        enable_interim: bool = True,
    ):
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_size = chunk_size
        self.enable_interim = enable_interim
        self.running = False
        self.final_transcript = ""
        self.current_interim = ""
        self.first_transcript_time = None
        self.stream_start_time = None
        self.transcript_count = 0

    async def stream_audio(
        self,
        duration_seconds: float | None = None,
        press_to_talk: bool = True,
        toggle_key: str = " ",
        exit_key: str = "q",
    ) -> str:
        """Stream audio from microphone to XAI API.

        If duration_seconds is provided, the stream will automatically stop after
        the specified number of seconds and the accumulated final transcript will
        be returned.
        If press_to_talk is True, the spacebar toggles recording on/off and 'q'
        exits and returns the accumulated transcript.
        """
        # Check if PyAudio is available
        if not PYAUDIO_AVAILABLE:
            print("❌ PyAudio is not installed")
            print("   Install with: pip install pyaudio")
            print("   Or use the Node.js version which uses ffmpeg")
            return

        # Get API key
        api_key = os.getenv("XAI_API_KEY")
        if not api_key:
            raise ValueError("XAI_API_KEY not found in environment variables")

        # Get base URL
        base_url = os.getenv("BASE_URL", "https://api.x.ai/v1")
        ws_url = base_url.replace("https://", "wss://").replace("http://", "ws://")
        uri = f"{ws_url}/realtime/audio/transcriptions"

        print(f"🎤 Connecting to {uri}")
        print(f"📊 Sample rate: {self.sample_rate} Hz")
        print(f"🎵 Channels: {self.channels}")
        print(f"📦 Chunk size: {self.chunk_size}")
        print(f"⏱️  Interim results: {'enabled' if self.enable_interim else 'disabled'}")

        # Set up headers
        headers = {"Authorization": f"Bearer {api_key}"}

        # Initialize PyAudio
        p = pyaudio.PyAudio()
        stream = p.open(
            format=pyaudio.paInt16,
            channels=self.channels,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.chunk_size,
        )

        print("✅ Microphone ready")

        # Track press-to-talk mode state and terminal settings
        self._press_to_talk = press_to_talk
        self.talking_enabled = False
        old_term_settings = None

        try:
            async with websockets.connect(uri, additional_headers=headers) as websocket:
                print("✅ Connected to XAI streaming STT API")
                if press_to_talk:
                    print("\n🎙️  Press SPACE to toggle talking, 'q' to finish.\n")
                else:
                    print("\n🎙️  Speak now... (Press Ctrl+C to stop)\n")

                # Send config message
                config_message = {
                    "type": "config",
                    "data": {
                        "encoding": "linear16",
                        "sample_rate_hertz": self.sample_rate,
                        "enable_interim_results": self.enable_interim,
                    },
                }
                await websocket.send(json.dumps(config_message))
                print(f"📤 Sent config")

                self.running = True
                self.stream_start_time = time.time()

                # Create tasks for sending and receiving
                send_task = asyncio.create_task(self._send_audio(websocket, stream))
                recv_task = asyncio.create_task(self._receive_transcripts(websocket))
                stopper_task = None
                keyboard_task = None

                # Optional: press-to-talk keyboard listener
                if press_to_talk:
                    # put stdin into cbreak mode for single-character reads
                    try:
                        old_term_settings = termios.tcgetattr(sys.stdin.fileno())
                        tty.setcbreak(sys.stdin.fileno())
                    except Exception:
                        old_term_settings = None
                    keyboard_task = asyncio.create_task(
                        self._keypress_listener(websocket, toggle_key=toggle_key, exit_key=exit_key)
                    )

                # Optional: stop automatically after duration_seconds
                if duration_seconds is not None and duration_seconds > 0:
                    async def stop_after_delay():
                        try:
                            await asyncio.sleep(duration_seconds)
                        finally:
                            # Signal tasks to stop and close websocket
                            self.running = False
                            try:
                                await websocket.close()
                            except Exception:
                                pass

                    stopper_task = asyncio.create_task(stop_after_delay())

                # Wait for both tasks
                try:
                    tasks = [send_task, recv_task]
                    if keyboard_task is not None:
                        tasks.append(keyboard_task)
                    await asyncio.gather(*tasks)
                finally:
                    if stopper_task is not None and not stopper_task.done():
                        stopper_task.cancel()
                        with contextlib.suppress(Exception):
                            await stopper_task
                    if keyboard_task is not None and not keyboard_task.done():
                        keyboard_task.cancel()
                        with contextlib.suppress(Exception):
                            await keyboard_task

        except KeyboardInterrupt:
            print("\n\n⚠️  Interrupted by user")
        except Exception as e:
            print(f"\n❌ Error: {e}")
            raise
        finally:
            self.running = False
            stream.stop_stream()
            stream.close()
            p.terminate()
            print("\n✅ Microphone closed")
            # restore terminal mode
            if old_term_settings is not None:
                with contextlib.suppress(Exception):
                    termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, old_term_settings)

            if self.final_transcript:
                print(f"\n📝 Final transcript:\n{self.final_transcript}")
                
            # Display metrics
            if self.stream_start_time:
                total_time = time.time() - self.stream_start_time
                print(f"\n📊 Performance Metrics:")
                if self.first_transcript_time:
                    time_to_first = (self.first_transcript_time - self.stream_start_time) * 1000
                    print(f"   ⚡ Time to first transcript: {time_to_first:.0f}ms")
                print(f"   📏 Total transcripts: {self.transcript_count}")
                print(f"   ⏱️  Total recording time: {total_time:.1f}s")
                if self.transcript_count > 0:
                    print(f"   🎯 Real-time transcription: Transcripts received WHILE speaking")
        # Return the collected final transcript string
        return self.final_transcript

    async def _send_audio(self, websocket, stream):
        """Send audio chunks to the WebSocket."""
        chunk_count = 0
        try:
            while self.running:
                # Read audio chunk
                audio_data = await asyncio.to_thread(stream.read, self.chunk_size, exception_on_overflow=False)

                # Only send when not in press-to-talk mode or when talking is enabled
                if not getattr(self, "_press_to_talk", False) or getattr(self, "talking_enabled", False):
                    # Convert to base64
                    audio_b64 = base64.b64encode(audio_data).decode("utf-8")
                    # Send audio message
                    audio_message = {
                        "type": "audio",
                        "data": {"audio": audio_b64},
                    }
                    await websocket.send(json.dumps(audio_message))

                chunk_count += 1
                if chunk_count % 50 == 0:  # Log every 50 chunks
                    print(f"  📤 Sent {chunk_count} audio chunks...", end="\r")

        except Exception as e:
            if self.running:
                print(f"\n❌ Error sending audio: {e}")
                self.running = False

    async def _receive_transcripts(self, websocket):
        """Receive and display transcripts from the WebSocket."""
        try:
            while self.running:
                response = await websocket.recv()
                data = json.loads(response)

                # Check if it's a transcript
                if data.get("data", {}).get("type") == "speech_recognized":
                    transcript_data = data["data"]["data"]
                    transcript = transcript_data.get("transcript", "")
                    is_final = transcript_data.get("is_final", False)
                    
                    # Track time to first transcript
                    if self.first_transcript_time is None and transcript:
                        self.first_transcript_time = time.time()
                        elapsed = (self.first_transcript_time - self.stream_start_time) * 1000
                        print(f"\r⚡ First transcript received in {elapsed:.0f}ms")

                    if is_final:
                        # Final transcript
                        self.final_transcript += transcript + " "
                        self.current_interim = ""
                        self.transcript_count += 1
                        elapsed = (time.time() - self.stream_start_time) * 1000
                        print(f"\r✅ [{elapsed:.0f}ms] {transcript}")
                    else:
                        # Interim transcript
                        self.current_interim = transcript
                        elapsed = (time.time() - self.stream_start_time) * 1000
                        print(f"\r💭 [{elapsed:.0f}ms] {transcript}", end="", flush=True)

        except websockets.exceptions.ConnectionClosedOK:
            print("\n✅ Connection closed normally")
        except websockets.exceptions.ConnectionClosedError as e:
            print(f"\n❌ Connection closed with error: {e}")
        except Exception as e:
            if self.running:
                print(f"\n❌ Error receiving transcripts: {e}")
                self.running = False

    async def _keypress_listener(self, websocket, toggle_key: str = " ", exit_key: str = "q"):
        """Listen for single-key presses to control talking state and exit."""
        try:
            while self.running:
                try:
                    ch = await asyncio.to_thread(sys.stdin.read, 1)
                except Exception:
                    break
                if not ch:
                    continue
                if ch == toggle_key:
                    self.talking_enabled = not self.talking_enabled
                    state_label = "Recording" if self.talking_enabled else "Paused"
                    print(f"\r🎙️  {state_label}             ", end="", flush=True)
                elif ch == exit_key:
                    # Stop the session and close the websocket
                    self.running = False
                    with contextlib.suppress(Exception):
                        await websocket.close()
                    break
        except Exception:
            # Ignore keyboard errors; fall back to normal flow
            pass


def speech_to_text():
    """Main entry point."""
    dotenv.load_dotenv()
    stt = StreamingSTT(
        sample_rate=16000,
        channels=1,
        chunk_size=1024,
        enable_interim=False,
    )

    try:
        asyncio.run(stt.stream_audio())
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


def transcribe_for(seconds: float = 5.0) -> str:
    """Capture microphone audio for `seconds` and return the final transcript."""
    load_dotenv()
    stt = StreamingSTT(
        sample_rate=16000,
        channels=1,
        chunk_size=1024,
        enable_interim=False,
    )
    try:
        return asyncio.run(stt.stream_audio(duration_seconds=seconds))
    except KeyboardInterrupt:
        # Gracefully handle user interrupt; return whatever was collected
        return stt.final_transcript
    except Exception:
        # On error, return what we have so far to the caller
        return stt.final_transcript


def press_to_talk(toggle_key: str = " ", exit_key: str = "q") -> str:
    """Run press-to-talk session; SPACE toggles talking, 'q' finishes and returns transcript."""
    load_dotenv()
    stt = StreamingSTT(
        sample_rate=16000,
        channels=1,
        chunk_size=1024,
        enable_interim=False,
    )
    try:
        return asyncio.run(stt.stream_audio(press_to_talk=True, toggle_key=toggle_key, exit_key=exit_key))
    except KeyboardInterrupt:
        return stt.final_transcript
    except Exception:
        return stt.final_transcript