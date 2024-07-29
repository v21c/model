from pathlib import Path
from openai import OpenAI
from openai_api_key import OPENAI_API_KEY

client = OpenAI(api_key=OPENAI_API_KEY)
speech_file_path = Path(__file__).parent / "speech_tts.mp3"

with client.audio.speech.with_streaming_response.create(
    model="tts-1",
    voice="alloy",
    input="안녕하세요 저는 한국어를 잘해요",
) as response:
    response.stream_to_file("speech.mp3")

