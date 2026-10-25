
import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY is missing from .env")

client = genai.Client(api_key=api_key)

print("Testing Gemini with a tiny prompt...", flush=True)

try:
    response = client.models.generate_content(
        model="gemini-3.5-flash-lite",
        contents="Reply with exactly: Gemini connection successful.",
        config={
            "http_options": {"timeout": 90000},
            "max_output_tokens": 50,
        },
    )
    print("Gemini response:", response.text, flush=True)

except Exception as error:
    print(f"Gemini test failed: {error}", flush=True)
