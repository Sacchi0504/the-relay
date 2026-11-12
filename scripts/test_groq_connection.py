"""Verify real Groq connection, query available models, and measure baseline latency."""

import asyncio
import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from groq import AsyncGroq

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.config import settings
from app.core.models import ModelRequest
from app.providers.groq_provider import GroqProvider

load_dotenv()


async def test_connection():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("[ERROR] GROQ_API_KEY is not set.")
        return

    print("[INFO] Testing Groq API authentication...")
    client = AsyncGroq(api_key=api_key)

    try:
        models_response = await client.models.list()
        model_ids = [m.id for m in models_response.data]
        print(f"[SUCCESS] Authenticated successfully with Groq!")
        print(f"[INFO] Total available models: {len(model_ids)}")

        # Find recommended models
        popular_models = [m for m in model_ids if "llama-3" in m or "mixtral" in m]
        print("[INFO] Selected supported models:")
        for m in sorted(popular_models)[:8]:
            print(f"  * {m}")

        selected_model = (
            "llama-3.3-70b-versatile"
            if "llama-3.3-70b-versatile" in model_ids
            else ("llama-3.1-8b-instant" if "llama-3.1-8b-instant" in model_ids else model_ids[0])
        )
        print(f"\n[INFO] Running live inference probe with model: {selected_model} ...")

        provider = GroqProvider(api_key=api_key, default_model=selected_model)
        request = ModelRequest(
            request_id="groq_probe_01",
            messages=[
                {"role": "system", "content": "You are a concise research assistant."},
                {"role": "user", "content": "Explain the scaled dot-product attention in one sentence with a citation."},
            ],
            model=selected_model,
            max_tokens=100,
        )

        resp = await provider.generate(request)
        print(f"[STATUS]: {resp.completion_status.value.upper()}")
        print(f"[RESPONSE]:\n{resp.content.strip()}")
        print(f"[MEASUREMENTS]:")
        print(f"  * Time to First Token (TTFT): {resp.first_token_ms:.1f} ms")
        print(f"  * Total Latency: {resp.total_ms:.1f} ms")
        print(f"  * Tokens In: {resp.tokens_in}")
        print(f"  * Tokens Out: {resp.tokens_out}")
        print(f"  * Generation Speed: {(resp.tokens_out / (resp.total_ms / 1000.0)):.1f} tokens/sec")

    except Exception as e:
        print(f"[ERROR] Groq API call failed: {e}")


if __name__ == "__main__":
    asyncio.run(test_connection())

