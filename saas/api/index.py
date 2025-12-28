import os
from fastapi import FastAPI, Depends  # type: ignore
from fastapi.responses import StreamingResponse  # type: ignore
from fastapi_clerk_auth import (
    ClerkConfig,
    ClerkHTTPBearer,
    HTTPAuthorizationCredentials,
)  # type: ignore
import google.generativeai as genai  # type: ignore

app = FastAPI()

# ------------------ Clerk Auth ------------------
clerk_config = ClerkConfig(jwks_url=os.getenv("CLERK_JWKS_URL"))
clerk_guard = ClerkHTTPBearer(clerk_config)

# ------------------ Gemini Config ------------------
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

@app.get("/api")
def idea(
    creds: HTTPAuthorizationCredentials = Depends(clerk_guard),
):
    # Authenticated user info (ready for future use)
    user_id = creds.decoded["sub"]

    model = genai.GenerativeModel("gemini-2.5-flash")

    PROMPT = """
Reply with a new business idea for AI Agents.

Formatting rules:
- Use clear headings
- Use sub-headings
- Use bullet points
- Keep it practical and concise
"""

    def event_stream():
        stream = model.generate_content(
            PROMPT,
            stream=True,
        )

        for chunk in stream:
            if hasattr(chunk, "text") and chunk.text:
                lines = chunk.text.split("\n")
                for line in lines:
                    # SSE-compliant format
                    yield f"data: {line}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
    )
