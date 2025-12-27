from fastapi import FastAPI  # type: ignore
from fastapi.responses import StreamingResponse  # type: ignore
import google.generativeai as genai  # type: ignore
import os

app = FastAPI()

# Configure Gemini API
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

@app.get("/api")
def idea():
    model = genai.GenerativeModel("gemini-2.5-flash")

    PROMPT = """
Reply with a new business idea for AI Agents.

Format the response using:
- Clear headings
- Sub-headings
- Bullet points

Keep it concise and practical.
"""

    def event_stream():
        stream = model.generate_content(
            PROMPT,
            stream=True
        )

        for chunk in stream:
            if hasattr(chunk, "text") and chunk.text:
                for line in chunk.text.split("\n"):
                    yield f"data: {line}\n"
                yield "\n"  # Required SSE separator

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream"
    )
