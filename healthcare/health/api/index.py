import os
from fastapi import FastAPI, Depends  # type: ignore
from fastapi.responses import StreamingResponse  # type: ignore
from pydantic import BaseModel  # type: ignore
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


class Visit(BaseModel):
    patient_name: str
    date_of_visit: str
    notes: str


SYSTEM_PROMPT = """
You are provided with notes written by a doctor from a patient's visit.
Your job is to summarize the visit for the doctor and provide an email.

Reply with exactly three sections with the headings:
### Summary of visit for the doctor's records
### Next steps for the doctor
### Draft of email to patient in patient-friendly language
"""


def user_prompt_for(visit: Visit) -> str:
    return f"""
Create the summary, next steps and draft email for:

Patient Name: {visit.patient_name}
Date of Visit: {visit.date_of_visit}

Notes:
{visit.notes}
"""


@app.post("/api")
def consultation_summary(
    visit: Visit,
    creds: HTTPAuthorizationCredentials = Depends(clerk_guard),
):
    # Authenticated user ID (ready for tracking/audit)
    user_id = creds.decoded["sub"]

    model = genai.GenerativeModel("gemini-2.5-flash")

    full_prompt = f"{SYSTEM_PROMPT}\n\n{user_prompt_for(visit)}"

    def event_stream():
        stream = model.generate_content(
            full_prompt,
            stream=True,
        )

        for chunk in stream:
            if hasattr(chunk, "text") and chunk.text:
                for line in chunk.text.split("\n"):
                    yield f"data: {line}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
    )
