import os
import json
import uuid
from pathlib import Path
from typing import Optional, List, Dict

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import google.generativeai as genai

# --------------------------------------------------
# Load environment variables
# --------------------------------------------------
load_dotenv(override=True)

app = FastAPI()

# --------------------------------------------------
# CORS
# --------------------------------------------------
origins = os.getenv("CORS_ORIGINS", "http://localhost:3000").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------------------------------
# Gemini Configuration
# --------------------------------------------------
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# --------------------------------------------------
# Memory directory
# --------------------------------------------------
MEMORY_DIR = Path("../memory")
MEMORY_DIR.mkdir(exist_ok=True)

# --------------------------------------------------
# Load personality
# --------------------------------------------------
def load_personality() -> str:
    with open("me.txt", "r", encoding="utf-8") as f:
        return f.read().strip()

PERSONALITY = load_personality()

# --------------------------------------------------
# Memory helpers
# --------------------------------------------------
def load_conversation(session_id: str) -> List[Dict]:
    file_path = MEMORY_DIR / f"{session_id}.json"
    if file_path.exists():
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def save_conversation(session_id: str, messages: List[Dict]):
    file_path = MEMORY_DIR / f"{session_id}.json"
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(messages, f, indent=2, ensure_ascii=False)

# --------------------------------------------------
# Models
# --------------------------------------------------
class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None


class ChatResponse(BaseModel):
    response: str
    session_id: str

# --------------------------------------------------
# Routes
# --------------------------------------------------
@app.get("/")
async def root():
    return {"message": "AI Digital Twin API with Memory (Gemini)"}


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        session_id = request.session_id or str(uuid.uuid4())

        # Load past conversation
        conversation = load_conversation(session_id)

        # Build prompt (Gemini does not support role-based messages)
        history_text = ""
        for msg in conversation:
            prefix = "User" if msg["role"] == "user" else "Assistant"
            history_text += f"{prefix}: {msg['content']}\n"

        full_prompt = f"""
{PERSONALITY}

Conversation so far:
{history_text}

User:
{request.message}

Assistant:
"""

        model = genai.GenerativeModel("gemini-2.5-flash")
        response = model.generate_content(full_prompt)

        assistant_response = response.text.strip()

        # Update memory
        conversation.append({"role": "user", "content": request.message})
        conversation.append({"role": "assistant", "content": assistant_response})
        save_conversation(session_id, conversation)

        return ChatResponse(
            response=assistant_response,
            session_id=session_id
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/sessions")
async def list_sessions():
    sessions = []
    for file_path in MEMORY_DIR.glob("*.json"):
        with open(file_path, "r", encoding="utf-8") as f:
            conversation = json.load(f)

        sessions.append({
            "session_id": file_path.stem,
            "message_count": len(conversation),
            "last_message": conversation[-1]["content"] if conversation else None,
        })

    return {"sessions": sessions}


# --------------------------------------------------
# Local run
# --------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
