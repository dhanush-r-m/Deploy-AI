import os
import json
import uuid
from datetime import datetime
from typing import Optional, List, Dict

import boto3
from botocore.exceptions import ClientError
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import google.generativeai as genai

from context import prompt  # your system prompt function

# ---------------------------------------------------
# ENV SETUP
# ---------------------------------------------------
load_dotenv()

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

app = FastAPI()

# ---------------------------------------------------
# CORS
# ---------------------------------------------------
origins = os.getenv("CORS_ORIGINS", "http://localhost:3000").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# ---------------------------------------------------
# MEMORY CONFIG
# ---------------------------------------------------
USE_S3 = os.getenv("USE_S3", "false").lower() == "true"
S3_BUCKET = os.getenv("S3_BUCKET", "")
MEMORY_DIR = os.getenv("MEMORY_DIR", "../memory")

if USE_S3:
    s3_client = boto3.client("s3")

# ---------------------------------------------------
# GEMINI MODEL
# ---------------------------------------------------
model = genai.GenerativeModel("gemini-2.5-flash")

# ---------------------------------------------------
# MODELS
# ---------------------------------------------------
class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None


class ChatResponse(BaseModel):
    response: str
    session_id: str


# ---------------------------------------------------
# MEMORY HELPERS
# ---------------------------------------------------
def get_memory_path(session_id: str) -> str:
    return f"{session_id}.json"


def load_conversation(session_id: str) -> List[Dict]:
    if USE_S3:
        try:
            obj = s3_client.get_object(
                Bucket=S3_BUCKET, Key=get_memory_path(session_id)
            )
            return json.loads(obj["Body"].read().decode("utf-8"))
        except ClientError as e:
            if e.response["Error"]["Code"] == "NoSuchKey":
                return []
            raise
    else:
        os.makedirs(MEMORY_DIR, exist_ok=True)
        path = os.path.join(MEMORY_DIR, get_memory_path(session_id))
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        return []


def save_conversation(session_id: str, messages: List[Dict]):
    if USE_S3:
        s3_client.put_object(
            Bucket=S3_BUCKET,
            Key=get_memory_path(session_id),
            Body=json.dumps(messages, indent=2),
            ContentType="application/json",
        )
    else:
        os.makedirs(MEMORY_DIR, exist_ok=True)
        path = os.path.join(MEMORY_DIR, get_memory_path(session_id))
        with open(path, "w", encoding="utf-8") as f:
            json.dump(messages, f, indent=2, ensure_ascii=False)


# ---------------------------------------------------
# ROUTES
# ---------------------------------------------------
@app.get("/")
async def root():
    return {
        "message": "AI Digital Twin API (Gemini)",
        "memory_enabled": True,
        "storage": "S3" if USE_S3 else "local",
    }


@app.get("/health")
async def health():
    return {"status": "healthy", "use_s3": USE_S3}


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        session_id = request.session_id or str(uuid.uuid4())
        conversation = load_conversation(session_id)

        # ---------------------------------------------------
        # BUILD GEMINI PROMPT
        # ---------------------------------------------------
        system_prompt = prompt()

        history_text = ""
        for msg in conversation[-10:]:
            role = msg["role"].upper()
            history_text += f"{role}: {msg['content']}\n"

        full_prompt = f"""
{system_prompt}

Conversation so far:
{history_text}

USER:
{request.message}

ASSISTANT:
"""

        response = model.generate_content(full_prompt)
        assistant_reply = response.text.strip()

        # ---------------------------------------------------
        # SAVE MEMORY
        # ---------------------------------------------------
        conversation.append(
            {
                "role": "user",
                "content": request.message,
                "timestamp": datetime.utcnow().isoformat(),
            }
        )
        conversation.append(
            {
                "role": "assistant",
                "content": assistant_reply,
                "timestamp": datetime.utcnow().isoformat(),
            }
        )

        save_conversation(session_id, conversation)

        return ChatResponse(
            response=assistant_reply,
            session_id=session_id,
        )

    except Exception as e:
        print("Chat error:", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/conversation/{session_id}")
async def get_conversation(session_id: str):
    try:
        return {
            "session_id": session_id,
            "messages": load_conversation(session_id),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------
# RUN
# ---------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
