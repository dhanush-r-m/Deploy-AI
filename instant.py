from fastapi import FastAPI
from fastapi.responses import HTMLResponse
import google.generativeai as genai
import os

# Configure Gemini
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

app = FastAPI()

@app.get("/", response_class=HTMLResponse)
def instant():
    model = genai.GenerativeModel("gemini-2.5-flash")

    prompt = """
    You are on a website that has just been deployed to production for the first time!
    Please reply with an enthusiastic announcement to welcome visitors to the site,
    explaining that it is live on production for the first time!
    """

    response = model.generate_content(prompt)
    reply = response.text.replace("\n", "<br/>")

    html = f"""
    <html>
        <head>
            <title>Live in an Instant!</title>
        </head>
        <body>
            <p>{reply}</p>
        </body>
    </html>
    """

    return html
