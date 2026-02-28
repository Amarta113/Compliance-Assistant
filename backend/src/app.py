from fastapi import FastAPI
from fastapi.responses import JSONResponse

app = FastAPI(
    title="RegBot",
    description="A chatbot for supporting researchers.",
    version="1.0.0",)

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/chat")
def chat(message: str):
    return JSONResponse(content={"my message", message}, status_code=200)

