from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class Frame(BaseModel):
    frameData: str

@app.post("/upload-frame")
async def upload_frame(frame: Frame):
    # Process the frame (save it, perform operations, etc.)
    # For now, just return success
    return {"message": "Frame received successfully"}