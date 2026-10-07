from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import uvicorn
import logging
import threading
import fitz  # PyMuPDF

# Initialize logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("server")

app = FastAPI()
model_lock = threading.Lock()

# Mount static files (HTML, JS, CSS)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Load the SpaceText Agent
logger.info("Loading SpaceText Agent into memory...")
try:
    from spacetext.rag.agent import SpaceTextAgent
    agent = SpaceTextAgent(
        model_path="./models/param-1-2.9b-instruct-q8_0.gguf", 
        db_path="./aerospace_taxonomy.adk"
    )
    logger.info("Agent loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load agent: {e}")
    agent = None

class QueryRequest(BaseModel):
    query: str

@app.post("/api/summarize")
def summarize(request: QueryRequest):
    if agent is None:
        raise HTTPException(status_code=500, detail="Backend agent failed to load.")
    
    logger.info(f"Received query: {request.query}")
    try:
        # Prevent C++ segfaults by ensuring only one generation runs at a time
        with model_lock:
            result = agent.summarize(topic=request.query)
        return result
    except Exception as e:
        logger.error(f"Summarization error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    if agent is None:
        raise HTTPException(status_code=500, detail="Backend agent failed to load.")
    
    logger.info(f"Received file upload: {file.filename}")
    try:
        contents = await file.read()
        
        # Parse based on extension
        text = ""
        if file.filename.lower().endswith(".pdf"):
            # Use PyMuPDF to extract text
            doc = fitz.open(stream=contents, filetype="pdf")
            for page in doc:
                text += page.get_text()
        elif file.filename.lower().endswith(".txt"):
            text = contents.decode("utf-8")
        else:
            raise HTTPException(status_code=400, detail="Only PDF and TXT files are supported right now.")
            
        if not text.strip():
            raise HTTPException(status_code=400, detail="Could not extract any text from the file.")

        logger.info(f"Extracted {len(text)} characters. Summarizing...")
        
        # Use the bypass agent method
        with model_lock:
            result = agent.summarize_document(text=text, filename=file.filename)
            
        return result
    except Exception as e:
        logger.error(f"File upload error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error(f"Summarization error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def read_root():
    return FileResponse("static/index.html")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
