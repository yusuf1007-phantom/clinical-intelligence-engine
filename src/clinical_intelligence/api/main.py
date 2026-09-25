from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from clinical_intelligence.services.engine import analyze, extract_entities, classify_structure

app = FastAPI(title="Clinical Intelligence Engine", version="1.0.0")

class TextRequest(BaseModel):
    text: str = Field(min_length=1, max_length=12000)

@app.get("/health")
def health():
    return {"status": "ok", "service": "clinical-intelligence-engine"}

@app.post("/entities")
def entities(req: TextRequest):
    return {"entities": extract_entities(req.text)}

@app.post("/classify")
def classify(req: TextRequest):
    result = classify_structure(req.text)
    if result is None:
        raise HTTPException(status_code=503, detail="Set STRUCTURE_MODEL_PATH to a saved fine-tuned model.")
    return result

@app.post("/analyze")
def full_analysis(req: TextRequest):
    return analyze(req.text)
