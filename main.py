import os
from typing import List, Optional, Any, Dict
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from database import create_document, get_documents, db

app = FastAPI(title="Prompt Generator API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class GeneratePromptRequest(BaseModel):
    title: Optional[str] = Field(None, description="Short label for this prompt")
    goal: str = Field(..., description="What the user wants to achieve")
    context: Optional[str] = Field(None, description="Extra background or constraints")
    audience: Optional[str] = Field(None, description="Target audience persona")
    tone: Optional[str] = Field(None, description="e.g., Friendly, Professional, Playful")
    style: Optional[str] = Field(None, description="e.g., Step-by-step, Analytical, Storytelling")
    format: Optional[str] = Field(None, description="e.g., Markdown list, JSON, Table")
    language: Optional[str] = Field("English")
    length: Optional[str] = Field(None, description="e.g., Short, Medium, Detailed")
    include_examples: bool = Field(False)
    variables: Optional[List[str]] = Field(default_factory=list, description="Variables like {product}, {audience}")
    tags: Optional[List[str]] = Field(default_factory=list)


class SavePromptRequest(BaseModel):
    title: str
    prompt: str
    use_case: Optional[str] = None
    tags: Optional[List[str]] = Field(default_factory=list)


@app.get("/")
def read_root():
    return {"message": "Prompt Generator Backend is running"}


@app.get("/test")
def test_database():
    response: Dict[str, Any] = {
        "backend": "✅ Running",
        "database": "❌ Not Available",
        "database_url": None,
        "database_name": None,
        "connection_status": "Not Connected",
        "collections": [],
    }
    try:
        if db is not None:
            response["database"] = "✅ Available"
            response["database_url"] = "✅ Set" if os.getenv("DATABASE_URL") else "❌ Not Set"
            response["database_name"] = db.name if hasattr(db, "name") else "✅ Connected"
            response["connection_status"] = "Connected"
            try:
                collections = db.list_collection_names()
                response["collections"] = collections[:10]
                response["database"] = "✅ Connected & Working"
            except Exception as e:  # pragma: no cover - safety
                response["database"] = f"⚠️  Connected but Error: {str(e)[:50]}"
        else:
            response["database"] = "⚠️  Available but not initialized"
    except Exception as e:
        response["database"] = f"❌ Error: {str(e)[:50]}"

    response["database_url"] = "✅ Set" if os.getenv("DATABASE_URL") else "❌ Not Set"
    response["database_name"] = "✅ Set" if os.getenv("DATABASE_NAME") else "❌ Not Set"
    return response


@app.post("/api/generate")
def generate_prompt(payload: GeneratePromptRequest):
    goal = payload.goal.strip()
    if not goal:
        raise HTTPException(status_code=400, detail="Goal is required")

    title = payload.title or goal[:60]

    # Build a rich, instruction-first system prompt
    sections = []
    sections.append("You are an expert prompt engineer. Craft a clear, structured prompt for an AI model to deliver the best possible result.")

    spec_lines = [f"Objective: {goal}"]
    if payload.context:
        spec_lines.append(f"Context: {payload.context}")
    if payload.audience:
        spec_lines.append(f"Audience: {payload.audience}")
    if payload.tone:
        spec_lines.append(f"Tone: {payload.tone}")
    if payload.style:
        spec_lines.append(f"Style: {payload.style}")
    if payload.format:
        spec_lines.append(f"Output Format: {payload.format}")
    if payload.length:
        spec_lines.append(f"Length: {payload.length}")
    if payload.language:
        spec_lines.append(f"Language: {payload.language}")
    if payload.variables:
        pretty_vars = ", ".join([f"{{{v.strip('{} ')}}}" for v in payload.variables if v])
        if pretty_vars:
            spec_lines.append(f"Variables to include: {pretty_vars}")

    sections.append("\n".join(spec_lines))

    guidance = [
        "Requirements:",
        "- Ask clarifying questions only if critical information is missing.",
        "- Provide step-by-step structure where helpful.",
        "- Use unambiguous instructions.",
        "- Include constraints and success criteria.",
    ]
    if payload.include_examples:
        guidance.append("- Include a short example in the specified format.")

    sections.append("\n".join(guidance))

    final_prompt = (
        "\n\n".join(sections)
        + "\n\nRespond only with the final prompt text ready to copy and paste."
    )

    return {
        "title": title,
        "prompt": final_prompt,
        "tags": payload.tags or [],
    }


@app.post("/api/prompts")
def save_prompt(payload: SavePromptRequest):
    try:
        doc_id = create_document("prompt", payload.model_dump())
        return {"id": doc_id, "status": "saved"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/prompts")
def list_prompts(limit: int = 20):
    try:
        docs = get_documents("prompt", limit=limit)
        # Convert ObjectId and datetime to strings for JSON safety
        def normalize(d: Dict[str, Any]):
            d = dict(d)
            if d.get("_id"):
                d["id"] = str(d.pop("_id"))
            for k, v in list(d.items()):
                if hasattr(v, "isoformat"):
                    d[k] = v.isoformat()
            return d

        return [normalize(d) for d in docs]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
