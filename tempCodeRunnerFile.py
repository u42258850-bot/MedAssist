from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
from parser import extract_text
from ai_layer import analyze_document
from validator import run_rag_validation

app = FastAPI(title="MedAssist API — RxLens Powered")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"status": "MedAssist API is live 🚀", "ml_model": "RxLens Integrated"}

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    try:
        print(f"UPLOAD: {file.filename}")
        contents = await file.read()
        filename = file.filename.lower()
        extracted_text = extract_text(contents, filename)
        print(f"TEXT PREVIEW: {extracted_text[:150]}")
        if not extracted_text or len(extracted_text.strip()) < 10:
            return JSONResponse(status_code=400,
                content={"success": False, "error": "Could not extract text from document."})
        return {
            "success": True,
            "filename": file.filename,
            "text": extracted_text,
            "char_count": len(extracted_text)
        }
    except Exception as e:
        print(f"UPLOAD ERROR: {e}")
        return JSONResponse(status_code=500,
            content={"success": False, "error": str(e)})

@app.post("/analyze")
async def analyze(
    text: str = Form(...),
    language: str = Form(default="English"),
    age: str = Form(default="")
):
    try:
        print(f"ANALYZE: lang={language} age={age}")
        print(f"TEXT: {text[:100]}")
        ai_result = analyze_document(text, language, age)
        print(f"AI DONE: {list(ai_result.keys())}")
        validation = run_rag_validation(text, ai_result, language)
        print(f"VALIDATION: {validation['overall']}")
        if not validation["safe_to_render"]:
            return JSONResponse(status_code=422,
                content={"success": False, "error": "Safety validation failed."})
        return {"success": True, "data": ai_result}
    except Exception as e:
        print(f"ANALYZE ERROR: {e}")
        return JSONResponse(status_code=500,
            content={"success": False, "error": str(e)})

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
