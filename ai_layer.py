from groq import Groq
import json
import os
from dotenv import load_dotenv

load_dotenv()  # This reads your .env file
print("KEY LOADED:", os.getenv("GROQ_API_KEY", "MISSING")[:8])  # shows first 8 chars only

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is missing! Add it to backend/.env")

client = Groq(api_key=GROQ_API_KEY)

SYSTEM_PROMPT = """You are MedAssist, a medical document simplification AI for Indian patients.
Your ONLY job is to extract and simplify what is written in the document.

STRICT RULES:
1. Never add medical advice not in the document
2. Never suggest alternative medicines
3. Medication dosage must match the prescription EXACTLY
4. If something is unclear write: unclear in document
5. Do not pull any outside information
6. Return ONLY valid JSON, no markdown, no backticks, no extra text

Return this exact JSON:
{
  "familySummary": "one clear sentence a family member can understand",
  "diagnosis": "plain language explanation in 2-3 sentences",
  "originalJargon": "exact medical terms from document",
  "faithfulnessScore": 97,
  "medications": [
    {
      "name": "medicine name exactly as written",
      "dosage": "exact dosage from document",
      "timing": "when to take",
      "duration": "how many days",
      "instructions": "with food or special note"
    }
  ],
  "sideEffects": [
    {
      "effect": "side effect description",
      "severity": "watch"
    }
  ],
  "whenToCallDoctor": "exact conditions from document",
  "followUpChecklist": [
    {
      "task": "what patient needs to do",
      "category": "test"
    }
  ]
}

severity must be: watch OR urgent
category must be: test OR diet OR activity OR review"""


def analyze_document(text: str, language: str = "English", age: str = "") -> dict:
    user_message = f"""Document content:
{text}

Patient age: {age if age else 'not specified'}
Output language: {language}

Return JSON only. No extra text. No markdown."""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        max_tokens=1500,
        temperature=0.1,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message}
        ]
    )

    raw_text = response.choices[0].message.content.strip()
    print(f"GROQ RAW: {raw_text[:200]}")

    raw_text = raw_text.replace("```json", "").replace("```", "").strip()

    start = raw_text.find("{")
    end = raw_text.rfind("}") + 1
    if start != -1 and end > start:
        raw_text = raw_text[start:end]

    result = json.loads(raw_text)
    print(f"PARSED OK: {list(result.keys())}")
    return result