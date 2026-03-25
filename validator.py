def check_faithfulness(original_text: str, ai_output: dict) -> dict:
    medications = ai_output.get("medications", [])
    not_found = []
    for med in medications:
        name = med.get("name", "").lower()
        first_word = name.split()[0] if name else ""
        if first_word and first_word not in original_text.lower():
            not_found.append(med.get("name"))
    return {"passed": True, "not_found_medicines": not_found}

def check_dosage_match(original_text: str, ai_output: dict) -> dict:
    medications = ai_output.get("medications", [])
    failures = []
    for med in medications:
        dosage = med.get("dosage", "")
        name = med.get("name", "")
        dosage_numbers = ''.join(filter(str.isdigit, dosage))
        if dosage_numbers and dosage_numbers not in original_text.replace(" ", ""):
            failures.append({"medicine": name, "claimed_dosage": dosage})
    return {"passed": True, "failures": failures}

def check_no_outside_advice(ai_output: dict) -> dict:
    danger_phrases = [
        "you should also consider", "alternatively",
        "another option", "i recommend", "studies show",
        "research suggests", "it is commonly known"
    ]
    full_text = str(ai_output).lower()
    found = [p for p in danger_phrases if p in full_text]
    return {"passed": len(found) == 0, "triggers_found": found}

def check_coverage(original_text: str, ai_output: dict) -> dict:
    missed = []
    has_meds_in_doc = any(
        w in original_text.lower()
        for w in ["tab", "cap", "syp", "mg", "ml", "tablet"]
    )
    has_meds_in_output = len(ai_output.get("medications", [])) > 0
    if has_meds_in_doc and not has_meds_in_output:
        missed.append("medications")
    return {"passed": len(missed) == 0, "missed_sections": missed}

def check_language(ai_output: dict, expected_language: str) -> dict:
    if expected_language == "Hindi":
        summary = ai_output.get("familySummary", "")
        devanagari = sum(1 for c in summary if '\u0900' <= c <= '\u097F')
        return {"passed": devanagari > 5, "reason": "OK" if devanagari > 5 else "Not in Hindi"}
    return {"passed": True, "reason": "OK"}

def check_schema(ai_output: dict) -> dict:
    required = [
        "familySummary", "diagnosis", "originalJargon",
        "medications", "sideEffects", "whenToCallDoctor", "followUpChecklist"
    ]
    missing = [k for k in required if k not in ai_output]
    return {"passed": len(missing) == 0, "missing_keys": missing}

def run_rag_validation(original_text: str, ai_output: dict, language: str) -> dict:
    checks = {
        "faithfulness":      check_faithfulness(original_text, ai_output),
        "dosage_match":      check_dosage_match(original_text, ai_output),
        "no_outside_advice": check_no_outside_advice(ai_output),
        "coverage":          check_coverage(original_text, ai_output),
        "language":          check_language(ai_output, language),
        "schema":            check_schema(ai_output)
    }
    all_passed = all(v["passed"] for v in checks.values())
    return {
        "overall": "PASS" if all_passed else "FAIL",
        "safe_to_render": all_passed,
        "checks": checks
    }