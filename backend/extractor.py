import os
import json
import re
import traceback
import ollama
from pydantic import ValidationError
from dotenv import load_dotenv

from schemas import CourseExtractionResult
from retriever import retrieve_context

load_dotenv()


def get_example_schema() -> dict:
    """Produce an example/template JSON structure (flat) to show the LLM what shape to output."""
    return {
        "course_code": "e.g. 410252 or null",
        "course_name": "e.g. Natural Language Processing or null",
        "lecture_hours_per_week": "number or null",
        "practical_hours_per_week": "number or null",
        "tutorial_hours_per_week": "number or null",
        "theory_credits": "number or null",
        "practical_credits": "number or null",
        "tutorial_credits": "number or null",
        "total_credits": "number or null",
        "in_semester_exam_marks": "number or null",
        "end_semester_exam_marks": "number or null",
        "term_work_marks": "number or null",
        "practical_oral_marks": "number or null",
        "total_marks": "number or null",
        "extraction_notes": "string or null"
    }


def build_extraction_prompt(course_query: str, context_text: str, schema_json: dict) -> str:
    prompt = f"""Extract structured information about the course "{course_query}" from the context below.

Context:
{context_text}

Output ONLY a valid JSON object matching this exact structure (no markdown code fences, no explanation, no preamble):

{json.dumps(schema_json, indent=2)}

Rules:
- If a value is not found in the context, use null for that field.
- Do not guess or hallucinate numbers — only extract what is explicitly stated.
- "total_credits" and "total_marks" should be the sum stated in the document if given, otherwise null.
- "extraction_notes" must be EITHER null OR a single short sentence (max 15 words) about a missing/ambiguous field.
  NEVER copy course objectives, descriptions, or any other unrelated text into extraction_notes.
- Output ONLY the JSON object, nothing else.

JSON:"""
    return prompt


def extract_raw_json(prompt: str, temperature: float = 0.0) -> str:
    """Call Ollama and return the raw text response."""
    response = ollama.chat(
        model='qwen2.5:7b-instruct',
        messages=[{'role': 'user', 'content': prompt}],
        options={
            'temperature': temperature,
            'num_predict': 1024,
        }
    )
    return response['message']['content']


def clean_json_response(raw_text: str) -> str:
    """Extract the JSON object from the LLM's raw response, defensively."""
    text = raw_text.strip()

    if "```" in text:
        parts = text.split("```")
        candidates = [p for p in parts if "{" in p]
        if candidates:
            text = max(candidates, key=len)
        text = text.strip()
        if text.lower().startswith("json"):
            text = text[4:].strip()

    start = text.find("{")
    if start == -1:
        return text

    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]

    return text[start:]


def extract_course_info(course_query: str, top_k: int = 7, max_retries: int = 2) -> dict:
    # Augment the user's course query with fixed terms describing what we actually
    # need to retrieve — teaching scheme / credits / exam pattern — so retrieval
    # targets the right table regardless of how vaguely the user phrases their input.
    retrieval_query = (
        f"{course_query} teaching scheme lecture practical tutorial hours "
        f"credits examination scheme ISE ESE term work marks"
    )
    
    contexts = retrieve_context(retrieval_query, top_k=top_k)

    print(f"\n--- Retrieved {len(contexts)} chunks for extraction ---")
    for i, ctx in enumerate(contexts, 1):
        print(f"[{i}] Page {ctx['metadata']['page_number']} (distance={ctx['distance']:.3f}): {ctx['text'][:100]}...")

    if not contexts:
        return {"success": False, "error": "No relevant context found.", "data": None}

    context_text = "\n\n".join([
        f"[{ctx['metadata']['filename']}, Page {ctx['metadata']['page_number']}]\n{ctx['text']}"
        for ctx in contexts
    ])

    schema_example = get_example_schema()
    base_prompt = build_extraction_prompt(course_query, context_text, schema_example)
    prompt = base_prompt

    last_error = None

    for attempt in range(max_retries + 1):
        temp = 0.0 if attempt == 0 else 0.2

        try:
            raw_response = extract_raw_json(prompt, temperature=temp)
        except Exception as e:
            print(f"\n--- Attempt {attempt + 1}: Ollama call failed ---")
            traceback.print_exc()
            last_error = f"Ollama call failed: {e}"
            continue

        cleaned = clean_json_response(raw_response)

        print(f"\n--- Attempt {attempt + 1} raw response ---")
        print(raw_response)
        print(f"--- Attempt {attempt + 1} cleaned ---")
        print(cleaned)

        try:
            parsed_json = json.loads(cleaned)
        except json.JSONDecodeError as e:
            print(f"JSON parse failed: {e}")
            last_error = f"JSON parse error: {e}"
            prompt = base_prompt + f"\n\nNOTE: Previous attempt had invalid JSON ({last_error}). Output ONLY valid, complete JSON."
            continue

        # Defensive safety net: cap extraction_notes length in case the model
        # ignores the length instruction — prevents a single verbose field
        # from ever being the reason validation fails.
        if isinstance(parsed_json.get("extraction_notes"), str) and len(parsed_json["extraction_notes"]) > 150:
            parsed_json["extraction_notes"] = parsed_json["extraction_notes"][:150] + "..."

        try:
            validated = CourseExtractionResult(**parsed_json)
        except ValidationError as e:
            print(f"Schema validation failed: {e}")
            last_error = f"Validation error: {e}"
            prompt = base_prompt + f"\n\nNOTE: Previous attempt failed schema validation ({last_error}). Fix the field types/structure."
            continue

        return {
            "success": True,
            "error": None,
            "data": validated.model_dump(),
            "sources": [
                {"filename": ctx["metadata"]["filename"], "page_number": ctx["metadata"]["page_number"]}
                for ctx in contexts
            ],
            "attempts": attempt + 1
        }

    return {
        "success": False,
        "error": f"Failed after {max_retries + 1} attempts. Last error: {last_error}",
        "data": None
    }


if __name__ == "__main__":
    test_query = "Natural Language Processing course teaching scheme and examination scheme"
    print(f"Extracting for: {test_query}\n")

    result = extract_course_info(test_query)

    print("\n=== FINAL RESULT ===")
    print(json.dumps(result, indent=2))