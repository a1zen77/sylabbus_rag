import os
import json
import ollama
from pydantic import ValidationError
from dotenv import load_dotenv

from schemas import CourseExtractionResult
from retriever import retrieve_context

load_dotenv()


def build_extraction_prompt(course_query: str, context_text: str, schema_json: dict) -> str:
    """Build a prompt instructing the LLM to output ONLY JSON matching the schema."""
    prompt = f"""Extract structured information about the course "{course_query}" from the context below.

Context:
{context_text}

Output ONLY a valid JSON object matching this exact structure (no markdown code fences, no explanation, no preamble):

{json.dumps(schema_json, indent=2)}

Rules:
- If a value is not found in the context, use null for that field.
- Do not guess or hallucinate numbers — only extract what is explicitly stated.
- "total_credits" and "total_marks" should be the sum stated in the document if given, otherwise null.
- Output ONLY the JSON object, nothing else.

JSON:"""
    return prompt


def get_example_schema() -> dict:
    """Produce an example/template JSON structure to show the LLM what shape to output."""
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


def extract_raw_json(prompt: str, temperature: float = 0.0) -> str:
    """Call Ollama and return the raw text response."""
    response = ollama.chat(
        model='llama3.2:3b',
        messages=[{'role': 'user', 'content': prompt}],
        options={
            'temperature': temperature,
            'num_predict': 1024,
        }
    )
    return response['message']['content']


def clean_json_response(raw_text: str) -> str:
    """
    LLMs often wrap JSON in markdown code fences despite instructions.
    Strip those out before parsing.
    """
    text = raw_text.strip()
    if text.startswith("```"):
        # Remove opening fence (```json or ```)
        text = text.split("\n", 1)[1] if "\n" in text else text
    if text.endswith("```"):
        text = text.rsplit("```", 1)[0]
    return text.strip()


def extract_course_info(course_query: str, top_k: int = 5, max_retries: int = 2) -> dict:
    """
    Main extraction pipeline:
    1. Retrieve relevant context chunks
    2. Prompt LLM for structured JSON
    3. Validate against Pydantic schema
    4. Retry once with error feedback if validation fails
    """
    contexts = retrieve_context(course_query, top_k=top_k)

    if not contexts:
        return {
            "success": False,
            "error": "No relevant context found for this course.",
            "data": None
        }

    context_text = "\n\n".join([
        f"[{ctx['metadata']['filename']}, Page {ctx['metadata']['page_number']}]\n{ctx['text']}"
        for ctx in contexts
    ])

    schema_example = get_example_schema()
    prompt = build_extraction_prompt(course_query, context_text, schema_example)

    last_error = None
    for attempt in range(max_retries + 1):
        raw_response = extract_raw_json(prompt)
        cleaned = clean_json_response(raw_response)

        try:
            parsed_json = json.loads(cleaned)
            validated = CourseExtractionResult(**parsed_json)

            return {
                "success": True,
                "error": None,
                "data": validated.model_dump(),
                "sources": [
                    {
                        "filename": ctx["metadata"]["filename"],
                        "page_number": ctx["metadata"]["page_number"]
                    }
                    for ctx in contexts
                ],
                "attempts": attempt + 1
            }

        except (json.JSONDecodeError, ValidationError) as e:
            last_error = None
            for attempt in range(max_retries + 1):
                # Use temp=0 on first try (deterministic), nudge up slightly on retries
                # so the model doesn't just repeat the same truncated output
                temp = 0.0 if attempt == 0 else 0.2
                raw_response = extract_raw_json(prompt, temperature=temp)
                cleaned = clean_json_response(raw_response)

    return {
        "success": False,
        "error": f"Failed to produce valid JSON after {max_retries + 1} attempts. Last error: {last_error}",
        "data": None
    }


if __name__ == "__main__":
    test_query = "Natural Language Processing course teaching scheme and examination scheme"
    print(f"Extracting for: {test_query}\n")

    result = extract_course_info(test_query)

    print(json.dumps(result, indent=2))