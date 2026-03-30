import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

print("Available models:")
for m in genai.list_models():
    if 'embed' in m.name.lower():
        print(f"  - {m.name}")
        print(f"    Supported methods: {m.supported_generation_methods}")