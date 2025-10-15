import os, httpx, time

OLLAMA_BASE = os.getenv("OLLAMA_BASE", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:1b")
OPENAI_BASE = os.getenv("OPENAI_BASE", "https://api.openai.com/v1")
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

def _ollama_available():
    return False
 #   try:
  #      r = httpx.get(OLLAMA_BASE + "/api/tags", timeout=3)
  #      return r.status_code == 200
   # except Exception:
   #     return False

def generate(prompt: str, temperature: float = 0.1, max_tokens: int = 800) -> str:
    payload = {"model": OLLAMA_MODEL, "prompt": prompt, "stream": False, "options": {"temperature": temperature}}
    if _ollama_available():
        for attempt in range(3):
            try:
                with httpx.Client(timeout=httpx.Timeout(600.0)) as client:
                    r = client.post(OLLAMA_BASE + "/api/generate", json=payload)
                r.raise_for_status()
                return r.json().get("response", "").strip()
            except httpx.ReadTimeout:
                time.sleep(2 * (attempt + 1))
            except Exception:
                break  # fall through

    if OPENAI_KEY:
        headers = {"Authorization": f"Bearer {OPENAI_KEY}"}
        data = {"model": OPENAI_MODEL, "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature, "max_tokens": max_tokens}
        r = httpx.post(OPENAI_BASE + "/chat/completions", headers=headers, json=data, timeout=120)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"].strip()

    return ("[Offline mode: no LLM configured]\n"
            "Below is a structured, grounded summary of retrieved evidence. "
            "Please review the quotations and citations carefully.\n")
