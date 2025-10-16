# llm.py — OpenAI-only JSON completions for RIA Assistant
import os, httpx, time, random

OPENAI_BASE  = os.getenv("OPENAI_BASE", "https://api.openai.com/v1")
OPENAI_KEY   = os.getenv("OPENAI_API_KEY", "")  # no default!
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_ORG   = os.getenv("OPENAI_ORG")  # optional

_CLIENT_TIMEOUT = httpx.Timeout(connect=10.0, read=120.0, write=30.0, pool=30.0)

SYSTEM_MSG = (
    "You are a Regulatory Impact Assessment assistant for Australian Red Cross Lifeblood. "
    "Answer strictly and ONLY using the provided evidence passages. "
    "Be concise, cite as [Document (page X)]. If unclear, say 'Unclear' and require human supervision. "
    "Return a JSON object that exactly matches the schema in the user message."
)

def _post_chat(messages, temperature=0.1, max_tokens=800, retries=3):
    if not OPENAI_KEY:
        raise RuntimeError("OPENAI_API_KEY not set")

    headers = {"Authorization": f"Bearer {OPENAI_KEY}"}
    if OPENAI_ORG:
        headers["OpenAI-Organization"] = OPENAI_ORG

    data = {
        "model": OPENAI_MODEL,
        "messages": messages,
        "response_format": {"type": "json_object"},
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    last_err = None
    for attempt in range(retries):
        try:
            with httpx.Client(timeout=_CLIENT_TIMEOUT) as client:
                r = client.post(f"{OPENAI_BASE}/chat/completions", headers=headers, json=data)
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"].strip()

        except (httpx.ReadTimeout, httpx.ConnectTimeout, httpx.RemoteProtocolError) as e:
            last_err = e

        except httpx.HTTPStatusError as e:
            # Retry on 429/5xx; otherwise surface immediately
            if e.response is None or e.response.status_code not in (429, 500, 502, 503, 504):
                raise
            last_err = e

        # exponential backoff with small jitter
        sleep_s = (1.5 ** attempt) + random.uniform(0, 0.4)
        time.sleep(sleep_s)

    raise last_err if last_err else RuntimeError("OpenAI request failed")

def generate(prompt: str, temperature: float = 0.1, max_tokens: int = 900) -> str:
    """
    `prompt` is built in rag.py and includes:
      - the user's question,
      - the evidence passages,
      - the target JSON schema.
    """
    messages = [
        {"role": "system", "content": SYSTEM_MSG},
        {"role": "user",   "content": prompt},
    ]
    return _post_chat(messages, temperature=temperature, max_tokens=max_tokens, retries=3)
