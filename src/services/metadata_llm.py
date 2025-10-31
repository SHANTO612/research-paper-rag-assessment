import os
import asyncio
import logging
from typing import Optional, Dict

import httpx
import google.generativeai as genai

logger = logging.getLogger(__name__)


class LLMMetadataExtractor:
    """LLM-backed extractor for paper metadata (title, authors, year)."""

    def __init__(self):
        self.llm_type = os.getenv("LLM_TYPE", "gemini").lower()
        self.gemini_api_key = os.getenv("GEMINI_API_KEY", "")
        self.gemini_model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        self.ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.ollama_model = os.getenv("OLLAMA_MODEL", "llama3")
        self.deepseek_api_key = os.getenv("DEEPSEEK_API_KEY", "")
        self.deepseek_api_url = "https://api.deepseek.com/v1/chat/completions"

        if self.llm_type == "gemini" and self.gemini_api_key:
            genai.configure(api_key=self.gemini_api_key)

    async def extract(self, filename: str, text_sample: str) -> Optional[Dict]:
        """
        Use the configured LLM to extract title, authors, and year from a small text sample (ideally first 1-2 pages).
        Returns a dict {title, authors, year} or None on failure.
        """
        prompt = self._build_prompt(filename, text_sample)

        try:
            if self.llm_type == "gemini":
                return await self._call_gemini(prompt)
            if self.llm_type == "ollama":
                return await self._call_ollama(prompt)
            if self.llm_type == "deepseek":
                return await self._call_deepseek(prompt)
        except Exception as e:
            logger.error(f"LLM metadata extraction failed: {e}")

        return None

    def _build_prompt(self, filename: str, text_sample: str) -> str:
        return (
            "You are extracting bibliographic metadata from the beginning of a research paper.\n"
            "Given the filename and the first page(s) of text, return a strict JSON object with keys: "
            "title (string), authors (string; comma-separated full names), year (integer or null).\n"
            "Rules:\n"
            "- Prefer the paper's actual title over headers like arXiv IDs or dates.\n"
            "- Authors should be human names; do not return single stop-words like 'The'.\n"
            "- Year should be the publication year if clear, else null.\n"
            "- Respond with ONLY the JSON, no extra text.\n\n"
            f"FILENAME: {filename}\n"
            "TEXT:\n"
            f"{text_sample[:4000]}\n"
        )

    async def _call_ollama(self, prompt: str) -> Optional[Dict]:
        async with httpx.AsyncClient(timeout=20.0) as client:
            resp = await client.post(
                f"{self.ollama_base_url}/api/generate",
                json={"model": self.ollama_model, "prompt": prompt, "stream": False},
            )
            resp.raise_for_status()
            content = resp.json().get("response", "").strip()
            return self._parse_json(content)

    async def _call_gemini(self, prompt: str) -> Optional[Dict]:
        if not self.gemini_api_key:
            return None
        model = genai.GenerativeModel(self.gemini_model)
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(None, lambda: model.generate_content(prompt))
        content = (response.text or "").strip()
        return self._parse_json(content)

    async def _call_deepseek(self, prompt: str) -> Optional[Dict]:
        if not self.deepseek_api_key:
            return None
        async with httpx.AsyncClient(timeout=20.0) as client:
            resp = await client.post(
                self.deepseek_api_url,
                headers={
                    "Authorization": f"Bearer {self.deepseek_api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "deepseek-chat",
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.1,
                    "max_tokens": 300,
                },
            )
            resp.raise_for_status()
            content = resp.json().get("choices", [{}])[0].get("message", {}).get("content", "").strip()
            return self._parse_json(content)

    def _parse_json(self, raw: str) -> Optional[Dict]:
        import json
        # Attempt to extract JSON if extra text leaked
        try:
            return json.loads(raw)
        except Exception:
            pass
        # Fallback: try to find first {...}
        try:
            start = raw.find("{")
            end = raw.rfind("}")
            if start != -1 and end != -1 and end > start:
                return json.loads(raw[start : end + 1])
        except Exception:
            return None


# Singleton dependency
_singleton = LLMMetadataExtractor()

async def get_llm_metadata_extractor() -> LLMMetadataExtractor:
    return _singleton


