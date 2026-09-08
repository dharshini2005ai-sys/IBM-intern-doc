from typing import List, Dict
from .config import WATSONX_API_KEY, WATSONX_PROJECT_ID, WATSONX_URL, WATSONX_MODEL_ID

class WatsonXLLM:
    def __init__(self):
        self.enabled = bool(WATSONX_API_KEY and WATSONX_PROJECT_ID)
        self.model = None
        if self.enabled:
            from ibm_watsonx_ai import Credentials
            from ibm_watsonx_ai.foundation_models import ModelInference
            credentials = Credentials(url=WATSONX_URL, api_key=WATSONX_API_KEY)
            self.model = ModelInference(
                model_id=WATSONX_MODEL_ID,
                credentials=credentials,
                project_id=WATSONX_PROJECT_ID,
            )

    def chat(self, messages: List[Dict[str, str]]) -> str:
        if not self.enabled:
            user_message = next(
                (m.get("content", "") for m in reversed(messages) if m.get("role") == "user"),
                ""
            )
            return (
                "IBM watsonx.ai is not configured yet. "
                "Add WATSONX_API_KEY and WATSONX_PROJECT_ID to .env. "
                f"Your request was: {user_message}"
            )
        try:
            response = self.model.chat(messages=messages)
            return response["choices"][0]["message"]["content"] or ""
        except Exception as exc:
            return f"I could not connect to the AI model. Check the watsonx.ai configuration. Technical detail: {exc}"

llm = WatsonXLLM()
