import logging

from langchain_core.callbacks.base import BaseCallbackHandler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("assistant.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("smart_assistant")


class PromptLoggingCallbackHandler(BaseCallbackHandler):
    def on_llm_start(self, serialized, prompts, **kwargs):
        for prompt in prompts:
            logger.info(f"[PROMPT SENT TO LLM]\n{prompt}\n")

    def on_llm_end(self, response, **kwargs):
        logger.info(f"[LLM RESPONSE]\n{response.generations[0][0].text}\n")
