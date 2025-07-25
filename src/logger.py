import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("assistant.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("smart_assistant")