import logging
from logging.handlers import RotatingFileHandler
import os

LOG_DIR = "logs"
LOG_FILE = os.path.join(LOG_DIR, "run.log")

# Ensure log directory exists
os.makedirs(LOG_DIR, exist_ok=True)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Console handler
console = logging.StreamHandler()
console.setLevel(logging.INFO)
console.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))

# File handler (with rotation)
file_handler = RotatingFileHandler(LOG_FILE, maxBytes=10_000_000, backupCount=3)

# Force rollover on startup if the log file already exists
if os.path.exists(LOG_FILE):
    file_handler.doRollover()

file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))

logger.addHandler(console)
logger.addHandler(file_handler)
