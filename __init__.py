"""Package initialization for DocMind AI.

Sets up logging configuration for the entire DocMind AI project,
creating a logs directory and configuring loguru for file and console logging.
"""  # noqa: N999

import sys
from pathlib import Path

from loguru import logger

# Create logs directory if it doesn't exist
logs_dir = Path("logs")
logs_dir.mkdir(exist_ok=True)

# Configure loguru with simple, effective logging
logger.remove()  # Remove default handler
logger.add(
    "logs/docmind_{time}.log", rotation="10 MB", retention="7 days", level="INFO"
)
logger.add(sys.stderr, level="WARNING")  # Console output for warnings+
