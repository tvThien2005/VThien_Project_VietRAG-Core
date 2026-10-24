import sys
from pathlib import Path

import uvicorn

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import get_config


if __name__ == "__main__":
	api_config = get_config().api
	uvicorn.run(
		"api.main:app",
		host=api_config.host,
		port=api_config.port,
		reload=api_config.reload,
	)
