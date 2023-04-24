

import os
from pathlib import Path


BASE_HTTP = "http://" + os.environ.get("STABILITY_HOST", "localhost:5000")
SOURCES = Path(__file__).parent.parent / "sources"
