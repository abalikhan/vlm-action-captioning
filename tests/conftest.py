"""Stub out heavy ML dependencies so unit tests run without GPU/torch installed."""

import sys
from unittest.mock import MagicMock

for mod in [
    "torch",
    "transformers",
    "qwen_vl_utils",
    "accelerate",
    "peft",
    "bitsandbytes",
]:
    sys.modules.setdefault(mod, MagicMock())
