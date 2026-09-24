"""Project folder, shared by every script (the scripts find data/, models/ and reports/ from here)."""
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
