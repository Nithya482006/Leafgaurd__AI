import os
import subprocess
import sys
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
LOCAL_VENV_PYTHON = BASE_DIR / ".venv" / "Scripts" / "python.exe"
APPDATA_VENV_PYTHON = (
    Path(os.environ["LOCALAPPDATA"]) / "LeafGuardAI" / "venv" / "Scripts" / "python.exe"
    if os.environ.get("LOCALAPPDATA")
    else None
)
MODEL_PATH = BASE_DIR / "models" / "plant_disease_model.keras"
CLASS_NAMES_PATH = BASE_DIR / "models" / "class_names.json"


def main():
    if not MODEL_PATH.is_file() or not CLASS_NAMES_PATH.is_file():
        raise SystemExit(
            "The trained model is missing. Ensure models/plant_disease_model.keras "
            "and models/class_names.json are present."
        )

    python = next(
        (candidate for candidate in (LOCAL_VENV_PYTHON, APPDATA_VENV_PYTHON) if candidate and candidate.is_file()),
        None,
    )
    if python is None:
        raise SystemExit(
            "The app's Python environment is missing. Create .venv and install "
            "requirements.txt as described in README.md."
        )

    command = [
        str(python),
        "-m",
        "flask",
        "--app",
        "app",
        "run",
        "--host",
        "127.0.0.1",
        "--port",
        "5000",
    ]
    return subprocess.call(command, cwd=BASE_DIR)


if __name__ == "__main__":
    sys.exit(main())
