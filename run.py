import os
import subprocess
import sys
import threading

import uvicorn


def start_ui() -> None:
    subprocess.run(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            "ui/app.py",
            "--server.headless=true",
        ],
        check=False,
    )


def main() -> None:
    if os.getenv("RUN_UI", "0") == "1":
        threading.Thread(target=start_ui, daemon=True).start()

    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=int(os.getenv("API_PORT", "8000")),
        reload=False,
    )


if __name__ == "__main__":
    main()
