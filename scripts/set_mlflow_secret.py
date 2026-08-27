"""Store allowlisted DagsHub/MLflow settings from stdin with mode 0600.

Expected input is dotenv-style ``KEY=value`` text. Unknown keys are ignored,
values are never printed, and the output file is written atomically. This lets
the controller filter the local ``.env`` and stream only the tracking settings
to the worker without putting credentials in Git or command arguments.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path


SECRET_PATH = Path("/root/.iaaa_mlflow.env")
ALLOWED = {
    "DAGSHUB_USER_TOKEN",
    "DAGSHUB_REPO_OWNER",
    "DAGSHUB_REPO_NAME",
    "DAGSHUB_TRACKING_URI",
}
REQUIRED = {
    "DAGSHUB_USER_TOKEN",
    "DAGSHUB_REPO_OWNER",
    "DAGSHUB_REPO_NAME",
    "DAGSHUB_TRACKING_URI",
}


def main() -> int:
    values: dict[str, str] = {}
    for raw_line in sys.stdin.read().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key in ALLOWED and value.strip():
            values[key] = value.strip()
    missing = sorted(REQUIRED - values.keys())
    if missing:
        raise SystemExit(f"Missing required tracking settings: {', '.join(missing)}")

    SECRET_PATH.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=".iaaa_mlflow.", dir=SECRET_PATH.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            for key in sorted(values):
                handle.write(f"{key}={values[key]}\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temp_name, 0o600)
        os.replace(temp_name, SECRET_PATH)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)
    print(
        f"Stored {len(values)} allowlisted tracking settings at "
        f"{SECRET_PATH} with mode 0600."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
