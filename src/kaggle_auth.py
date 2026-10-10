"""Kaggle authentication for Colab and local runs.

Token-only. The credential is the Kaggle API token (``KAGGLE_API_TOKEN``),
the ``KGAT_...`` string from kaggle.com -> Settings -> API -> Create New Token.

On Google Colab the token is read from Colab Secrets (``userdata``) and placed
in the environment so the official Kaggle API picks it up. Off Colab this
module leaves the environment alone: a token already in the environment (or in
``~/.kaggle/access_token``) is used as-is.

Secret values are never printed.
"""

from __future__ import annotations

import os
from pathlib import Path

SECRET_NAME = "KAGGLE_API_TOKEN"


def on_colab() -> bool:
    """True when running inside Google Colab."""
    try:
        import google.colab  # noqa: F401
    except ImportError:
        return False
    return True


def has_token() -> bool:
    """True when a Kaggle API token is available without contacting Colab."""
    if os.environ.get(SECRET_NAME):
        return True
    home = Path.home() / ".kaggle"
    return (home / "access_token").exists() or (home / "access_token.txt").exists()


def ensure_kaggle_credentials() -> None:
    """Make the Kaggle API token available, reading Colab Secrets if needed.

    - If ``KAGGLE_API_TOKEN`` (or ``~/.kaggle/access_token``) is already
      present, this is a no-op.
    - On Colab, the token is copied from the ``KAGGLE_API_TOKEN`` Secret into
      the environment. A missing secret (or notebook access not granted)
      raises a RuntimeError naming the exact fix.
    """
    if has_token():
        return
    if not on_colab():
        return
    try:
        from google.colab import userdata
    except ImportError as exc:
        raise RuntimeError(
            f"{SECRET_NAME} is not set and Colab Secrets are unavailable."
        ) from exc
    try:
        token = userdata.get(SECRET_NAME)
    except Exception as exc:
        raise RuntimeError(
            f"Could not read the {SECRET_NAME!r} Colab Secret. Add it in the "
            "Secrets panel (key icon), paste your KGAT_... token, and grant "
            "this notebook access."
        ) from exc
    if not token:
        raise RuntimeError(
            f"The {SECRET_NAME!r} Colab Secret is empty. Paste your KGAT_... "
            "token into it (Kaggle -> Settings -> API -> Create New Token)."
        )
    os.environ[SECRET_NAME] = token
