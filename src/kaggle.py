"""Download an arbitrary Kaggle dataset, reusing the project's auth pattern.

``src/fetch_data.py`` knows how to fetch the one diabetes dataset this project
was built on. For the exam the teacher may hand over *any* Kaggle link, so this
module generalises that download step::

    from src.kaggle import download_dataset
    csv_path = download_dataset("https://www.kaggle.com/datasets/owner/slug",
                                dest=Path("/content/newdata"))

Authentication is the ``KAGGLE_API_TOKEN`` -- from Colab Secrets on Colab,
from the environment (or ``~/.kaggle/access_token``) locally -- and the
failure message is written for a fresh Colab runtime, since that is where
it will actually be read.
"""

from __future__ import annotations

import re
from pathlib import Path

from .kaggle_auth import ensure_kaggle_credentials, has_token

# Accepts https://www.kaggle.com/datasets/<owner>/<slug> with any trailing
# path (/data, ?select=..., #discussion) stripped.
_URL_RE = re.compile(r"kaggle\.com/datasets/([^/\s?#]+)/([^/\s?#]+)")


def parse_slug(url_or_slug: str) -> str:
    """Normalise a Kaggle URL or ``owner/slug`` string to ``owner/slug``."""
    text = url_or_slug.strip().rstrip("/")
    if not text:
        raise ValueError("empty Kaggle URL/slug")

    if text.startswith("http"):
        match = _URL_RE.search(text)
        if not match:
            raise ValueError(
                f"could not find a dataset slug in {url_or_slug!r}; expected a "
                "URL like https://www.kaggle.com/datasets/<owner>/<slug>"
            )
        return f"{match.group(1)}/{match.group(2)}"

    parts = [p for p in text.split("/") if p]
    if len(parts) >= 2:
        return f"{parts[-2]}/{parts[-1]}"

    raise ValueError(
        f"could not parse {url_or_slug!r}; pass an 'owner/slug' pair or a "
        "full kaggle.com/datasets URL"
    )


def _credential_hint() -> str:
    return (
        "No Kaggle API token found. Get one at kaggle.com -> Settings -> API "
        "-> Create New Token, then:\n"
        "  on Colab: add it as a KAGGLE_API_TOKEN Secret (key icon, grant "
        "this notebook access);\n"
        "  locally: export KAGGLE_API_TOKEN, or save it to "
        "~/.kaggle/access_token."
    )


def download_dataset(url_or_slug: str, dest: Path) -> Path:
    """Download and unzip a Kaggle dataset, returning the main CSV path.

    ``dest`` is created if needed and receives the extracted files. When the
    archive holds several CSVs the largest one is returned (the others are
    printed) -- that is almost always the table that carries the labels.
    """
    slug = parse_slug(url_or_slug)
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)

    ensure_kaggle_credentials()
    if not has_token():
        raise SystemExit(_credential_hint())

    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError as exc:
        raise SystemExit(
            "The kaggle package is required to download a dataset. "
            "Install it with: pip install kaggle"
        ) from exc

    api = KaggleApi()
    api.authenticate()

    print(f"Downloading {slug} from the Kaggle API ...")
    api.dataset_download_files(slug, path=str(dest), unzip=True, quiet=False)

    csvs = sorted(dest.rglob("*.csv"))
    if not csvs:
        raise SystemExit(f"No CSV found under {dest} after downloading {slug}.")
    if len(csvs) > 1:
        csvs = sorted(csvs, key=lambda p: p.stat().st_size, reverse=True)
        print(f"Multiple CSVs found; using the largest:")
        for path in csvs:
            print(f"  {path.name}  ({path.stat().st_size:,} bytes)")
    print(f"Using CSV: {csvs[0]}")
    return csvs[0]


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        raise SystemExit("usage: python -m src.kaggle <url-or-owner/slug> [dest]")
    destination = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("data_new")
    print(download_dataset(sys.argv[1], destination))
