"""Download the scans listed in sources.toml from archive.org into data/raw/<archive_id>/.

    python scripts/fetch_archive.py                # all
    python scripts/fetch_archive.py --only ghalib  # one collection
    python scripts/fetch_archive.py --with-text    # also grab archive.org's own OCR (_djvu.txt)
"""

from __future__ import annotations

import argparse
import os
import sys
import tomllib
from pathlib import Path
from urllib.parse import quote

import httpx

API = "https://archive.org"


def pick_files(meta: dict, wanted: str | None, with_text: bool) -> list[dict]:
    files = meta.get("files", [])
    if wanted:
        chosen = [f for f in files if f["name"] == wanted]
    else:
        pdfs = [f for f in files if f["name"].lower().endswith(".pdf") and not f["name"].endswith("_text.pdf")]
        chosen = sorted(pdfs, key=lambda f: int(f.get("size", 0) or 0))[-1:]  # largest = original scan
    if with_text:
        chosen += [f for f in files if f["name"].endswith("_djvu.txt")]
    return chosen


def download(client: httpx.Client, ident: str, name: str, dest: Path) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  = {name} (already here)")
        return
    url = f"{API}/download/{ident}/{quote(name)}"
    tmp = dest.with_suffix(dest.suffix + ".part")
    with client.stream("GET", url, follow_redirects=True) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        done = 0
        with tmp.open("wb") as fh:
            for chunk in r.iter_bytes(1 << 16):
                fh.write(chunk)
                done += len(chunk)
                if total:
                    print(f"\r  ↓ {name}  {done * 100 // total:3d}%", end="", flush=True)
    tmp.rename(dest)
    print(f"\r  ✓ {name}  {done / 1e6:.1f} MB")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", default="sources.toml")
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--with-text", action="store_true")
    args = ap.parse_args()

    cfg = tomllib.loads(Path(args.sources).read_text(encoding="utf-8"))
    raw = Path(os.environ.get("CHITTHI_DATA", "data")) / "raw"
    with httpx.Client(timeout=120, headers={"User-Agent": "chitthi/0.1 (hacktoberfest project)"}) as client:
        for col in cfg.get("collection", []):
            if args.only and col["slug"] not in args.only:
                continue
            for doc in col.get("document", []):
                ident = doc.get("archive_id")
                if not ident:
                    continue
                print(f"{col['slug']}: {doc['title']}  ({ident})")
                meta = client.get(f"{API}/metadata/{ident}").raise_for_status().json()
                folder = raw / ident
                folder.mkdir(parents=True, exist_ok=True)
                files = pick_files(meta, doc.get("file"), args.with_text)
                if not files:
                    print("  ! no matching file; set `file =` in sources.toml")
                for f in files:
                    download(client, ident, f["name"], folder / Path(f["name"]).name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
