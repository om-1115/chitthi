"""Ingest scans into the local index.

    python -m chitthi.ingest sources sources.toml            # everything listed in sources.toml
    python -m chitthi.ingest sources sources.toml --only nehru
    python -m chitthi.ingest add scan.pdf --collection nehru --title "..." --lang hi --pages 1-10
"""

from __future__ import annotations

import argparse
import sys
import time
import tomllib
from pathlib import Path

from .chunk import split_pages
from .config import Settings, get_settings
from .llm import LLM, get_llm
from .ocr import detect_script, read_pages
from .store import Store


def resolve_path(doc: dict, data_dir: Path) -> Path | None:
    if doc.get("path"):
        return Path(doc["path"])
    if doc.get("archive_id"):
        folder = data_dir / "raw" / doc["archive_id"]
        if doc.get("file") and (folder / doc["file"]).exists():
            return folder / doc["file"]
        for pattern in ("*.pdf", "*_djvu.txt", "*.txt"):
            found = sorted(p for p in folder.glob(pattern) if not p.name.endswith("_text.pdf"))
            if found:
                return found[0]
    return None


def ingest_file(
    store: Store,
    llm: LLM,
    s: Settings,
    path: Path,
    *,
    collection: str,
    title: str,
    lang: str,
    ocr: str = "auto",
    pages: str | None = None,
    **meta,
) -> int:
    doc_id = store.add_document(collection, title, lang, str(path), **meta)
    img_dir = s.pages_dir / str(doc_id)
    img_dir.mkdir(parents=True, exist_ok=True)

    page_texts: list[tuple[int, str]] = []
    t0 = time.time()
    for page in read_pages(path, llm, lang=lang, mode=ocr, pages=pages):
        rel = None
        if page.image_png:
            rel = f"{doc_id}/{page.number}.png"
            (s.pages_dir / rel).write_bytes(page.image_png)
        store.add_page(doc_id, page.number, page.method, page.text, rel)
        page_texts.append((page.number, page.text))
        print(f"  p.{page.number:<4} {page.method:<10} {len(page.text):>5} chars  ({time.time() - t0:.0f}s)", flush=True)

    passages = [p for p in split_pages(page_texts) if p.text.strip()]
    if not passages:
        print("  no text found")
        return doc_id
    embs = llm.embed([p.text for p in passages])
    store.add_chunks(doc_id, [(p.page, detect_script(p.text), p.text, e) for p, e in zip(passages, embs)])
    print(f"  indexed {len(passages)} passages")
    return doc_id


def cmd_sources(args, s: Settings, store: Store, llm: LLM) -> int:
    cfg = tomllib.loads(Path(args.file).read_text(encoding="utf-8"))
    missing = 0
    for col in cfg.get("collection", []):
        if args.only and col["slug"] not in args.only:
            continue
        store.upsert_collection(col["slug"], col["name"], col.get("years", ""), col.get("persona", ""))
        print(f"\n== {col['name']} ({col['slug']})")
        for doc in col.get("document", []):
            path = resolve_path(doc, s.data_dir)
            if not path or not path.exists():
                print(f"  ! missing file for '{doc['title']}' - run scripts/fetch_archive.py first")
                missing += 1
                continue
            print(f"- {doc['title']}  [{path.name}]")
            ingest_file(
                store, llm, s, path,
                collection=col["slug"], title=doc["title"], lang=doc["lang"],
                ocr=args.ocr or doc.get("ocr", "auto"), pages=args.pages or doc.get("pages"),
                year=doc.get("year"), source_url=doc.get("source_url"), rights=doc.get("rights"),
            )
    return 1 if missing else 0


def cmd_add(args, s: Settings, store: Store, llm: LLM) -> int:
    if not store.collection(args.collection):
        store.upsert_collection(args.collection, args.name or args.collection, args.years or "")
    ingest_file(
        store, llm, s, Path(args.path),
        collection=args.collection, title=args.title, lang=args.lang, ocr=args.ocr, pages=args.pages,
        year=args.year, source_url=args.source_url,
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="chitthi.ingest")
    sub = ap.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("sources", help="ingest everything in a sources.toml")
    a.add_argument("file")
    a.add_argument("--only", nargs="*", help="collection slugs")
    a.add_argument("--pages", help="override page range, e.g. 1-20")
    a.add_argument("--ocr", choices=["auto", "text", "gemma", "tesseract"])

    b = sub.add_parser("add", help="ingest one file")
    b.add_argument("path")
    b.add_argument("--collection", required=True)
    b.add_argument("--title", required=True)
    b.add_argument("--lang", required=True, choices=["en", "hi", "ur"])
    b.add_argument("--name"), b.add_argument("--years"), b.add_argument("--year")
    b.add_argument("--source-url")
    b.add_argument("--pages")
    b.add_argument("--ocr", default="auto", choices=["auto", "text", "gemma", "tesseract"])

    args = ap.parse_args(argv)
    s = get_settings()
    store, llm = Store(s.db_path), get_llm(s)
    return {"sources": cmd_sources, "add": cmd_add}[args.cmd](args, s, store, llm)


if __name__ == "__main__":
    sys.exit(main())
