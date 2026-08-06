"""Archive bids no longer listed as ongoing on GeM and remove them from search.

The GeM API scan must complete successfully before any local data is changed.
Closed bid PDFs are moved to ``downloads/expired/YYYY-MM-DD`` (recoverable),
while their chunks and Chroma vectors are removed from the searchable portal.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from create_embeddings import DEFAULT_CHROMA_PATH, DEFAULT_COLLECTION, batched, iter_bids
from downloading_v2 import fetch_page_with_retry, get_session_and_csrf


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_CHUNKS = PROJECT_DIR / "bid_chunks.json"
DEFAULT_BIDS_DIR = PROJECT_DIR / "downloads" / "bids"
DEFAULT_ARCHIVE_DIR = PROJECT_DIR / "downloads" / "expired"


def fetch_ongoing_bid_numbers(delay_seconds: float = 0.3) -> set[str]:
    """Return every current Bid number from GeM's ongoing-bids listing.

    Unlike individual bid lookups, this scans the listing once and therefore
    scales well for a weekly check of the entire local corpus.
    """
    session, csrf = get_session_and_csrf()
    active: set[str] = set()
    page = 1
    while True:
        bids, raw_doc_count, csrf, session = fetch_page_with_retry(session, csrf, page)
        if raw_doc_count == 0:
            break
        active.update(str(bid["bid_number"]) for bid in bids if bid.get("bid_number"))
        print(f"Scanned GeM page {page:,}; {len(active):,} active bids found", flush=True)
        page += 1
        time.sleep(delay_seconds)
    return active


def safely_archive_pdf(pdf_path_value: Any, bids_dir: Path, archive_dir: Path, archived: set[Path]) -> str | None:
    """Move one indexed PDF into the dated archive, staying inside bids_dir."""
    if not isinstance(pdf_path_value, str) or not pdf_path_value:
        return None
    root = bids_dir.resolve()
    source = (root / pdf_path_value).resolve()
    try:
        source.relative_to(root)
    except ValueError:
        return None
    if source in archived or not source.is_file():
        return None

    destination = archive_dir / pdf_path_value
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        destination = destination.with_name(f"{destination.stem}_{int(time.time())}{destination.suffix}")
    shutil.move(str(source), str(destination))
    archived.add(source)
    return str(destination)


def remove_from_chroma(chunk_ids: list[str], chroma_path: Path, collection_name: str) -> None:
    if not chunk_ids:
        return
    try:
        import chromadb

        collection = chromadb.PersistentClient(path=str(chroma_path)).get_collection(collection_name)
        for ids in batched(chunk_ids, 1_000):
            collection.delete(ids=ids)
    except Exception as error:
        # Do not hide a partial cleanup: the archive and chunks file remain
        # recoverable, and the scheduled run will report the failure clearly.
        raise RuntimeError(f"Could not remove expired chunks from Chroma: {error}") from error


def archive_expired_bids(
    *,
    chunk_file: Path = DEFAULT_CHUNKS,
    bids_dir: Path = DEFAULT_BIDS_DIR,
    archive_root: Path = DEFAULT_ARCHIVE_DIR,
    chroma_path: Path = DEFAULT_CHROMA_PATH,
    collection_name: str = DEFAULT_COLLECTION,
    apply: bool = False,
) -> dict[str, Any]:
    if not chunk_file.is_file():
        raise FileNotFoundError(chunk_file)

    active_bid_numbers = fetch_ongoing_bid_numbers()
    expired_bid_ids: list[str] = []
    expired_chunk_ids: list[str] = []
    expired_pdf_paths: list[Any] = []
    # Stream retained bids straight into a temporary file. This avoids loading
    # the ~900 MB chunk corpus into memory during the weekly cleanup.
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False, dir=chunk_file.parent, suffix=".json") as temp:
        temp.write('{"bids":[\n')
        retained_count = 0
        for bid in iter_bids(chunk_file):
            if str(bid.get("bid_id", "")) in active_bid_numbers:
                if retained_count:
                    temp.write(',\n')
                json.dump(bid, temp, ensure_ascii=False)
                retained_count += 1
                continue
            expired_bid_ids.append(str(bid.get("bid_id", "")))
            expired_pdf_paths.append(bid.get("metadata", {}).get("pdf_path"))
            expired_chunk_ids.extend(
                str(chunk["chunk_id"]) for chunk in bid.get("chunks", []) if chunk.get("chunk_id")
            )
        temp.write('\n]}\n')
        temporary_chunks_file = Path(temp.name)

    report: dict[str, Any] = {
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "active_bid_count": len(active_bid_numbers),
        "expired_bid_count": len(expired_bid_ids),
        "expired_chunk_count": len(expired_chunk_ids),
        "expired_bid_ids": expired_bid_ids,
        "applied": apply,
    }
    if not apply:
        temporary_chunks_file.unlink(missing_ok=True)
        return report

    archive_dir = archive_root / datetime.now().date().isoformat()
    # Remove vectors and replace the source state before moving any files.
    # If either operation fails, PDFs stay in their original dated folder.
    remove_from_chroma(expired_chunk_ids, chroma_path, collection_name)
    os.replace(temporary_chunks_file, chunk_file)
    archived: set[Path] = set()
    archived_paths = []
    for pdf_path in expired_pdf_paths:
        archived_path = safely_archive_pdf(pdf_path, bids_dir, archive_dir, archived)
        if archived_path:
            archived_paths.append(archived_path)
    report["archived_pdf_count"] = len(archived_paths)
    report["archive_dir"] = str(archive_dir)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Archive bids that are no longer ongoing on GeM.")
    parser.add_argument("--chunks", type=Path, default=DEFAULT_CHUNKS)
    parser.add_argument("--bids-dir", type=Path, default=DEFAULT_BIDS_DIR)
    parser.add_argument("--archive-dir", type=Path, default=DEFAULT_ARCHIVE_DIR)
    parser.add_argument("--chroma-path", type=Path, default=DEFAULT_CHROMA_PATH)
    parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    parser.add_argument("--apply", action="store_true", help="Move expired PDFs and remove their searchable data.")
    args = parser.parse_args()
    report = archive_expired_bids(
        chunk_file=args.chunks,
        bids_dir=args.bids_dir,
        archive_root=args.archive_dir,
        chroma_path=args.chroma_path,
        collection_name=args.collection,
        apply=args.apply,
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
