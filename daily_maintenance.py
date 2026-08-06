"""Scheduled daily GeM ingestion, embedding, and weekly expiry maintenance."""

from __future__ import annotations

import argparse
import json
import os
from argparse import Namespace
from datetime import datetime, timedelta, timezone
from pathlib import Path

from create_embeddings import DEFAULT_CHROMA_PATH, DEFAULT_COLLECTION, DEFAULT_MODEL, build_embeddings
from downloading_v2 import main as download_new_bids
from gem_expiry_cleanup import archive_expired_bids
from gem_extract_and_chunks import process_directory
from gem_hybrid_retrieval import HybridRetriever


PROJECT_DIR = Path(__file__).resolve().parent
DOWNLOADS_DIR = PROJECT_DIR / "downloads"
STATE_FILE = DOWNLOADS_DIR / "maintenance_state.json"
LOCK_FILE = DOWNLOADS_DIR / ".maintenance.lock"


def load_state() -> dict:
    try:
        return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def save_state(state: dict) -> None:
    DOWNLOADS_DIR.mkdir(exist_ok=True)
    temporary = STATE_FILE.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, indent=2), encoding="utf-8")
    os.replace(temporary, STATE_FILE)


def expiry_is_due(state: dict) -> bool:
    value = state.get("last_expiry_check_at")
    if not value:
        return True
    try:
        return datetime.now(timezone.utc) - datetime.fromisoformat(value) >= timedelta(days=7)
    except ValueError:
        return True


def run_maintenance(force_expiry: bool = False, skip_expiry: bool = False) -> None:
    DOWNLOADS_DIR.mkdir(exist_ok=True)
    try:
        lock_handle = os.open(LOCK_FILE, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as error:
        raise RuntimeError("Another maintenance run is already active.") from error

    try:
        with os.fdopen(lock_handle, "w", encoding="utf-8") as lock:
            lock.write(datetime.now(timezone.utc).isoformat())

        print("[1/4] Downloading newly listed GeM bid PDFs...", flush=True)
        download_new_bids()

        print("[2/4] Chunking new or changed PDFs...", flush=True)
        chunk_file = PROJECT_DIR / "bid_chunks.json"
        sync_file = DOWNLOADS_DIR / "chunk_sync.json"
        process_directory(
            str(DOWNLOADS_DIR / "bids"),
            str(DOWNLOADS_DIR / "downloaded_bid_manifest.json"),
            str(chunk_file),
            str(sync_file),
        )
        sync = json.loads(sync_file.read_text(encoding="utf-8"))

        print("[3/4] Embedding only new or changed chunks...", flush=True)
        changed_chunks = len(sync.get("upsert_chunk_ids", [])) + len(sync.get("deleted_chunk_ids", []))
        build_embeddings(Namespace(
            input=chunk_file,
            chroma_path=DEFAULT_CHROMA_PATH,
            collection=DEFAULT_COLLECTION,
            model=DEFAULT_MODEL,
            batch_size=64,
            limit=0,
            reset=False,
            sync_file=sync_file,
        ))

        if changed_chunks:
            print("Refreshing the lexical index...", flush=True)
            HybridRetriever().build_lexical_index(rebuild=True)

        state = load_state()
        if not skip_expiry and (force_expiry or expiry_is_due(state)):
            print("[4/4] Running weekly GeM expiry check...", flush=True)
            report = archive_expired_bids(apply=True)
            state["last_expiry_check_at"] = datetime.now(timezone.utc).isoformat()
            state["last_expiry_report"] = report
            if report["expired_chunk_count"]:
                print("Refreshing lexical index after expiry cleanup...", flush=True)
                HybridRetriever().build_lexical_index(rebuild=True)
        else:
            print("[4/4] Weekly expiry check is not due yet.", flush=True)

        state["last_daily_run_at"] = datetime.now(timezone.utc).isoformat()
        save_state(state)
        print("Maintenance complete.", flush=True)
    finally:
        try:
            LOCK_FILE.unlink()
        except FileNotFoundError:
            pass


def main() -> None:
    parser = argparse.ArgumentParser(description="Run daily GeM portal maintenance.")
    parser.add_argument("--force-expiry", action="store_true", help="Run the weekly expiry step now.")
    parser.add_argument("--skip-expiry", action="store_true", help="Skip the expiry step for this run.")
    args = parser.parse_args()
    run_maintenance(force_expiry=args.force_expiry, skip_expiry=args.skip_expiry)


if __name__ == "__main__":
    main()
