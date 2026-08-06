import json
import os
import re
import time
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

BASE_URL = "https://bidplus.gem.gov.in"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
}

# Configuration
MAX_WORKERS = 5  # Safe concurrency limit for GeM
BASE_DOWNLOAD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "downloads")
BIDS_DIR = os.path.join(BASE_DOWNLOAD_DIR, "bids")
# Persists successful bid IDs even after old local files are deleted.
MANIFEST_PATH = os.path.join(BASE_DOWNLOAD_DIR, "downloaded_bid_manifest.json")
INDIA_TZ = ZoneInfo("Asia/Kolkata")

# API Sort Key: If results are empty on page 1 or throw HTTP 400/422 errors,
# inspect DevTools -> Network -> payload for the exact "sort" value on the site.
SORT_KEY = "Bid-Start-Date-Latest"

# With newest-first sorting, three consecutive known Bid pages are enough to
# establish that the remaining results are older ones already processed.
EARLY_STOP_LIMIT = 3

# Set MAX_NEW_DOWNLOADS_PER_RUN only for a capped manual/test run, e.g.
# PowerShell: $env:MAX_NEW_DOWNLOADS_PER_RUN = "10"
# Leave it unset (or 0) for the normal unlimited incremental sync.
MAX_NEW_DOWNLOADS_PER_RUN = max(0, int(os.environ.get("MAX_NEW_DOWNLOADS_PER_RUN", "0")))

# Set FULL_SYNC=1 only when every currently available Bid PDF must be fetched.
# Normal scheduled runs leave this unset and stop after known newest pages.
FULL_SYNC = os.environ.get("FULL_SYNC", "").strip().lower() in {"1", "true", "yes"}

# Thread locks and counters
print_lock = Lock()
progress_counter = 0
success_counter = 0
failed_counter = 0


def get_session_and_csrf():
    """Load main page, grab cookies + CSRF token."""
    print("[Session] Fetching main page for CSRF token...")
    session = requests.Session()
    session.headers.update(HEADERS)
    resp = session.get(f"{BASE_URL}/all-bids", timeout=30)
    resp.raise_for_status()

    match = re.search(r"csrf_bd_gem_nk['\"\s:]+['\"]([a-f0-9]+)['\"]", resp.text)
    if not match:
        raise ValueError("CSRF token not found in page source")

    csrf = match.group(1)
    print(f"[Session] CSRF Token retrieved: {csrf[:12]}...")
    return session, csrf


def fetch_bid_ids_page(session, csrf, page_num):
    """Fetch bid-document IDs from one GeM results page.

    A result whose current number is an RA still exposes its original Bid
    document through ``b_id_parent``.  We queue that parent Bid document and
    never queue the RA document itself.
    """
    payload = {
        "page": page_num,
        "param": {"searchBid": "", "searchType": "fullText"},
        "filter": {
            "bidStatusType": "ongoing_bids",
            "byType": "all",
            "highBidValue": "",
            "byEndDate": {"from": "", "to": ""},
            "sort": SORT_KEY
        }
    }

    resp = session.post(
        f"{BASE_URL}/all-bids-data",
        data={
            "payload": json.dumps(payload),
            "csrf_bd_gem_nk": csrf,
        },
        headers={
            "Referer": f"{BASE_URL}/all-bids",
            "X-Requested-With": "XMLHttpRequest",
            "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
        },
        timeout=30,
    )

    if resp.status_code != 200:
        snippet = resp.text[:300].replace("\n", " ").strip()
        raise requests.exceptions.HTTPError(
            f"{resp.status_code} on page {page_num} -- body starts: {snippet!r}"
        )

    data = resp.json()
    docs = data["response"]["response"]["docs"]
    raw_doc_count = len(docs)

    bids = []
    for doc in docs:
        bid_id = doc["b_id"][0]
        bid_num = doc.get("b_bid_number", [""])[0]
        # An RA card includes the original Bid's ID and number. Download the
        # original Bid PDF, not the RA schedule/document currently listed.
        if "/R/" in bid_num:
            parent_ids = doc.get("b_id_parent", [])
            parent_numbers = doc.get("b_bid_number_parent", [])
            if not parent_ids:
                print(f"  Skipping RA {bid_num}: no parent Bid document ID was returned.")
                continue
            bids.append({
                "bid_id": parent_ids[0],
                "bid_number": parent_numbers[0] if parent_numbers else "",
            })
            continue
        bids.append({
            "bid_id": bid_id,
            "bid_number": bid_num,
        })

    return bids, raw_doc_count, session.cookies.get("csrf_gem_cookie", csrf)


def fetch_page_with_retry(session, csrf, page_num, max_retries=2):
    """Fetch page with session refresh retry strategy."""
    last_exc = None
    for attempt in range(max_retries + 1):
        try:
            bids, raw_doc_count, csrf = fetch_bid_ids_page(session, csrf, page_num)
            return bids, raw_doc_count, csrf, session
        except Exception as e:
            last_exc = e
            with print_lock:
                print(f"  [Attempt {attempt + 1}/{max_retries + 1}] page {page_num}: {e}")
            if attempt < max_retries:
                time.sleep(3 * (attempt + 1))
                try:
                    session, csrf = get_session_and_csrf()
                except Exception as refresh_err:
                    with print_lock:
                        print(f"  Session refresh failed: {refresh_err}")
    raise last_exc


def get_downloaded_bid_ids(directory):
    """Recursively scan the dated download folders for valid bid IDs."""
    downloaded = set()
    if not os.path.exists(directory):
        return downloaded

    for root, _, filenames in os.walk(directory):
        for fname in filenames:
            filepath = os.path.join(root, fname)
            if os.path.isfile(filepath) and os.path.getsize(filepath) > 0:
                match = re.match(r"^bid_(\d+)(?:_.*|\..*)$", fname)
                if match:
                    downloaded.add(int(match.group(1)))
    return downloaded


def load_downloaded_bid_ids(manifest_path):
    """Load IDs from previous successful downloads.

    The manifest stays in place when older PDFs are deleted, preventing those
    bids from being downloaded again on later scheduled runs.
    """
    if not os.path.exists(manifest_path):
        return set()

    try:
        with open(manifest_path, "r", encoding="utf-8") as manifest_file:
            manifest = json.load(manifest_file)
        return {int(bid_id) for bid_id in manifest.get("downloaded_bid_ids", [])}
    except (OSError, ValueError, TypeError) as exc:
        print(f"[Manifest] Could not read {manifest_path}: {exc}. Rebuilding from local files.")
        return set()


def load_bid_records(manifest_path):
    """Return per-bid provenance used by the chunker and New-result badge."""
    if not os.path.exists(manifest_path):
        return {}
    try:
        with open(manifest_path, "r", encoding="utf-8") as manifest_file:
            records = json.load(manifest_file).get("bid_records", {})
        return records if isinstance(records, dict) else {}
    except (OSError, ValueError, TypeError):
        return {}


def save_downloaded_bid_ids(manifest_path, bid_ids, bid_records=None):
    """Atomically persist IDs and provenance for successful downloads."""
    os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
    if bid_records is None:
        bid_records = load_bid_records(manifest_path)
    manifest = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "downloaded_bid_ids": sorted(bid_ids),
        "bid_records": bid_records,
    }
    temporary_path = f"{manifest_path}.tmp"
    with open(temporary_path, "w", encoding="utf-8") as manifest_file:
        json.dump(manifest, manifest_file, indent=2)
    os.replace(temporary_path, manifest_path)


def cleanup_zero_byte_files(directory):
    """Remove empty residual files from failed downloads."""
    if not os.path.exists(directory):
        return
    deleted_count = 0
    for fname in os.listdir(directory):
        filepath = os.path.join(directory, fname)
        if os.path.isfile(filepath) and os.path.getsize(filepath) == 0:
            try:
                os.remove(filepath)
                deleted_count += 1
            except Exception:
                pass
    if deleted_count > 0:
        print(f"[Cleanup] Deleted {deleted_count} empty (0-byte) files.")


def download_single_document(session, bid, total_tasks, destination_dir):
    """Downloads a single PDF document."""
    global progress_counter, success_counter, failed_counter

    bid_id = bid["bid_id"]
    url = f"{BASE_URL}/showbidDocument/{bid_id}"

    try:
        resp = session.get(url, timeout=60, stream=True)
        resp.raise_for_status()

        cd = resp.headers.get("Content-Disposition", "")
        filename = None
        if cd:
            fname_match = re.search(r'filename[^;=\n]*=(["\']?)(.+?)\1(;|$)', cd)
            if fname_match:
                filename = fname_match.group(2).strip()
            else:
                fname_match_star = re.search(r"filename\*=(?:UTF-8''|utf-8'')(.+)", cd)
                if fname_match_star:
                    filename = requests.utils.unquote(fname_match_star.group(1).strip())

        if not filename:
            ct = resp.headers.get("Content-Type", "")
            ext = ".pdf" if "pdf" in ct else ".zip" if "zip" in ct else ".html" if "html" in ct else ".bin"
            filename = f"document{ext}"

        filename = re.sub(r'[\\/*?:"<>|]', "_", filename)
        save_name = f"bid_{bid_id}_{filename}"
        filepath = os.path.join(destination_dir, save_name)

        with open(filepath, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

        file_size_kb = os.path.getsize(filepath) / 1024

        with print_lock:
            progress_counter += 1
            if file_size_kb > 0:
                success_counter += 1
                print(f"  [{progress_counter:04d}/{total_tasks:04d}] Bid {bid_id}: OK ({file_size_kb:.1f} KB)")
                downloaded_at = datetime.now(INDIA_TZ).isoformat()
                return {
                    "bid_id": bid_id,
                    "bid_number": bid.get("bid_number", ""),
                    "pdf_path": os.path.relpath(filepath, BIDS_DIR).replace(os.sep, "/"),
                    "download_date": downloaded_at[:10],
                    "downloaded_at": downloaded_at,
                }
            else:
                failed_counter += 1
                print(f"  [{progress_counter:04d}/{total_tasks:04d}] Bid {bid_id}: FAILED (0 bytes)")
                return False

    except Exception as e:
        with print_lock:
            progress_counter += 1
            failed_counter += 1
            print(f"  [{progress_counter:04d}/{total_tasks:04d}] Bid {bid_id}: FAILED ({e})")
        return False


def main():
    global progress_counter, success_counter, failed_counter

    print("=" * 70)
    print("GeM Bid Downloader -- Incremental Sync Engine")
    print("Mode: Full available-Bid sync" if FULL_SYNC else "Mode: New bids only (Bid Start Date: Latest First)")
    if MAX_NEW_DOWNLOADS_PER_RUN:
        print(f"Run limit: {MAX_NEW_DOWNLOADS_PER_RUN} new bid(s)")
    print("=" * 70)

    os.makedirs(BIDS_DIR, exist_ok=True)
    run_date = datetime.now(INDIA_TZ).date().isoformat()
    destination_dir = os.path.join(BIDS_DIR, run_date)
    os.makedirs(destination_dir, exist_ok=True)
    cleanup_zero_byte_files(destination_dir)

    # The disk scan seeds the manifest on the first upgraded run. Thereafter
    # the manifest also remembers files that a later cleanup has deleted.
    disk_bid_ids = get_downloaded_bid_ids(BIDS_DIR)
    downloaded_bids = load_downloaded_bid_ids(MANIFEST_PATH)
    bid_records = load_bid_records(MANIFEST_PATH)
    manifest_count_before_seed = len(downloaded_bids)
    downloaded_bids.update(disk_bid_ids)
    if len(downloaded_bids) != manifest_count_before_seed:
        save_downloaded_bid_ids(MANIFEST_PATH, downloaded_bids, bid_records)
        print(f"[Manifest] Seeded with {len(disk_bid_ids)} IDs found in existing files.")
    print(f"[Storage] Tracking {len(downloaded_bids)} downloaded bid IDs "
          f"({len(disk_bid_ids)} PDFs currently on disk).")

    session, csrf = get_session_and_csrf()
    adapter = requests.adapters.HTTPAdapter(pool_connections=MAX_WORKERS * 2, pool_maxsize=MAX_WORKERS * 2)
    session.mount("http://", adapter)
    session.mount("https://", adapter)

    page_num = 1
    start_time = time.time()
    no_more_pages = False
    early_stop_triggered = False
    consecutive_known_pages = 0
    new_downloads_this_run = 0

    while (
        not no_more_pages
        and not early_stop_triggered
        and (not MAX_NEW_DOWNLOADS_PER_RUN or new_downloads_this_run < MAX_NEW_DOWNLOADS_PER_RUN)
    ):
        candidates = []
        pages_to_fetch = 10
        print(f"\nScanning pages {page_num} to {page_num + pages_to_fetch - 1}...")

        for p in range(page_num, page_num + pages_to_fetch):
            try:
                page_bids, raw_doc_count, csrf, session = fetch_page_with_retry(session, csrf, p)
            except Exception as e:
                print(f"  Skipping page {p} after failures: {e}")
                continue

            if raw_doc_count == 0:
                no_more_pages = True
                break

            candidates.extend(page_bids)

            # Early stop tracking logic
            if page_bids:
                page_has_new = any(b["bid_id"] not in downloaded_bids for b in page_bids)
                if page_has_new:
                    consecutive_known_pages = 0
                else:
                    consecutive_known_pages += 1
                    if not FULL_SYNC and consecutive_known_pages >= EARLY_STOP_LIMIT:
                        print(
                            f"\n[Early Stop] Encountered {consecutive_known_pages} consecutive pages with zero new bids. Stopping scan.")
                        early_stop_triggered = True
                        break

            time.sleep(0.3)

        page_num += pages_to_fetch

        to_download = [bid for bid in candidates if bid["bid_id"] not in downloaded_bids]
        if MAX_NEW_DOWNLOADS_PER_RUN:
            remaining = MAX_NEW_DOWNLOADS_PER_RUN - new_downloads_this_run
            to_download = to_download[:remaining]

        if to_download:
            print(f"\n[Batch] Found {len(to_download)} new bids. Downloading with {MAX_WORKERS} workers...")
            progress_counter = 0
            success_counter = 0
            failed_counter = 0

            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                futures = {
                    executor.submit(download_single_document, session, bid, len(to_download), destination_dir): bid
                    for bid in to_download
                }
                for future in as_completed(futures):
                    bid_obj = futures[future]
                    download_record = future.result()
                    if download_record:
                        downloaded_bids.add(bid_obj["bid_id"])
                        bid_records[str(bid_obj["bid_id"])] = download_record
                        new_downloads_this_run += 1

            # Save immediately after each successful batch, so a later
            # cleanup or interrupted run cannot cause re-downloads.
            save_downloaded_bid_ids(MANIFEST_PATH, downloaded_bids, bid_records)
            print(f"[Manifest] Recorded {new_downloads_this_run} new successful download(s) this run.")

            time.sleep(1)

        if no_more_pages:
            print("[Info] Reached end of available portal pages.")
            break

    duration = time.time() - start_time
    final_count = len(get_downloaded_bid_ids(BIDS_DIR))
    print("\n" + "=" * 70)
    print("Process Complete!")
    print(f"  Total Duration: {duration:.1f} seconds (~{duration / 60:.1f} mins)")
    print(f"  New Files Downloaded: {new_downloads_this_run}")
    print(f"  PDFs Currently Saved: {final_count} files")
    print(f"  Processed IDs Tracked: {len(downloaded_bids)}")
    print(f"  Manifest: {MANIFEST_PATH}")
    print(f"  Today's folder: {destination_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
