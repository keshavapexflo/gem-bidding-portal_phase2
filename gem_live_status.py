"""
Checks whether bids have been extended (end date changed) by querying
GeM's live search API and comparing against what's stored in your Chroma
metadata. Reuses the same session/CSRF flow as your downloader.

Confirmed field from peek_gem_bid_status.py: GeM's live doc includes
final_end_date_sort (ISO UTC, e.g. "2026-08-03T09:00:00Z") and b_is_inactive
(0/1, whether the bid has closed). No explicit corrigendum counter is
exposed in this schema, so end-date comparison is the signal used.
"""

import json
import re
import time
from datetime import datetime

import requests

BASE_URL = "https://bidplus.gem.gov.in"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
}


def get_session_and_csrf():
    session = requests.Session()
    session.headers.update(HEADERS)
    resp = session.get(f"{BASE_URL}/all-bids", timeout=30)
    resp.raise_for_status()
    match = re.search(r"csrf_bd_gem_nk['\"\s:]+['\"]([a-f0-9]+)['\"]", resp.text)
    if not match:
        raise ValueError("CSRF token not found in page source")
    return session, match.group(1)


def search_exact_bid(session, csrf, bid_number, timeout=30):
    payload = {
        "page": 1,
        "param": {"searchBid": bid_number, "searchType": "fullText"},
        "filter": {
            "bidStatusType": "ongoing_bids",
            "byType": "all",
            "highBidValue": "",
            "byEndDate": {"from": "", "to": ""},
            "sort": "Bid-End-Date-Oldest",
        },
    }
    resp = session.post(
        f"{BASE_URL}/all-bids-data",
        data={"payload": json.dumps(payload), "csrf_bd_gem_nk": csrf},
        headers={
            "Referer": f"{BASE_URL}/all-bids",
            "X-Requested-With": "XMLHttpRequest",
            "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
        },
        timeout=timeout,
    )
    resp.raise_for_status()
    data = resp.json()
    return data["response"]["response"]["docs"]


def _parse_live_end_date_ist(doc):
    """Parse the portal's end date in the same displayed time basis as PDFs.

    Although ``final_end_date_sort`` carries a trailing ``Z``, GeM's Bid
    Listing UI presents that clock value directly (for example, a value ending
    in ``T21:00:00Z`` is shown as ``9:00 PM``). Adding an IST offset therefore
    creates a false 5½-hour extension for otherwise unchanged bids.
    """
    raw = doc.get("final_end_date_sort", [None])[0]
    if not raw:
        return None
    return datetime.strptime(raw, "%Y-%m-%dT%H:%M:%SZ")


def _parse_stored_end_date(stored_str):
    """Stored end dates in your metadata have appeared in a couple of
    formats depending on extraction path -- try the common ones."""
    if not stored_str or str(stored_str).strip() in ("", "N/A"):
        return None
    stored_str = str(stored_str).strip()
    formats = ["%d-%m-%Y %H:%M:%S", "%Y-%m-%d %H:%M:%S", "%d-%m-%Y", "%Y-%m-%d"]
    for fmt in formats:
        try:
            return datetime.strptime(stored_str, fmt)
        except ValueError:
            continue
    return None


def check_bids_for_extension(bid_stored_end_dates: dict, delay_seconds=0.4, progress_callback=None):
    """
    bid_stored_end_dates: {bid_number: stored_end_date_str}
    Returns: {bid_number: {"live_end_date": str|None, "extended": bool|None,
                            "status": "ok"|"not_found"|"error", "b_is_inactive": int|None}}

    "extended" is None when we can't determine (missing/unparseable stored
    date, or bid not found live -- e.g. it may have already closed and
    dropped out of the ongoing_bids filter).
    """
    session, csrf = get_session_and_csrf()
    results = {}
    total = len(bid_stored_end_dates)

    for i, (bid_number, stored_end_str) in enumerate(bid_stored_end_dates.items(), start=1):
        try:
            docs = search_exact_bid(session, csrf, bid_number)
            matching = [d for d in docs if d.get("b_bid_number", [None])[0] == bid_number]

            if not matching:
                results[bid_number] = {
                    "live_end_date": None,
                    "extended": None,
                    "status": "not_found",
                    "b_is_inactive": None,
                }
            else:
                doc = matching[0]
                live_end = _parse_live_end_date_ist(doc)
                stored_end = _parse_stored_end_date(stored_end_str)
                is_inactive = doc.get("b_is_inactive", [None])[0]

                if live_end is None or stored_end is None:
                    extended = None
                else:
                    # Compare at minute resolution -- ignore trivial second
                    # differences from parsing/rounding
                    extended = live_end.replace(second=0) != stored_end.replace(second=0)

                results[bid_number] = {
                    "live_end_date": live_end.strftime("%d-%m-%Y %H:%M:%S") if live_end else None,
                    "extended": extended,
                    "status": "ok",
                    "b_is_inactive": is_inactive,
                }
        except Exception as e:
            results[bid_number] = {
                "live_end_date": None,
                "extended": None,
                "status": f"error: {e}",
                "b_is_inactive": None,
            }

        if progress_callback:
            progress_callback(i, total, bid_number)

        time.sleep(delay_seconds)  # be polite to GeM's servers

    return results
