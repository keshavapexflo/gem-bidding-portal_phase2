"""
GeM Bid PDF -> Metadata + Chunks (for embedding)
==================================================
Extracts per-bid metadata (matching the BID_Scraper sheet schema, plus
ministry/organisation as hard-filter fields) and splits each bid's PDF text
into embedding-ready chunks:

  overview               - 1 synthesized summary sentence/paragraph
  eligibility_financial  - turnover / EMD / experience / past performance
  scope_of_work          - item/category description (the actual procurement)
  atc_clause             - one chunk per top-level numbered ATC item
  preference_policy      - MII/MSE specifics, ONLY if the bid has real
                            numbers (price band %, max qty %) -- generic
                            boilerplate explanation paragraphs are skipped

Every chunk carries the same `metadata` dict, so Chroma queries can filter
on Bid ID / organisation / ministry / dates etc. before or after the
semantic search.

Output:
  bid_chunks.json     - flat list of chunk records, ready to embed
  bids_metadata.json  - one metadata record per bid (for quick lookup /
                         building a filter UI without re-scanning chunks)

Resume support: each bid's PDF content hash is tracked; unchanged PDFs are
skipped on re-runs so re-processing the whole corpus after a fresh
downloader run only costs time for genuinely new/changed bids.
"""

import os
import re
import json
import hashlib
import sys
from datetime import datetime, timezone
import fitz  # PyMuPDF

BASE_URL = "https://bidplus.gem.gov.in"

if sys.stdout:
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass


def utcnow_iso():
    return datetime.now(timezone.utc).isoformat()


def calculate_file_hash(filepath):
    hash_md5 = hashlib.md5()
    try:
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    except Exception:
        return None


def parse_date(date_str):
    if not date_str:
        return None
    for fmt in ("%d-%m-%y", "%d-%m-%Y"):
        try:
            return datetime.strptime(date_str.strip(), fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return date_str


# ---------------------------------------------------------------------------
# Small field-finder helpers (line-scan regex, same approach as the CSV
# extractor -- GeM PDFs are consistently "Label" then value on next line(s))
# ---------------------------------------------------------------------------

def find_datetime_after_keyword(lines, keyword):
    for idx, line in enumerate(lines):
        if keyword in line:
            for offset in range(1, 4):
                if idx + offset < len(lines):
                    m = re.search(r'\d{2}-\d{2}-\d{4}\s+\d{2}:\d{2}:\d{2}', lines[idx + offset].strip())
                    if m:
                        return m.group(0)
    return None


def find_total_quantity(lines):
    for idx, line in enumerate(lines):
        if "Total Quantity" in line:
            for offset in range(1, 3):
                if idx + offset < len(lines):
                    clean = lines[idx + offset].strip().replace(",", "")
                    if clean.isdigit():
                        return int(clean)
    return None


def find_item_category(lines):
    """The item/category description block -- doubles as both 'Bid Title'
    (short proxy) and the scope_of_work chunk's source text (full block)."""
    for idx, line in enumerate(lines):
        if "Item Category" in line:
            category_lines = []
            for offset in range(1, 10):
                if idx + offset < len(lines):
                    nxt = lines[idx + offset].strip()
                    if any(t in nxt for t in ["GeMARPTS", "Turnover", "Experience", "Dated", "Bid Number"]):
                        break
                    if nxt:
                        category_lines.append(nxt)
            return " , ".join(category_lines).strip()
    return None


def parse_turnover(lines, keyword):
    for idx, line in enumerate(lines):
        if keyword in line:
            for offset in range(1, 4):
                if idx + offset < len(lines):
                    m = re.search(r'([\d\.,]+)\s*(Lakh|Crore|Thousand)?', lines[idx + offset].strip(), re.IGNORECASE)
                    if m:
                        try:
                            return {"value": float(m.group(1).replace(",", "")), "unit": m.group(2) or "Lakh"}
                        except ValueError:
                            pass
    return {"value": None, "unit": "Lakh"}


def find_experience_years(lines):
    for idx, line in enumerate(lines):
        if "Past Experience Required" in line:
            for offset in range(1, 4):
                if idx + offset < len(lines):
                    m = re.search(r'(\d+)\s*Year', lines[idx + offset].strip(), re.IGNORECASE)
                    if m:
                        return int(m.group(1))
    return None


def find_past_performance(lines):
    for idx, line in enumerate(lines):
        if "Past Performance" in line:
            for offset in range(1, 4):
                if idx + offset < len(lines):
                    m = re.search(r'(\d+)\s*%', lines[idx + offset].strip())
                    if m:
                        return int(m.group(1))
    return None


def find_emd_details(lines):
    emd_amount, advisory_bank, required = None, None, True
    for idx, line in enumerate(lines):
        if "EMD Detail" in line:
            for offset in range(1, 5):
                if idx + offset < len(lines) and "Required" in lines[idx + offset]:
                    if idx + offset + 1 < len(lines) and lines[idx + offset + 1].strip().lower() == "no":
                        required = False
            if not required:
                break
            for offset in range(1, 10):
                if idx + offset < len(lines):
                    l = lines[idx + offset].strip()
                    if "Advisory Bank" in l:
                        advisory_bank = lines[idx + offset + 1].strip()
                    elif "EMD Amount" in l:
                        m = re.search(r'\d+', lines[idx + offset + 1].strip().replace(",", ""))
                        if m:
                            try:
                                emd_amount = float(m.group(0))
                            except ValueError:
                                pass
            break
    return emd_amount, advisory_bank


def find_epbg_details(lines):
    epbg_percent, epbg_months, advisory_bank, required = None, None, None, True
    for idx, line in enumerate(lines):
        if "ePBG Detail" in line:
            for offset in range(1, 5):
                if idx + offset < len(lines) and "Required" in lines[idx + offset]:
                    if idx + offset + 1 < len(lines) and lines[idx + offset + 1].strip().lower() == "no":
                        required = False
            if not required:
                break
            for offset in range(1, 12):
                if idx + offset < len(lines):
                    l = lines[idx + offset].strip()
                    if "Advisory Bank" in l:
                        advisory_bank = lines[idx + offset + 1].strip()
                    elif "ePBG Percentage" in l:
                        m = re.search(r'[\d\.]+', lines[idx + offset + 1].strip().replace(",", ""))
                        if m:
                            try:
                                epbg_percent = float(m.group(0))
                            except ValueError:
                                pass
                    elif "Duration of ePBG" in l:
                        m = re.search(r'\d+', lines[idx + offset + 1].strip().replace(",", ""))
                        if m:
                            try:
                                epbg_months = int(m.group(0))
                            except ValueError:
                                pass
            break
    return required, epbg_percent, epbg_months, advisory_bank


def find_percentage_after_keyword(lines, keyword):
    for idx, line in enumerate(lines):
        if keyword in line:
            for offset in range(1, 5):
                if idx + offset < len(lines):
                    m = re.search(r'\d+', lines[idx + offset].strip())
                    if m:
                        return int(m.group(0))
    return None


def find_purchase_preference(lines, keyword):
    for idx, line in enumerate(lines):
        if keyword in line:
            for offset in range(1, 4):
                if idx + offset < len(lines):
                    v = lines[idx + offset].strip().lower()
                    if "yes" in v:
                        return True
                    elif "no" in v:
                        return False
    return False


def find_local_content_pct(text):
    m = re.search(r'Minimum\s*(\d+)%\s*and\s*(\d+)%\s*Local\s*Content', text, re.IGNORECASE)
    return (int(m.group(1)), int(m.group(2))) if m else (None, None)


def extract_atc_clauses(text):
    """Split 'Buyer Added Bid Specific Terms and Conditions' into one chunk
    per top-level numbered item (nested lettered/roman sub-clauses stay as
    flat text within their parent item -- see project notes on why)."""
    atc_start = -1
    for marker in ["Buyer Added Bid Specific Terms and Conditions", "Buyer Added Bid Specific ATC"]:
        idx = text.find(marker)
        if idx != -1:
            atc_start = idx
            break
    if atc_start == -1:
        return []

    atc_text = text[atc_start:]
    for marker in ["Disclaimer", "Thank You"]:
        idx = atc_text.find(marker)
        if idx != -1:
            atc_text = atc_text[:idx]
            break

    lines = atc_text.split('\n')
    clauses = []
    current = None
    header_pat = re.compile(r'^\s*(\d+)\.\s*([A-Z][a-zA-Z\s\-\(\)/&]+)(?::|$)$')

    for line in lines:
        stripped = line.strip()
        m = header_pat.match(stripped)
        if m:
            if current:
                clauses.append(current)
            current = {"clause_no": int(m.group(1)), "heading": m.group(2).strip(), "text_lines": []}
        elif current:
            current["text_lines"].append(line)
    if current:
        clauses.append(current)

    return [
        {"clause_no": c["clause_no"], "heading": c["heading"], "text": "\n".join(c["text_lines"]).strip()}
        for c in clauses
        if "\n".join(c["text_lines"]).strip()  # drop empty clauses
    ]


# ---------------------------------------------------------------------------
# Metadata extraction (matches the 12-column sheet + ministry/organisation)
# ---------------------------------------------------------------------------

def extract_metadata(full_text, lines, pdf_path, manifest_entry=None, pdf_path_value=None):
    bid_match = re.search(r'GEM/\d{4}/[BR]/\d+', full_text)
    bid_id = bid_match.group(0) if bid_match else None
    if not bid_id:
        fname_match = re.search(r'(\d+)', os.path.basename(pdf_path))
        bid_id = f"GEM/UNKNOWN/B/{fname_match.group(1)}" if fname_match else "GEM/UNKNOWN/B/0000000"

    pub_match = re.search(r'Dated:\s*(\d{2}-\d{2}-\d{4})', full_text, re.IGNORECASE)
    published_date = parse_date(pub_match.group(1)) if pub_match else None

    bid_start = find_datetime_after_keyword(lines, "Bid Start Date/Time")
    bid_end = find_datetime_after_keyword(lines, "Bid End Date/Time")

    ministry = department = organisation = None
    for idx, line in enumerate(lines):
        if "Ministry/State Name" in line:
            ministry = lines[idx + 1].strip()
        elif "Department Name" in line:
            department = lines[idx + 1].strip()
        elif "Organisation Name" in line:
            organisation = lines[idx + 1].strip()

    item_category = find_item_category(lines)
    quantity = find_total_quantity(lines)
    emd_amount, _ = find_emd_details(lines)

    internal_id = (manifest_entry or {}).get("bid_id")
    is_ra = "/R/" in (bid_id or "")
    pdf_link = (f"{BASE_URL}/{'showradocumentPdf' if is_ra else 'showbidDocument'}/{internal_id}"
                if internal_id else os.path.basename(pdf_path))

    return {
        "bid_id": bid_id,
        "portal": "GeM",
        "date": (manifest_entry or {}).get("download_date") or utcnow_iso(),
        # ``added_at`` is when this document first entered our portal, not
        # the original GeM publication date. The Streamlit UI uses it for a
        # clear New badge on recently ingested bids.
        "added_at": (manifest_entry or {}).get("downloaded_at")
            or (manifest_entry or {}).get("download_date")
            or utcnow_iso(),
        "client_organization": organisation or department,
        "ministry": ministry,             # hard-filter field, not in the sheet but needed for exact matching
        "organisation": organisation,     # kept distinct from client_organization for filter precision
        "bid_title": item_category,
        "quantity": quantity,
        "emd_amount": emd_amount,
        "delivery_period": None,          # filled in from schedule/consignee parsing if present
        "start_date_to_submit_bid": bid_start,
        "end_date_to_submit_bid": bid_end,
        "pdf_link": pdf_link,
        "published_date": published_date,
        "document_hash": calculate_file_hash(pdf_path),
        "pdf_path": pdf_path_value or os.path.basename(pdf_path),
    }


# ---------------------------------------------------------------------------
# Chunk builders
# ---------------------------------------------------------------------------

def build_overview_chunk(meta):
    parts = []
    if meta.get("bid_title"):
        parts.append(f"Procurement of {meta['bid_title']}.")
    if meta.get("client_organization"):
        org_line = f"Required by {meta['client_organization']}"
        if meta.get("ministry"):
            org_line += f" ({meta['ministry']})"
        parts.append(org_line + ".")
    if meta.get("quantity"):
        parts.append(f"Quantity: {meta['quantity']}.")
    if meta.get("emd_amount"):
        parts.append(f"EMD Amount: {meta['emd_amount']}.")
    if meta.get("end_date_to_submit_bid"):
        parts.append(f"Bid closes: {meta['end_date_to_submit_bid']}.")
    text = " ".join(parts).strip()
    return {"chunk_type": "overview", "text": text} if text else None


def build_eligibility_financial_chunk(lines):
    bidder_turnover = parse_turnover(lines, "Minimum Average Annual Turnover of the bidder")
    oem_turnover = parse_turnover(lines, "OEM Average Turnover")
    experience_years = find_experience_years(lines)
    past_performance_pct = find_past_performance(lines)
    emd_amount, advisory_bank = find_emd_details(lines)
    epbg_required, epbg_percent, epbg_months, epbg_bank = find_epbg_details(lines)

    parts = []
    if bidder_turnover["value"]:
        parts.append(f"Minimum average annual turnover required: {bidder_turnover['value']} {bidder_turnover['unit']}.")
    if oem_turnover["value"]:
        parts.append(f"OEM average turnover required: {oem_turnover['value']} {oem_turnover['unit']}.")
    if experience_years:
        parts.append(f"Past experience required: {experience_years} years.")
    if past_performance_pct is not None:
        parts.append(f"Past performance requirement: {past_performance_pct}%.")
    if emd_amount:
        parts.append(f"EMD amount: {emd_amount}" + (f", advisory bank {advisory_bank}." if advisory_bank else "."))
    if epbg_required and epbg_percent:
        parts.append(f"ePBG required: {epbg_percent}% for {epbg_months} months.")

    text = " ".join(parts).strip()
    return {"chunk_type": "eligibility_financial", "text": text} if text else None


def build_scope_of_work_chunk(item_category):
    if not item_category:
        return None
    return {"chunk_type": "scope_of_work", "text": item_category}


def build_atc_chunks(full_text):
    clauses = extract_atc_clauses(full_text)
    return [
        {"chunk_type": "atc_clause", "text": f"{c['heading']}: {c['text']}", "clause_no": c["clause_no"]}
        for c in clauses
    ]


def build_preference_policy_chunk(lines, full_text):
    """Only emitted when the bid has real MII/MSE numbers -- generic
    boilerplate explanation paragraphs (present in almost every bid) are
    excluded per the project's earlier boilerplate-stripping decision."""
    local1, local2 = find_local_content_pct(full_text)
    mii_pref = find_purchase_preference(lines, "MII Purchase Preference")
    mii_band = find_percentage_after_keyword(lines, "Preference to MII sellers availabele upto price within L1+X%")
    mii_qty = find_percentage_after_keyword(lines, "Maximum Percentage of Bid quantity for MII purchase preference")
    mse_pref = find_purchase_preference(lines, "MSE Purchase Preference")
    mse_band = find_percentage_after_keyword(lines, "Purchase Preference to MSE OEMs available upto price within L1+X%")
    mse_qty = find_percentage_after_keyword(lines, "Percentage of Bid quantity/amount for MSE")

    has_real_numbers = any([local1, local2, mii_band, mii_qty, mse_band, mse_qty])
    if not has_real_numbers:
        return None

    parts = []
    if local1 or local2:
        parts.append(f"Local content requirement: minimum {local1}% (Class I) / {local2}% (Class II).")
    if mii_pref and mii_band:
        parts.append(f"MII purchase preference applies up to L1+{mii_band}%, max {mii_qty}% of quantity." if mii_qty else f"MII purchase preference applies up to L1+{mii_band}%.")
    if mse_pref and mse_band:
        parts.append(f"MSE purchase preference applies up to L1+{mse_band}%, max {mse_qty}% of quantity." if mse_qty else f"MSE purchase preference applies up to L1+{mse_band}%.")

    text = " ".join(parts).strip()
    return {"chunk_type": "preference_policy", "text": text} if text else None


# ---------------------------------------------------------------------------
# Per-PDF orchestration
# ---------------------------------------------------------------------------

def extract_and_chunk_pdf(pdf_path, manifest_entry=None, pdf_path_value=None):
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        print(f"  Error opening {pdf_path}: {e}")
        return None, None

    full_text = "\n".join(page.get_text() for page in doc)
    doc.close()
    lines = [l.strip() for l in full_text.split("\n")]

    meta = extract_metadata(full_text, lines, pdf_path, manifest_entry, pdf_path_value)

    raw_chunks = []
    overview = build_overview_chunk(meta)
    if overview:
        raw_chunks.append(overview)

    elig_fin = build_eligibility_financial_chunk(lines)
    if elig_fin:
        raw_chunks.append(elig_fin)

    scope = build_scope_of_work_chunk(meta.get("bid_title"))
    if scope:
        raw_chunks.append(scope)

    raw_chunks.extend(build_atc_chunks(full_text))

    pref = build_preference_policy_chunk(lines, full_text)
    if pref:
        raw_chunks.append(pref)

    chunks = []
    for i, c in enumerate(raw_chunks):
        chunk_id = f"{meta['bid_id']}::{c['chunk_type']}"
        if c['chunk_type'] == "atc_clause":
            chunk_id += f"::{c.get('clause_no', i)}"
        chunks.append({
            "chunk_id": chunk_id,
            "bid_id": meta["bid_id"],
            "chunk_type": c["chunk_type"],
            "text": c["text"],
            "metadata": meta,
        })

    return meta, chunks


# ---------------------------------------------------------------------------
# Directory processing with resume-by-hash
# ---------------------------------------------------------------------------

def load_manifest(manifest_path):
    if not manifest_path or not os.path.exists(manifest_path):
        return {}
    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except Exception:
        return {}
    # Current downloader format: {"bid_records": {internal_id: {...}}}.
    # Keep support for the older mapping-only format as well.
    records = raw.get("bid_records") if isinstance(raw, dict) else None
    if isinstance(records, dict):
        return {
            str(entry.get("bid_id") or key): entry
            for key, entry in records.items()
            if isinstance(entry, dict)
        }
    return {
        str(entry.get("bid_id") or key): entry
        for key, entry in raw.items()
        if isinstance(entry, dict)
    }


def load_existing_chunks_state(state_path):
    """Returns {bid_id: {document_hash, chunks: [...], metadata: {...}}}"""
    if not os.path.exists(state_path):
        return {}
    try:
        with open(state_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return {rec["bid_id"]: rec for rec in data.get("bids", [])}
    except Exception as e:
        print(f"Warning: could not load existing chunk state ({e}); starting fresh.")
        return {}


def process_directory(bids_dir, manifest_path, output_path, sync_path=None):
    manifest_by_bid_number = load_manifest(manifest_path)
    prev_state = load_existing_chunks_state(output_path)

    bids_out = {}
    skipped = 0
    processed = 0
    upsert_chunk_ids = []
    deleted_chunk_ids = []

    files = []
    if os.path.exists(bids_dir):
        for root, _, filenames in os.walk(bids_dir):
            files.extend(os.path.join(root, name) for name in filenames if name.lower().endswith(".pdf"))
    print(f"Found {len(files)} PDFs in {bids_dir}")

    for idx, pdf_path in enumerate(files):
        relative_pdf_path = os.path.relpath(pdf_path, bids_dir).replace(os.sep, "/")
        fname = os.path.basename(pdf_path)
        current_hash = calculate_file_hash(pdf_path)

        prev = next((rec for rec in prev_state.values()
                     if rec.get("metadata", {}).get("pdf_path") == relative_pdf_path), None)
        if prev and prev.get("metadata", {}).get("document_hash") == current_hash:
            bids_out[prev["bid_id"]] = prev
            skipped += 1
            continue

        m = re.match(r"^bid_(\d+)_", fname)
        internal_id_guess = m.group(1) if m else None
        manifest_entry = manifest_by_bid_number.get(internal_id_guess) if internal_id_guess else None

        print(f"  [{idx + 1}/{len(files)}] {fname} ...", end=" ", flush=True)
        meta, chunks = extract_and_chunk_pdf(pdf_path, manifest_entry, relative_pdf_path)
        if not meta:
            print("FAILED")
            continue

        bids_out[meta["bid_id"]] = {"bid_id": meta["bid_id"], "metadata": meta, "chunks": chunks}
        upsert_chunk_ids.extend(chunk["chunk_id"] for chunk in chunks)
        if prev:
            new_ids = {chunk["chunk_id"] for chunk in chunks}
            deleted_chunk_ids.extend(
                chunk["chunk_id"] for chunk in prev.get("chunks", [])
                if chunk.get("chunk_id") not in new_ids
            )
        processed += 1
        print(f"OK ({len(chunks)} chunks)")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"bids": list(bids_out.values())}, f, indent=2, ensure_ascii=False)

    # If a PDF disappeared outside the normal expiry process, also record
    # its old vectors for removal on the next embedding sync.
    removed_bid_ids = set(prev_state) - set(bids_out)
    for bid_id in removed_bid_ids:
        deleted_chunk_ids.extend(
            chunk["chunk_id"] for chunk in prev_state[bid_id].get("chunks", [])
            if chunk.get("chunk_id")
        )

    if sync_path:
        sync_data = {
            "generated_at": utcnow_iso(),
            "upsert_chunk_ids": upsert_chunk_ids,
            "deleted_chunk_ids": deleted_chunk_ids,
            "removed_bid_ids": sorted(removed_bid_ids),
        }
        with open(sync_path, "w", encoding="utf-8") as f:
            json.dump(sync_data, f, indent=2)

    total_chunks = sum(len(b["chunks"]) for b in bids_out.values())
    print(f"\nDone. {len(bids_out)} bids ({processed} newly processed, {skipped} skipped-unchanged), "
          f"{total_chunks} total chunks -> {output_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Extract metadata + chunks from GeM bid PDFs.")
    parser.add_argument("--bids-dir", default=r"D:\lets bid\downloads\bids")
    parser.add_argument("--manifest", default=r"D:\lets bid\downloads\downloaded_bid_manifest.json",
                         help="Optional downloader manifest. Fine if missing -- pdf_link falls back "
                              "to the local filename and date falls back to processing time.")
    parser.add_argument("--output", default=r"D:\lets bid\bid_chunks.json")
    parser.add_argument("--sync-output", default=r"D:\lets bid\downloads\chunk_sync.json",
                        help="Changed and removed chunk IDs for the embedding sync.")
    args = parser.parse_args()

    process_directory(args.bids_dir, args.manifest, args.output, args.sync_output)
