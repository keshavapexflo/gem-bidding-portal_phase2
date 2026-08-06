import csv
import io
import json
from datetime import datetime, timezone
from html import escape
from pathlib import Path
import re
import traceback

from gem_hybrid_retrieval import (
    DEFAULT_CHROMA_PATH,
    DEFAULT_COLLECTION,
    HybridRetriever,
    SearchResult,
)
from gem_live_status import check_bids_for_extension
import streamlit as st

NEW_BID_WINDOW_DAYS = 7


def is_new_bid(metadata) -> bool:
  """A bid is New for one week after this portal first downloaded it."""
  added_at = metadata.get("added_at")
  if not added_at:
    return False
  try:
    parsed = datetime.fromisoformat(str(added_at).replace("Z", "+00:00"))
    if parsed.tzinfo is None:
      parsed = parsed.replace(tzinfo=timezone.utc)
    age_seconds = (datetime.now(timezone.utc) - parsed.astimezone(timezone.utc)).total_seconds()
    return 0 <= age_seconds < NEW_BID_WINDOW_DAYS * 24 * 60 * 60
  except (TypeError, ValueError):
    return False


def find_pdf_file(metadata) -> Path | None:
  # 1. Try finding by pdf_path in metadata
  pdf_path_val = metadata.get("pdf_path") or metadata.get("pdf_link")
  if pdf_path_val:
    p = Path("downloads/bids") / pdf_path_val
    if p.exists():
      return p

  # 2. Try finding by bid_id numeric part
  bid_id = metadata.get("bid_id")
  if bid_id:
    digits_match = re.findall(r"\d+", bid_id)
    if digits_match:
      for digit_seq in reversed(digits_match):
        if len(digit_seq) >= 5:
          matches = list(Path("downloads/bids").glob(f"*{digit_seq}*.pdf"))
          if matches:
            return matches[0]
  return None


# Exact column order matching the user's spreadsheet format
CSV_COLUMNS = [
    "Date", "Portal", "Bid ID", "Client / Organization", "Bid Title",
    "Quantity", "EMD Amount", "Quoted Bid", "Delivery period",
    "Start date to submit bid", "End date to submit bid", "PDF Link",
]

# Extra columns added only when the live GeM extension check has been run
CSV_COLUMNS_WITH_STATUS = CSV_COLUMNS + [
    "Live End Date (GeM)", "Extended?", "Live Status",
]


def build_export_rows(results):
  """Dedupe search results by bid_id (a bid can appear multiple times
  across chunk types -- overview, atc_clause, eligibility_financial, etc.
  -- but the spreadsheet format is one row per BID, not per chunk) and map
  each bid's metadata onto the exact column set the user's sheet uses.
  First occurrence wins for dedup, since metadata is duplicated identically
  across a bid's chunks (all chunks carry the same bid-level metadata)."""
  seen_bid_ids = set()
  rows = []

  for res in results:
    meta = res.metadata
    bid_id = meta.get("bid_id", "N/A")
    if bid_id in seen_bid_ids:
      continue
    seen_bid_ids.add(bid_id)

    rows.append({
        "Date": meta.get("date", ""),
        "Portal": meta.get("portal", "GeM"),
        "Bid ID": bid_id,
        "Client / Organization": meta.get("client_organization")
            or meta.get("organisation", ""),
        "Bid Title": meta.get("bid_title", ""),
        "Quantity": meta.get("quantity", ""),
        "EMD Amount": meta.get("emd_amount", ""),
        "Quoted Bid": "",  # not scraped data -- left blank for manual entry
        "Delivery period": meta.get("delivery_period", ""),
        "Start date to submit bid": meta.get("start_date_to_submit_bid")
            or meta.get("published_date", ""),
        "End date to submit bid": meta.get("end_date_to_submit_bid", ""),
        "PDF Link": meta.get("pdf_link") or meta.get("pdf_path", ""),
    })

  return rows


def rows_to_csv_bytes(rows, fieldnames=CSV_COLUMNS) -> bytes:
  buffer = io.StringIO()
  writer = csv.DictWriter(buffer, fieldnames=fieldnames)
  writer.writeheader()
  writer.writerows(rows)
  return buffer.getvalue().encode("utf-8-sig")  # BOM so Excel/Sheets read UTF-8 correctly


# Set page configuration for a wider layout
st.set_page_config(
    page_title="Lets Bid | GeM Intelligence",
    page_icon="LB",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS styling
st.markdown(
    """
<style>
    .reportview-container {
        background: #0f1116;
    }
    .main {
        background-color: #0f1116;
        color: #e2e8f0;
    }
    .stButton>button {
        background-color: #3b82f6;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 8px 16px;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #2563eb;
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.3);
    }
    .card {
        background-color: #1e293b;
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 16px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
    }
    .card-title {
        color: #f8fafc;
        font-size: 1.15rem;
        font-weight: 600;
        margin-bottom: 8px;
    }
    .card-subtitle {
        color: #94a3b8;
        font-size: 0.85rem;
        margin-bottom: 12px;
    }
    .card-text {
        color: #cbd5e1;
        font-size: 0.95rem;
        line-height: 1.6;
        background-color: #0f172a;
        padding: 12px;
        border-radius: 8px;
        border-left: 4px solid #3b82f6;
    }
    .badge {
        display: inline-block;
        padding: 4px 8px;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-right: 8px;
    }
    .badge-high {
        background-color: #064e3b;
        color: #6ee7b7;
        border: 1px solid #047857;
    }
    .badge-med {
        background-color: #78350f;
        color: #fde047;
        border: 1px solid #b45309;
    }
    .badge-low {
        background-color: #451a03;
        color: #f97316;
        border: 1px solid #c2410c;
    }
    .badge-none {
        background-color: #1e293b;
        color: #94a3b8;
        border: 1px solid #475569;
    }
    .badge-info {
        background-color: #1e3a8a;
        color: #93c5fd;
        border: 1px solid #1d4ed8;
    }
    .badge-new {
        background-color: #14532d;
        color: #bbf7d0;
        border: 1px solid #22c55e;
    }
    :root { --ink: #e8edf5; --muted: #9ca9bd; --accent: #5b8cff; }
    .stApp { background: #0b1020; color: var(--ink); }
    [data-testid="stAppViewContainer"] > .main { background: radial-gradient(circle at 78% -20%, #1a315b 0, transparent 34%), #0b1020; }
    [data-testid="stMainBlockContainer"] { max-width: 1280px; padding-top: 2.4rem; }
    [data-testid="stSidebar"] { background: #101827; border-right: 1px solid #26344b; }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { color: var(--muted); }
    h1, h2, h3 { letter-spacing: -0.02em; }
    .hero { padding: 2rem 2.15rem; border: 1px solid #30446a; border-radius: 18px; background: linear-gradient(115deg, #152544 0%, #152039 55%, #102b40 100%); box-shadow: 0 18px 38px rgba(0, 0, 0, 0.22); margin: 0 0 2rem 0; }
    .eyebrow { color: #8fb1ff; font-size: 0.74rem; font-weight: 750; letter-spacing: 0.14em; margin-bottom: 0.5rem; }
    .hero h1 { font-size: 2.15rem; margin: 0; color: #f7f9ff; }
    .hero p { max-width: 700px; color: #c5d1e7; font-size: 1rem; line-height: 1.6; margin: 0.65rem 0 0; }
    .section-label { color: #91a9d9; font-size: 0.72rem; font-weight: 750; letter-spacing: 0.12em; margin-bottom: 0.2rem; }
    .stButton>button { background: linear-gradient(135deg, #4f7ff1, #3566d9); color: white; border-radius: 9px; border: 1px solid #6d98ff; padding: 0.55rem 1.05rem; font-weight: 650; transition: all 0.3s ease; }
    .stButton>button:hover { background: #3d71e9; transform: translateY(-1px); box-shadow: 0 7px 16px rgba(59, 130, 246, 0.28); }
    .card { background: linear-gradient(145deg, #172238, #131d2f); border: 1px solid #30405b; border-radius: 15px; padding: 1.35rem; margin: 1.1rem 0 0.8rem; box-shadow: 0 10px 24px rgba(0, 0, 0, 0.16); }
    .card-title { color: #f5f8ff; font-size: 1.12rem; font-weight: 700; margin-bottom: 0.7rem; }
    .card-subtitle { color: #9daec8; font-size: 0.85rem; line-height: 2.25; margin-bottom: 0.9rem; }
    .card-text { color: #d5deed; font-size: 0.95rem; line-height: 1.6; background: #0d1422; padding: 0.9rem 1rem; border-radius: 9px; border-left: 3px solid #5c8fff; }
    .badge { border-radius: 999px; font-weight: 650; margin: 0 0.35rem 0.2rem 0; }
    [data-testid="stTextInput"] input, [data-testid="stSelectbox"] div[data-baseweb="select"] > div, [data-testid="stDateInput"] input { border-radius: 9px !important; border-color: #3a4964 !important; background: #121b2a !important; }
    [data-testid="stExpander"] { border: 1px solid #2d3d56; border-radius: 10px; }
</style>
""",
    unsafe_allow_html=True,
)

# App title and product header
st.markdown(
    """
    <section class="hero">
      <div class="eyebrow">GOVERNMENT PROCUREMENT INTELLIGENCE</div>
      <h1>Lets Bid</h1>
      <p>Search GeM bid documents with semantic relevance and exact keyword matching in one focused workspace.</p>
    </section>
    """,
    unsafe_allow_html=True,
)


# Initialize Hybrid Retriever
@st.cache_resource
def get_retriever(chroma_path, collection_name, lexical_db_path=None):
  return HybridRetriever(
      chroma_path=chroma_path,
      collection_name=collection_name,
      lexical_db_path=lexical_db_path,
  )


# Sidebar configurations
st.sidebar.header("Workspace")

chroma_path_str = st.sidebar.text_input("Chroma Path", str(DEFAULT_CHROMA_PATH))
collection_name = st.sidebar.text_input("Collection Name", DEFAULT_COLLECTION)
lexical_db = st.sidebar.text_input("Lexical SQLite Path (Optional)", "")

# Load retriever
try:
  retriever = get_retriever(
      chroma_path=chroma_path_str,
      collection_name=collection_name,
      lexical_db_path=Path(lexical_db) if lexical_db else None,
  )
except Exception as e:
  st.sidebar.error(f"Error initializing retriever: {e}")
  retriever = None

# Index management section in sidebar
st.sidebar.markdown("---")
st.sidebar.subheader("Index status")

if retriever:
  try:
    state = retriever.lexical.state()
    if state:
      st.sidebar.success("Index is present & current.")
      st.sidebar.write(f"**Collection:** `{state.get('collection')}`")
      st.sidebar.write(f"**Chunks:** `{state.get('chunk_count', 0):,}`")
      st.sidebar.write(
          f"**Last Indexed:** `{state.get('indexed_at', 'Unknown')}`"
      )
    else:
      st.sidebar.warning(
          "Lexical index not built yet (using Chroma native fallback)."
      )
  except Exception as e:
    st.sidebar.error(f"Could not read index state: {e}")

  if st.sidebar.button("Rebuild Lexical Index"):
    with st.sidebar.status(
        "Building lexical index (this might take a while)..."
    ) as status:
      try:
        indexed = retriever.build_lexical_index(rebuild=True)
        status.update(
            label=f"Successfully built index for {indexed:,} chunks!",
            state="complete",
        )
        st.cache_resource.clear()
        st.rerun()
      except Exception as e:
        status.update(label=f"Failed to build index: {e}", state="error")
        st.sidebar.code(traceback.format_exc())

st.sidebar.markdown("---")
st.sidebar.subheader("Retrieval tuning")
rrf_k = st.sidebar.slider(
    "RRF K parameter", min_value=1, max_value=200, value=60
)
dense_weight = st.sidebar.slider(
    "Dense Weight", min_value=0.0, max_value=5.0, value=1.0, step=0.1
)
lexical_weight = st.sidebar.slider(
    "Lexical Weight", min_value=0.0, max_value=5.0, value=1.0, step=0.1
)
exclude_boilerplate = st.sidebar.checkbox(
    "Exclude Boilerplate Documents", value=True
)

# Main search form
with st.container():
  st.markdown('<div class="section-label">DISCOVER OPPORTUNITIES</div>', unsafe_allow_html=True)
  st.subheader("Search bid intelligence")
  st.caption("Use natural language, precise product terms, or a known bid ID. Add filters when you need to narrow the result set.")
  query = st.text_input(
      "Enter your query manually",
      placeholder="e.g., bullet resistant vehicle for police force",
  )

  # Date filters
  st.markdown("##### Refine results")
  col_date1, col_date2 = st.columns(2)
  with col_date1:
    bid_start_date = st.date_input(
        "Published/Start Date (On or After)",
        value=None,
        help="Filters for bids published on or after this date.",
    )
  with col_date2:
    bid_end_date = st.date_input(
        "Bid Closing Date (On or Before)",
        value=None,
        help="Filters for bids that close on or before this date.",
    )

  limit = st.slider("Result Limit", min_value=1, max_value=100, value=10)

  search_button = st.button("Search")

# Initialize session state for search results
if "search_results" not in st.session_state:
  st.session_state.search_results = None

# Executing Search
if search_button:
  if not query.strip():
    st.warning("Please enter a non-empty search query.")
  elif not retriever:
    st.error("Retriever is not initialized. Please verify configuration.")
  else:
    where_filter = None

    with st.spinner("Searching and ranking results..."):
      try:
        # Fetch extra results if date range filtering is active
        fetch_limit = limit * 4 if (bid_start_date or bid_end_date) else limit

        results = retriever.search(
            query,
            limit=fetch_limit,
            where=where_filter,
            exclude_boilerplate=exclude_boilerplate,
            rrf_k=rrf_k,
            dense_weight=dense_weight,
            lexical_weight=lexical_weight,
        )

        # --- PYTHON POST-FILTERING FOR DATES ---
        if results and (bid_start_date or bid_end_date):
          filtered_results = []
          for res in results:
            keep = True
            meta = res.metadata

            # 1. Parse Published/Start Date (e.g. "2026-07-10")
            pub_date_raw = (
                meta.get("published_date")
                or meta.get("bid_start_date")
                or meta.get("date")
            )
            if bid_start_date and pub_date_raw:
              try:
                clean_start = str(pub_date_raw).split("T")[0]
                doc_start = datetime.strptime(
                    clean_start, "%Y-%m-%d"
                ).date()
                if doc_start < bid_start_date:
                  keep = False
              except Exception:
                pass

            # 2. Parse End/Submission Date (e.g. "30-07-2026 21:00:00")
            end_date_raw = meta.get("end_date_to_submit_bid") or meta.get(
                "bid_end_date"
            )
            if bid_end_date and end_date_raw:
              try:
                clean_end = str(end_date_raw).split(" ")[0]
                # Check format: DD-MM-YYYY vs YYYY-MM-DD
                if "-" in clean_end and len(clean_end.split("-")[0]) == 4:
                  doc_end = datetime.strptime(clean_end, "%Y-%m-%d").date()
                else:
                  doc_end = datetime.strptime(clean_end, "%d-%m-%Y").date()

                if doc_end > bid_end_date:
                  keep = False
              except Exception:
                pass

            if keep:
              filtered_results.append(res)

          results = filtered_results[:limit]

        st.session_state.search_results = results
        if not results:
          st.info("No matching results found.")
      except Exception as search_err:
        st.error(f"Search failed: {search_err}")
        st.code(traceback.format_exc())

# Display Results from Session State
if st.session_state.search_results:
  results = st.session_state.search_results
  st.success(f"Showing {len(results)} matching chunks.")

  # --- CSV EXPORT ---
  export_rows = build_export_rows(results)
  csv_bytes = rows_to_csv_bytes(export_rows)
  st.download_button(
      label=f"📊 Export {len(export_rows)} bid(s) to CSV",
      data=csv_bytes,
      file_name=f"gem_bids_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
      mime="text/csv",
      key="export_csv_button",
  )
  st.caption(
      f"{len(results)} chunk(s) collapse to {len(export_rows)} unique bid(s) "
      "in the export (one row per bid, matching your sheet format)."
  )

  # --- LIVE EXTENSION CHECK ---
  st.markdown("##### 🔄 Check for Extensions on GeM (Live)")
  st.caption(
      "Queries the live GeM portal for each bid above and compares its "
      "current end date against what's stored. This makes one network "
      "request per bid (with a short polite delay between each), so it "
      "takes longer for larger result sets -- run it only when you actually "
      "need the up-to-date status, not on every search."
  )

  if st.button(f"Check {len(export_rows)} bid(s) for extensions", key="check_extensions_button"):
    bid_stored_end_dates = {
        row["Bid ID"]: row["End date to submit bid"]
        for row in export_rows
        if row["Bid ID"] and row["Bid ID"] != "N/A"
    }

    progress_bar = st.progress(0, text="Starting live check...")

    def _update_progress(i, total, bid_number):
      progress_bar.progress(
          i / total, text=f"Checking {i}/{total}: {bid_number}"
      )

    with st.spinner("Checking live status on GeM..."):
      try:
        status_results = check_bids_for_extension(
            bid_stored_end_dates, progress_callback=_update_progress
        )
      except Exception as check_err:
        st.error(f"Live check failed: {check_err}")
        st.code(traceback.format_exc())
        status_results = None

    progress_bar.empty()

    if status_results:
      augmented_rows = []
      extended_count = 0
      not_found_count = 0

      for row in export_rows:
        bid_id = row["Bid ID"]
        status = status_results.get(bid_id, {})
        extended = status.get("extended")
        live_status = status.get("status", "unknown")

        if extended is True:
          extended_label = "YES"
          extended_count += 1
        elif live_status == "not_found":
          # Reliable, not ambiguous: an extended bid's new end date is
          # necessarily in the future, so it would still appear in
          # ongoing_bids. Not found here means it closed without being
          # extended -- the one caveat is GeM's own ~15 min indexing lag
          # for very recent changes, per the portal's own banner.
          extended_label = "Closed (not extended)"
        elif extended is False:
          extended_label = "No"
        else:
          extended_label = "Unknown (could not parse dates)"

        if live_status == "not_found":
          not_found_count += 1

        augmented_row = dict(row)
        augmented_row["Live End Date (GeM)"] = status.get("live_end_date", "")
        augmented_row["Extended?"] = extended_label
        augmented_row["Live Status"] = live_status
        augmented_rows.append(augmented_row)

      st.success(
          f"Checked {len(augmented_rows)} bid(s): {extended_count} extended, "
          f"{not_found_count} not found live (likely already closed)."
      )

      augmented_csv_bytes = rows_to_csv_bytes(
          augmented_rows, fieldnames=CSV_COLUMNS_WITH_STATUS
      )
      st.download_button(
          label=f"📥 Download extension-checked CSV ({extended_count} extended)",
          data=augmented_csv_bytes,
          file_name=f"gem_bids_extension_check_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
          mime="text/csv",
          key="export_csv_with_status_button",
      )

      if extended_count > 0:
        st.markdown("**Bids with changed end dates:**")
        for row in augmented_rows:
          if row["Extended?"] == "YES":
            st.markdown(
                f"- **{row['Bid ID']}** — was `{row['End date to submit bid']}`, "
                f"now `{row['Live End Date (GeM)']}`"
            )

  for idx, res in enumerate(results, 1):
    score = res.score
    dense_rank = res.dense_rank
    lexical_rank = res.lexical_rank

    if dense_rank is not None and dense_rank <= 3:
      conf_class = "badge-high"
      conf_lbl = "High Match (Semantic)"
    elif lexical_rank is not None and lexical_rank <= 3:
      conf_class = "badge-high"
      conf_lbl = "High Match (Lexical)"
    elif dense_rank is not None or lexical_rank is not None:
      conf_class = "badge-med"
      conf_lbl = "Medium Match"
    else:
      conf_class = "badge-low"
      conf_lbl = "Low Match"

    bid_title = res.metadata.get("bid_title", "N/A")
    bid_id = res.metadata.get("bid_id", "N/A")
    chunk_type = res.metadata.get("chunk_type", "N/A")
    ministry = res.metadata.get("ministry", "N/A")

    # Read from actual metadata schema fields
    start_date_str = (
        res.metadata.get("published_date")
        or res.metadata.get("bid_start_date")
        or "N/A"
    )
    end_date_str = (
        res.metadata.get("end_date_to_submit_bid")
        or res.metadata.get("bid_end_date")
        or "N/A"
    )
    badges = [
        f'<span class="badge {conf_class}">{escape(conf_lbl)}</span>',
        '<span class="badge badge-new">NEW</span>' if is_new_bid(res.metadata) else "",
        f'<span class="badge badge-info">Bid ID: {escape(str(bid_id))}</span>',
        f'<span class="badge badge-none">Type: {escape(str(chunk_type))}</span>',
        f'<span class="badge badge-none">Ministry: {escape(str(ministry))}</span>',
        f'<span class="badge badge-none">Published: {escape(str(start_date_str))}</span>',
        f'<span class="badge badge-none">End: {escape(str(end_date_str))}</span>',
        f'<span class="badge badge-none">Fused Score: {score:.5f}</span>',
        f'<span class="badge badge-none">Dense Rank: {dense_rank or "N/A"}</span>',
        f'<span class="badge badge-none">Lexical Rank: {lexical_rank or "N/A"}</span>',
    ]
    card_html = (
        f'<div class="card"><div class="card-title">#{idx} | {escape(str(bid_title))}</div>'
        f'<div class="card-subtitle">{"".join(badges)}</div>'
        f'<div class="card-text">{escape(str(res.text))}</div></div>'
    )
    st.markdown(card_html, unsafe_allow_html=True)

    col_meta, col_dl = st.columns([4, 1])
    with col_meta:
      with st.expander(f"Inspect metadata for chunk: {res.chunk_id}"):
        st.json(res.metadata)
    with col_dl:
      pdf_file = find_pdf_file(res.metadata)
      if pdf_file:
        try:
          static_dir = Path("static")
          static_dir.mkdir(exist_ok=True)
          target_static_path = static_dir / pdf_file.name
          if not target_static_path.exists():
            import shutil

            shutil.copy2(pdf_file, target_static_path)

          with open(pdf_file, "rb") as f:
            pdf_bytes = f.read()

          st.download_button(
              label="📥 Download PDF",
              data=pdf_bytes,
              file_name=pdf_file.name,
              mime="application/pdf",
              key=f"dl_{res.chunk_id}_{idx}",
              use_container_width=True,
          )

          st.markdown(
              f'<a href="/static/{pdf_file.name}" target="_blank"'
              ' style="text-decoration: none;"><button style="background-color:'
              " #1e293b; color: #3b82f6; border: 1px solid #3b82f6; "
              "border-radius: 8px; padding: 6px 12px; width: 100%; cursor:"
              ' pointer; margin-top: 4px; font-weight: 500;">📄 Open PDF in'
              " Tab</button></a>",
              unsafe_allow_html=True,
          )
        except Exception as e:
          st.error("Error loading PDF")
      else:
        st.caption("PDF not found locally")
