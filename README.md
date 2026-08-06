# Lets Bid — Phase 2 local automation

This is the laptop-ready folder for the stage after Colab has completed the
initial embedding job. It runs the full Lets Bid search app and automatically
maintains only new or changed GeM bids.

## Before enabling automation

Copy these Phase 1 outputs into this folder:

```text
bid_chunks.json
downloads/                  (including bids/ and downloaded_bid_manifest.json)
chroma_db/                  (returned by the Colab embedding job)
```

Keeping the PDFs, chunk file, manifest, and Chroma database together prevents
the maintenance job from treating the initial corpus as missing or new.

## Set up the laptop

Run PowerShell in this folder:

```powershell
.\setup_new_laptop.ps1
.\activate_phase_2.ps1
.\start_portal.ps1
```

`activate_phase_2.ps1` creates the daily task for 11:00 AM by default. Choose
a different time if needed:

```powershell
.\activate_phase_2.ps1 -Time 06:00
```

## What the scheduled task does

```text
new GeM PDFs -> new/changed chunks -> CPU embeddings for changed chunks only
                                      -> weekly removal of expired bids
```

It uses the existing initial `chroma_db/` from Colab and does not rebuild the
whole initial corpus.
