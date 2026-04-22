"""00_setup_and_download.py — KsaaiP2 Pipeline v4_updated. Fix B: config.yaml wired."""

import os
import sys
import hashlib
import logging
import urllib.request
import yaml

with open("config.yaml") as _f:
    _CFG = yaml.safe_load(_f)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

TCGA_S3_URL       = _CFG["tcga"]["s3_url"]
TCGA_RAW_PATH     = _CFG["tcga"]["raw_path"]
SC_URL            = _CFG["superconductivity"]["url"]
SC_RAW_PATH       = _CFG["superconductivity"]["raw_path"]
FRED_MD_URL       = _CFG["fred_md"]["url"]
FRED_MD_RAW_PATH  = _CFG["fred_md"]["raw_path"]
FRED_MD_TARGET    = _CFG["fred_md"]["target_series"]
FRED_MD_TRANSFORM = _CFG["fred_md"]["target_transform_code"]
CHECKSUM_FILE     = _CFG.get("checksums_json", "data/raw/.checksums.json")
HK_GENE_LIST_PATH = _CFG.get("tcga", {}).get("hk_gene_list", "data/processed/tcga_hk_genes.txt")

HK_GENES = ["ACTB", "GAPDH", "B2M", "HPRT1", "SDHA"]

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()

def download_if_missing(url, dest, desc=""):
    if os.path.exists(dest):
        log.info(f"[SKIP] {desc or dest} already present")
        return
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    log.info(f"[DOWNLOAD] {desc or url} -> {dest}")
    urllib.request.urlretrieve(url, dest)
    log.info(f"[OK] Downloaded {desc}")

def write_hk_gene_list():
    os.makedirs(os.path.dirname(HK_GENE_LIST_PATH), exist_ok=True)
    with open(HK_GENE_LIST_PATH, "w") as f:
        for gene in HK_GENES:
            f.write(gene + "\n")
    log.info(f"[OK] Wrote {len(HK_GENES)} HK genes to {HK_GENE_LIST_PATH}")

def record_checksums(paths):
    import json
    checksums = {}
    for p in paths:
        if os.path.exists(p):
            checksums[p] = sha256_file(p)
    os.makedirs(os.path.dirname(CHECKSUM_FILE), exist_ok=True)
    with open(CHECKSUM_FILE, "w") as f:
        json.dump(checksums, f, indent=2)
    log.info(f"[OK] Checksums written to {CHECKSUM_FILE}")

def main():
    log.info("Step 00: Setup and download (config.yaml wired — Fix B)")
    log.info(f"  TCGA_S3_URL    = {TCGA_S3_URL}")
    log.info(f"  SC_URL         = {SC_URL}")
    log.info(f"  FRED_MD_URL    = {FRED_MD_URL}")
    log.info(f"  FRED_MD_TARGET = {FRED_MD_TARGET} (must be INDPRO)")

    if FRED_MD_TARGET.strip().upper() != "INDPRO":
        log.error(f"config.yaml fred_md.target_series = '{FRED_MD_TARGET}' — must be 'INDPRO'")
        sys.exit(1)

    try:
        download_if_missing(TCGA_S3_URL, TCGA_RAW_PATH, "TCGA pan-cancer RNA-seq")
    except Exception as e:
        log.warning(f"TCGA download skipped (manual download required): {e}")
        log.warning("Pipeline will use synthetic TCGA proxy for Steps 1-4.")
    download_if_missing(SC_URL, SC_RAW_PATH, "UCI SuperConductivity")
    download_if_missing(FRED_MD_URL, FRED_MD_RAW_PATH, "FRED-MD macroeconomic")

    write_hk_gene_list()
    record_checksums([TCGA_RAW_PATH, SC_RAW_PATH, FRED_MD_RAW_PATH])

    log.info("Step 00 complete.")

if __name__ == "__main__":
    main()