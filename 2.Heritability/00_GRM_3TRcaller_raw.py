"""
Process classical TR TSV datasets: STRling, adVNTR, and VNTRseek.

This script parses raw TSV genotype matrices, performs manual imputation and
standardization, constructs genetic relationship matrices (GRMs) directly from
the processed matrices, and saves the resulting GRM files for downstream
heritability estimation.
"""

import pandas as pd
import numpy as np
import os
import shutil
import uuid
import time
import threading
import sys
import psutil


# ==========================================
# 0. Resource Monitor
# ==========================================
def resource_monitor(interval=60):
    print("[Monitor] Hardware resource tracking started...", flush=True)
    io_start = psutil.disk_io_counters()

    while True:
        time.sleep(interval)

        cpu_percent = psutil.cpu_percent(interval=None)
        mem = psutil.virtual_memory()
        io_now = psutil.disk_io_counters()

        if io_now and io_start:
            read_mbs = (io_now.read_bytes - io_start.read_bytes) / interval / (1024**2)
            write_mbs = (io_now.write_bytes - io_start.write_bytes) / interval / (1024**2)
            io_start = io_now
        else:
            read_mbs = write_mbs = 0.0

        print(
            f"[Stats] CPU: {cpu_percent:5.1f}% | RAM: {mem.percent:4.1f}% | "
            f"Disk R/W: {read_mbs:5.1f}/{write_mbs:5.1f} MB/s",
            flush=True
        )


threading.Thread(target=resource_monitor, args=(60,), daemon=True).start()


# ==========================================
# 1. Configuration
# ==========================================
MAIN_DIR = "25_hsq_real_data"
WORK_DIR = f"{MAIN_DIR}/03_WG_hsq_comparison_ManualGRM_TRs_raw"
GRM_OUT_DIR = f"{WORK_DIR}/GRMs"  # Directory for persistent GRM outputs
SHM_BASE = f"/dev/shm/s30_trs_grm_{uuid.uuid4().hex[:8]}"

# Phenotype file used to extract the set of valid shared sample IDs
PHENO_FILE = "16_cis_eQTL/phenotype.parquet"

TR_DIR = "/home/s3020226030/1_rSV/01_human_chm13/17_TR_beta"
BASE_MODELS = {
    "STR": f"{TR_DIR}/731samples.STR.raw.tsv",
    "adVNTR": f"{TR_DIR}/731samples.adVNTR.raw.tsv",
    "VNTRseek": f"{TR_DIR}/731samples.VNTRseek.raw.tsv"
}

TARGET_MODELS = ["STR", "adVNTR", "VNTRseek"]


# ==========================================
# 2. Data Loading and Manual GRM Construction
# ==========================================
def load_and_impute_tsv(filepath, valid_samples):
    df = pd.read_csv(filepath, sep="\t", index_col=0, na_values=["NA", ""])
    df.index = df.index.astype(str)

    common_samples = df.index.intersection(valid_samples)
    if len(common_samples) == 0:
        raise ValueError(
            "Sample alignment failed.\n"
            f"First five TSV samples: {df.index[:5].tolist()}\n"
            f"First five phenotype samples: {valid_samples[:5]}"
        )

    df = df.loc[common_samples]

    original_cols = df.shape[1]
    df.dropna(axis=1, how="all", inplace=True)
    dropped_cols = original_cols - df.shape[1]

    if dropped_cols > 0:
        print(f"        [Clean] Dropped {dropped_cols} columns that were entirely NA.")

    iids = df.index.tolist()
    X = df.values.astype(np.float64)

    col_means = np.nanmean(X, axis=0)
    col_means[np.isnan(col_means)] = 0

    nan_mask = np.isnan(X)
    X[nan_mask] = np.take(col_means, np.where(nan_mask)[1])

    return iids, X


def standardize_and_cov(X):
    stds = X.std(axis=0, ddof=1)
    nz = stds > 0

    Z = np.zeros_like(X)
    Z[:, nz] = (X[:, nz] - X[:, nz].mean(axis=0)) / stds[nz]

    P = int(nz.sum())

    if P == 0:
        return np.zeros((X.shape[0], X.shape[0])), 0

    G = (Z @ Z.T) / P

    return G, P


def save_grm(G, P, sample_ids, out_prefix):
    n = len(sample_ids)
    rows, cols = np.tril_indices(n)

    with open(out_prefix + ".grm.id", "w") as f:
        for s in sample_ids:
            f.write(f"{s}\t{s}\n")

    G[rows, cols].astype(np.float32).tofile(out_prefix + ".grm.bin")
    np.full(len(rows), float(P), dtype=np.float32).tofile(out_prefix + ".grm.N.bin")

    return out_prefix


# ==========================================
# 3. Initialization and Valid Sample Extraction
# ==========================================
os.makedirs(WORK_DIR, exist_ok=True)
os.makedirs(GRM_OUT_DIR, exist_ok=True)
os.makedirs(SHM_BASE, exist_ok=True)

print("=== Starting Classical TR GRM Construction ===", flush=True)
print(f"[Init] RAM disk: {SHM_BASE}", flush=True)
print(f"[Init] GRM output directory: {GRM_OUT_DIR}", flush=True)

print("[Init] Reading phenotype file to extract valid sample IDs...", flush=True)
pheno_df = pd.read_parquet(PHENO_FILE)
valid_pheno_samples = pheno_df.columns.astype(str).tolist()
print(f"[Init] Found {len(valid_pheno_samples)} valid sample IDs from phenotype.\n", flush=True)


# ==========================================
# 4. Generate and Save GRMs
# ==========================================
print("[Prep] Manually calculating GRMs directly from TSV genotype matrices...", flush=True)

for model in TARGET_MODELS:
    # Save GRM outputs directly to the persistent disk directory
    grm_out = f"{GRM_OUT_DIR}/grm_manual_{model}"

    print(f"    [GRM] Building GRM for {model}...", flush=True)
    target_file = BASE_MODELS[model]

    try:
        iids, X = load_and_impute_tsv(target_file, valid_pheno_samples)
        G, P = standardize_and_cov(X)
        save_grm(G, P, iids, grm_out)

        del X, G

        print(f"[Success] Successfully created GRM for {model} -> {grm_out}.grm.bin")
        print(f"          -> Effective variants: {P}, aligned samples: {len(iids)}\n", flush=True)

    except Exception as e:
        print(f"[Error] Exception while processing {model}: {str(e)}")
        shutil.rmtree(SHM_BASE, ignore_errors=True)
        sys.exit(1)


# ==========================================
# 5. Cleanup
# ==========================================
print("\n[Cleanup] Removing temporary RAM disk directory...", flush=True)
shutil.rmtree(SHM_BASE, ignore_errors=True)

print(f"\n[Done] GRMs for all classical TR datasets have been generated and saved to: {GRM_OUT_DIR}")
print("Done!", flush=True)