"""
Build joint GRMs by horizontally concatenating copy-number features from TSV
files with rLAV+nLAV dosage features from PLINK files.

Target models:
    - CN_rLAV_nLAV_STR
    - CN_rLAV_nLAV_VNTR
"""

import pandas as pd
import numpy as np
import os
import subprocess
import shutil
import uuid
import time
import threading
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
MAIN_DIR = "/home/s3020226030/1_rSV/01_human_chm13/25_hsq_real_data"
WORK_DIR = f"{MAIN_DIR}/06_WG_hsq_comparison_CN_rLAV_nLAV"
GRM_OUT_DIR = f"{WORK_DIR}/GRMs"
SHM_BASE = f"/dev/shm/s30_joint_{uuid.uuid4().hex[:8]}"

PHENO_FILE = "/home/s3020226030/1_rSV/01_human_chm13/16_cis_eQTL/phenotype.parquet"
QC_BFILE_DIR = "/home/s3020226030/1_rSV/01_human_chm13/25_hsq_real_data/bfile"
TR_DIR = "/home/s3020226030/1_rSV/01_human_chm13/06_dosage"

TR_TYPES = ["STR", "VNTR"]

os.makedirs(GRM_OUT_DIR, exist_ok=True)
os.makedirs(SHM_BASE, exist_ok=True)


# ==========================================
# 2. Helper Functions
# ==========================================
def run_cmd(cmd):
    try:
        subprocess.run(
            cmd,
            shell=True,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        return True
    except subprocess.CalledProcessError:
        return False


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


# ==========================================
# 3. Main Workflow: Build Joint Feature Matrices
# ==========================================
print("[Init] Reading valid phenotype samples...", flush=True)
pheno_df = pd.read_parquet(PHENO_FILE)
valid_phenos = pheno_df.columns.astype(str).tolist()

for tr_type in TR_TYPES:
    print(f"\n{'=' * 40}")
    print(f"Processing: {tr_type}")
    print(f"{'=' * 40}", flush=True)

    # Input and output path configuration
    bfile_rLAV = f"{QC_BFILE_DIR}/rLAV_{tr_type}"
    bfile_nLAV = f"{QC_BFILE_DIR}/nLAV_{tr_type}"
    cn_tsv = f"{TR_DIR}/LAV_{tr_type}.copy_number_exact.tsv"

    merged_bfile = f"{SHM_BASE}/rLAV_nLAV_{tr_type}"
    raw_file = f"{SHM_BASE}/rLAV_nLAV_{tr_type}.raw"
    grm_out = f"{GRM_OUT_DIR}/grm_manual_CN_rLAV_nLAV_{tr_type}"

    # 1. Merge rLAV and nLAV PLINK files and recode them into additive dosage format
    print(f"  [1/4] Merging rLAV and nLAV for {tr_type}...", flush=True)
    run_cmd(
        f"plink --keep-allele-order "
        f"--bfile {bfile_rLAV} "
        f"--bmerge {bfile_nLAV} "
        f"--make-bed "
        f"--out {merged_bfile} "
        f"--allow-extra-chr "
        f"--allow-no-sex"
    )

    print("  [2/4] Recoding merged PLINK file to .raw format...", flush=True)
    run_cmd(
        f"plink --bfile {merged_bfile} "
        f"--recode A "
        f"--out {merged_bfile} "
        f"--allow-extra-chr"
    )

    # 2. Load rLAV+nLAV dosage matrix
    print("  [3/4] Loading and aligning feature matrices...", flush=True)
    df_raw = pd.read_csv(raw_file, sep=r"\s+")
    df_raw.index = df_raw["IID"].astype(str)
    raw_feature_cols = list(df_raw.columns[6:])

    # 3. Load copy-number TSV matrix
    df_tsv = pd.read_csv(cn_tsv, sep="\t", na_values=["NA", "", "."])
    meta_cols = ["#CHROM", "POS", "ID", "REPEAT_UNIT", "REPEAT_LENGTH"]
    sample_cols = [c for c in df_tsv.columns if c not in meta_cols]

    df_tsv = df_tsv[sample_cols].T
    df_tsv.index = df_tsv.index.astype(str)

    # 4. Strictly align samples across PLINK dosage, copy-number TSV, and phenotype data
    common_samples = df_raw.index.intersection(df_tsv.index).intersection(valid_phenos).tolist()
    print(f"        -> Aligned common samples: {len(common_samples)}")

    df_raw_aligned = df_raw.loc[common_samples, raw_feature_cols]
    df_tsv_aligned = df_tsv.loc[common_samples]

    # Remove copy-number features with entirely missing values
    df_tsv_aligned = df_tsv_aligned.dropna(axis=1, how="all")

    # Extract rLAV+nLAV dosage matrix and perform mean imputation
    X_raw = df_raw_aligned.values.astype(np.float64)
    raw_means = np.nanmean(X_raw, axis=0)
    raw_nan_mask = np.isnan(X_raw)
    X_raw[raw_nan_mask] = np.take(raw_means, np.where(raw_nan_mask)[1])

    # Extract copy-number matrix and perform mean imputation
    X_tsv = df_tsv_aligned.values.astype(np.float64)
    tsv_means = np.nanmean(X_tsv, axis=0)
    tsv_means[np.isnan(tsv_means)] = 0
    tsv_nan_mask = np.isnan(X_tsv)
    X_tsv[tsv_nan_mask] = np.take(tsv_means, np.where(tsv_nan_mask)[1])

    # 5. Horizontally concatenate copy-number and rLAV+nLAV feature matrices
    X_joint = np.hstack([X_tsv, X_raw])
    print(f"        -> Joint matrix shape: {X_joint.shape} samples x variants")

    # 6. Calculate and save the joint GRM
    print("  [4/4] Calculating and saving joint GRM...", flush=True)
    G, P = standardize_and_cov(X_joint)
    save_grm(G, P, common_samples, grm_out)

    print(f"[Success] Created GRM: {grm_out}.grm.bin")
    print(f"          -> Effective variants: {P}")

    # Release memory and remove temporary files
    del df_raw, df_tsv, df_raw_aligned, df_tsv_aligned, X_raw, X_tsv, X_joint, G
    os.remove(raw_file)


print("\n[Cleanup] Removing temporary RAM disk directory...", flush=True)
shutil.rmtree(SHM_BASE, ignore_errors=True)

print("[Done] All joint GRMs were successfully generated.")