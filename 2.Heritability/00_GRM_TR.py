"""
Whole-genome GRM construction using pure Python matrix operations.

Tested models:
    - rLAV
    - LAV
    - rLAV+nLAV

This script generates and persistently saves GRM files. The variant category
can be switched by modifying TR_TYPE in the configuration section.
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
# Options: "VNTR" or "nonTR"
TR_TYPE = "nonTR"

MAIN_DIR = "25_hsq_real_data"

# The working directory is automatically named according to TR_TYPE,
# for example raw_VNTR or raw_nonTR.
WORK_DIR = f"{MAIN_DIR}/01_WG_hsq_comparison_ManualGRM_raw_{TR_TYPE}"
GRM_OUT_DIR = f"{WORK_DIR}/GRMs"
SHM_BASE = f"/dev/shm/s30_wg_grm_{TR_TYPE}_{uuid.uuid4().hex[:8]}"

# QC-filtered PLINK binary files. The suffix is automatically determined by TR_TYPE.
QC_BFILE_DIR = "/home/s3020226030/1_rSV/01_human_chm13/25_hsq_real_data/bfile"
BASE_MODELS = {
    "rLAV": f"{QC_BFILE_DIR}/rLAV_{TR_TYPE}",
    "nLAV": f"{QC_BFILE_DIR}/nLAV_{TR_TYPE}",
    "LAV": f"{QC_BFILE_DIR}/LAV_{TR_TYPE}",
}

# Target models to be processed
TARGET_MODELS = ["rLAV", "LAV", "rLAV_nLAV"]


# ==========================================
# 2. Helper Functions and Manual GRM Construction
# ==========================================
def run_cmd(cmd, check=True):
    try:
        subprocess.run(
            cmd,
            shell=True,
            check=check,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        return True
    except subprocess.CalledProcessError:
        return False


def recode_to_raw(bfile_prefix, out_prefix):
    """Call PLINK to generate an additive dosage .raw file."""
    run_cmd(
        f"plink --bfile {bfile_prefix} "
        f"--recode A "
        f"--out {out_prefix} "
        f"--allow-extra-chr ",
        check=False
    )

    raw_file = out_prefix + ".raw"

    return raw_file if os.path.exists(raw_file) else None


def load_raw_genotypes(raw_file):
    """Load a PLINK .raw file into a NumPy array with mean imputation."""
    df = pd.read_csv(raw_file, sep=r"\s+")
    iids = df["IID"].astype(str).tolist()

    raw_cols = list(df.columns[6:])
    snp_ids = [c.rsplit("_", 1)[0] for c in raw_cols]

    X = df[raw_cols].values.astype(np.float64)

    col_means = np.nanmean(X, axis=0)
    nan_mask = np.isnan(X)
    X[nan_mask] = np.take(col_means, np.where(nan_mask)[1])

    return iids, snp_ids, X


def standardize_and_cov(X):
    """
    Standardize the genotype matrix and return the GRM, effective variant count,
    and standardized matrix.

    The GRM is computed as:

        G = Z @ Z.T / P

    where Z is the standardized genotype matrix and P is the number of
    non-constant variants.
    """
    stds = X.std(axis=0, ddof=1)
    nz = stds > 0

    Z = np.zeros_like(X)
    Z[:, nz] = (X[:, nz] - X[:, nz].mean(axis=0)) / stds[nz]

    P = int(nz.sum())

    if P == 0:
        return np.zeros((X.shape[0], X.shape[0])), 0, Z

    G = (Z @ Z.T) / P

    return G, P, Z


def save_grm(G, P, sample_ids, out_prefix):
    """Save a GRM matrix in GCTA binary format."""
    n = len(sample_ids)
    rows, cols = np.tril_indices(n)

    with open(out_prefix + ".grm.id", "w") as f:
        for s in sample_ids:
            f.write(f"{s}\t{s}\n")

    G[rows, cols].astype(np.float32).tofile(out_prefix + ".grm.bin")
    np.full(len(rows), float(P), dtype=np.float32).tofile(out_prefix + ".grm.N.bin")

    return out_prefix


def build_grm_pipeline(bfile_prefix, out_prefix):
    """Run the complete pure-Python GRM construction workflow."""
    print(f"    [GRM] Processing {os.path.basename(bfile_prefix)}...", flush=True)

    raw_file = recode_to_raw(bfile_prefix, out_prefix)

    if not raw_file:
        print(f"[Error] Failed to generate .raw file for {bfile_prefix}")
        return False

    iids, snp_ids, X = load_raw_genotypes(raw_file)
    G, P, Z = standardize_and_cov(X)
    save_grm(G, P, iids, out_prefix)

    os.remove(raw_file)
    del X, Z, G

    return True


# ==========================================
# 3. Initialization and PLINK File Preparation
# ==========================================
os.makedirs(WORK_DIR, exist_ok=True)
os.makedirs(GRM_OUT_DIR, exist_ok=True)
os.makedirs(SHM_BASE, exist_ok=True)

print(f"=== Starting GRM construction for variant type: {TR_TYPE} ===", flush=True)
print(f"[Init] RAM disk: {SHM_BASE}", flush=True)
print(f"[Init] GRM output directory: {GRM_OUT_DIR}", flush=True)

bfile_cache_dir = f"{SHM_BASE}/bfile_wg"
os.makedirs(bfile_cache_dir, exist_ok=True)

print("[Prep] Step 1: Merging required PLINK file combinations...", flush=True)

merged_rn = f"{bfile_cache_dir}/rLAV_nLAV"
run_cmd(
    f"plink --keep-allele-order "
    f"--bfile {BASE_MODELS['rLAV']} "
    f"--bmerge {BASE_MODELS['nLAV']} "
    f"--make-bed "
    f"--out {merged_rn} "
    f"--allow-extra-chr "
    f"--allow-no-sex"
)

FINAL_BFILES = {
    "rLAV": BASE_MODELS["rLAV"],
    "LAV": BASE_MODELS["LAV"],
    "rLAV_nLAV": merged_rn
}


# ==========================================
# 4. Generate and Save GRMs
# ==========================================
print("\n[Prep] Step 2: Manually calculating GRMs from genotype matrices...", flush=True)

for model in TARGET_MODELS:
    grm_out = f"{GRM_OUT_DIR}/grm_manual_{model}"

    if build_grm_pipeline(FINAL_BFILES[model], grm_out):
        print(f"[Success] Successfully created GRM for {model} ({TR_TYPE}) -> {grm_out}.grm.bin")
    else:
        print(f"[Error] Failed to construct manual GRM for {model}. Exiting.")
        shutil.rmtree(SHM_BASE, ignore_errors=True)
        exit(1)


# ==========================================
# 5. Cleanup
# ==========================================
print("\n[Cleanup] Removing temporary RAM disk directory...", flush=True)
shutil.rmtree(SHM_BASE, ignore_errors=True)

print(f"\n[Done] All requested GRMs for {TR_TYPE} have been generated and saved to: {GRM_OUT_DIR}")
print("Done!", flush=True)