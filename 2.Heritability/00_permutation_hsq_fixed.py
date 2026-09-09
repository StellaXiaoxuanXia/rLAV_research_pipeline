"""
Single-node full-scale cluster version for synchronized permutation REML analysis.
The script processes all synchronized permutation replicates in an internal loop,
with I/O retry handling and checkpoint-aware resume support.
"""

import pandas as pd
import os
import subprocess
import shutil
import uuid
import sys
import threading
import time
import psutil
from concurrent.futures import ThreadPoolExecutor, as_completed


# ==========================================
# 0. Resource Monitor
# ==========================================
def resource_monitor(interval=300):
    """Print resource usage every five minutes in cluster logs."""
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


threading.Thread(target=resource_monitor, args=(300,), daemon=True).start()


# ==========================================
# 1. Global Configuration
# ==========================================
MAIN_DIR = "/home/s3020226030/1_rSV/01_human_chm13/25_hsq_real_data"
WORK_DIR = f"{MAIN_DIR}/05_WG_hsq_permutation/results"

WORKER_THREADS = 40
GCTA_THREADS = 1
TOTAL_REPS = 30

os.makedirs(WORK_DIR, exist_ok=True)


# ==========================================
# 2. GRM Registry
# ==========================================
GRM_CONFIGS = {
    "STR_rLAV": f"{MAIN_DIR}/01_WG_hsq_comparison_ManualGRM_raw_STR/GRMs/grm_manual_rLAV",
    "STR_LAV": f"{MAIN_DIR}/01_WG_hsq_comparison_ManualGRM_raw_STR/GRMs/grm_manual_LAV",
    "STR_rLAV_nLAV": f"{MAIN_DIR}/01_WG_hsq_comparison_ManualGRM_raw_STR/GRMs/grm_manual_rLAV_nLAV",
    "VNTR_rLAV": f"{MAIN_DIR}/01_WG_hsq_comparison_ManualGRM_raw_VNTR/GRMs/grm_manual_rLAV",
    "VNTR_LAV": f"{MAIN_DIR}/01_WG_hsq_comparison_ManualGRM_raw_VNTR/GRMs/grm_manual_LAV",
    "VNTR_rLAV_nLAV": f"{MAIN_DIR}/01_WG_hsq_comparison_ManualGRM_raw_VNTR/GRMs/grm_manual_rLAV_nLAV",
    "nonTR_rLAV": f"{MAIN_DIR}/01_WG_hsq_comparison_ManualGRM_raw_nonTR/GRMs/grm_manual_rLAV",
    "nonTR_LAV": f"{MAIN_DIR}/01_WG_hsq_comparison_ManualGRM_raw_nonTR/GRMs/grm_manual_LAV",
    "nonTR_rLAV_nLAV": f"{MAIN_DIR}/01_WG_hsq_comparison_ManualGRM_raw_nonTR/GRMs/grm_manual_rLAV_nLAV",
    "TRs_STR": f"{MAIN_DIR}/03_WG_hsq_comparison_ManualGRM_TRs/GRMs/grm_manual_STR",
    "TRs_adVNTR": f"{MAIN_DIR}/03_WG_hsq_comparison_ManualGRM_TRs/GRMs/grm_manual_adVNTR",
    "TRs_VNTRseek": f"{MAIN_DIR}/03_WG_hsq_comparison_ManualGRM_TRs/GRMs/grm_manual_VNTRseek",
    "CN_LAV_STR": f"{MAIN_DIR}/04_WG_hsq_comparison_ManualGRM_LAV_CN/GRMs/grm_manual_LAV_STR",
    "CN_LAV_VNTR": f"{MAIN_DIR}/04_WG_hsq_comparison_ManualGRM_LAV_CN/GRMs/grm_manual_LAV_VNTR",
}

print("[Init] Validating 14 GRMs...", flush=True)
for model_name, grm_path in GRM_CONFIGS.items():
    if not os.path.exists(grm_path + ".grm.bin"):
        print(f"[Error] Missing GRM file for {model_name} -> {grm_path}.grm.bin")
        sys.exit(1)
print("[Init] All GRMs passed validation.\n", flush=True)


# ==========================================
# 3. Helper Functions
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


def parse_gcta_reml(hsq_file):
    vg, se, pv = "NA", "NA", "NA"

    if not os.path.exists(hsq_file):
        return vg, se, pv

    with open(hsq_file) as fh:
        for line in fh:
            if line.startswith("V(G)/Vp"):
                parts = line.split()
                vg, se = parts[1], parts[2]
            elif line.startswith("Pval"):
                pv = line.split()[1]

    return vg, se, pv


def run_gcta_reml(grm, pheno, out, covar, qcovar):
    ok = run_cmd(
        f"gcta64 --reml "
        f"--grm {grm} "
        f"--pheno {pheno} "
        f"--covar {covar} "
        f"--qcovar {qcovar} "
        f"--out {out} "
        f"--thread-num {GCTA_THREADS}"
    )

    return parse_gcta_reml(f"{out}.hsq") if ok else ("NA", "NA", "NA")


# ==========================================
# 4. Main Master Loop
# ==========================================
global_time_start = time.time()
write_lock = threading.Lock()

for rep_id in range(1, TOTAL_REPS + 1):
    rep_time_start = time.time()

    print("=" * 60)
    print(f"Starting synchronized permutation replicate {rep_id:02d} / {TOTAL_REPS}")
    print("=" * 60, flush=True)

    PHENO_FILE = f"{MAIN_DIR}/05_WG_hsq_permutation/synced_inputs/pheno_perm_{rep_id:02d}.parquet"
    COV_PERM_FILE = f"{MAIN_DIR}/05_WG_hsq_permutation/synced_inputs/covar_perm_{rep_id:02d}.parquet"
    SUMMARY_FILE = f"{WORK_DIR}/summary_Synced_Permutation_rep{rep_id:02d}.tsv"
    SHM_BASE = f"/dev/shm/s30_master_{rep_id:02d}_{uuid.uuid4().hex[:8]}"

    if not os.path.exists(PHENO_FILE) or not os.path.exists(COV_PERM_FILE):
        print(f"[Warning] Missing shuffled input files for replicate {rep_id:02d}; skipping this replicate.")
        continue

    # Check checkpoint file and resume unfinished phenotypes if possible.
    pheno_df = pd.read_parquet(PHENO_FILE)
    all_phenos = pheno_df.index.tolist()
    pending_phenos = all_phenos.copy()

    if os.path.exists(SUMMARY_FILE):
        try:
            # Read phenotype records that have already been completed.
            existing_df = pd.read_csv(SUMMARY_FILE, sep="\t", usecols=["Phenotype"])
            completed_phenos = set(existing_df["Phenotype"].unique())
            pending_phenos = [p for p in all_phenos if p not in completed_phenos]

            print(
                f"  -> Checkpoint detected: {len(completed_phenos)} phenotypes completed, "
                f"{len(pending_phenos)} phenotypes remaining.",
                flush=True
            )

            if len(pending_phenos) == 0:
                print(f"[Done] Replicate {rep_id:02d} has already been completed; skipping.\n", flush=True)
                continue

        except Exception as e:
            print(
                f"  -> [Warning] Failed to read checkpoint file ({e}); "
                f"restarting this replicate from the beginning.",
                flush=True
            )
            with open(SUMMARY_FILE, "w") as f:
                f.write("Phenotype\tModel\tVG_Vp\tSE\tPval\n")
    else:
        # First run for this replicate: initialize the summary file header.
        os.makedirs(os.path.dirname(SUMMARY_FILE), exist_ok=True)
        with open(SUMMARY_FILE, "w") as f:
            f.write("Phenotype\tModel\tVG_Vp\tSE\tPval\n")

    # Initialize the temporary RAM-disk workspace.
    os.makedirs(SHM_BASE, exist_ok=True)
    gcta_input_dir = f"{SHM_BASE}/gcta_inputs"
    os.makedirs(gcta_input_dir, exist_ok=True)

    COVAR = f"{gcta_input_dir}/discrete_covar.txt"
    QCOVAR = f"{gcta_input_dir}/quantitative_qcovar.txt"

    # Load permuted covariates for the current replicate.
    cov_df = pd.read_parquet(COV_PERM_FILE)
    cov_df.insert(0, "IID", cov_df.index)
    cov_df.insert(0, "FID", cov_df.index)

    cov_df[["FID", "IID", "Sex"]].to_csv(
        COVAR,
        sep="\t",
        index=False,
        header=False,
        na_rep="NA"
    )

    qcov_cols = ["FID", "IID"] + [c for c in cov_df.columns if c.startswith("PC") or c.startswith("PEER")]
    cov_df[qcov_cols].to_csv(
        QCOVAR,
        sep="\t",
        index=False,
        header=False,
        na_rep="NA"
    )

    # Define the worker function for the current replicate.
    def process_pheno(pheno_name):
        results = []
        pheno_file = f"{gcta_input_dir}/pheno_{pheno_name}.txt"
        pheno_vals = pheno_df.loc[pheno_name]

        pd.DataFrame(
            {
                "FID": pheno_vals.index,
                "IID": pheno_vals.index,
                "Trait": pheno_vals.values,
            }
        ).to_csv(
            pheno_file,
            sep="\t",
            index=False,
            header=False,
            na_rep="NA"
        )

        for model_name, grm_path in GRM_CONFIGS.items():
            out_prefix = f"{SHM_BASE}/reml_{pheno_name}_{model_name}"
            vg, se, pv = run_gcta_reml(grm_path, pheno_file, out_prefix, COVAR, QCOVAR)
            results.append(f"{pheno_name}\t{model_name}\t{vg}\t{se}\t{pv}\n")

            for ext in [".hsq", ".log"]:
                try:
                    os.remove(f"{out_prefix}{ext}")
                except OSError:
                    pass

        try:
            os.remove(pheno_file)
        except OSError:
            pass

        return results

    # Run REML jobs concurrently for phenotypes that are still pending.
    total = len(all_phenos)
    completed = total - len(pending_phenos)

    try:
        with ThreadPoolExecutor(max_workers=WORKER_THREADS) as executor:
            future_to_pheno = {executor.submit(process_pheno, p): p for p in pending_phenos}

            for future in as_completed(future_to_pheno):
                res_lines = future.result()
                if not res_lines:
                    continue

                # Retry writes to handle transient cluster/NFS I/O failures.
                write_success = False
                for attempt in range(10):
                    try:
                        with write_lock:
                            # Recreate the output directory defensively in case it becomes unavailable.
                            os.makedirs(os.path.dirname(SUMMARY_FILE), exist_ok=True)
                            with open(SUMMARY_FILE, "a") as f:
                                f.writelines(res_lines)

                        write_success = True
                        break

                    except Exception as e:
                        print(
                            f"[Warning] Failed to write to {SUMMARY_FILE} ({e}). "
                            f"Waiting for filesystem recovery... ({attempt + 1}/10)",
                            flush=True
                        )
                        time.sleep(60)

                if not write_success:
                    print(
                        "[Error] Failed to write results after 10 retries; "
                        "skipping the current result batch.",
                        flush=True
                    )

                completed += 1
                if completed % 1000 == 0 or completed == total:
                    elapsed_min = (time.time() - rep_time_start) / 60
                    print(
                        f"  [Rep {rep_id:02d}] Processed {completed}/{total} "
                        f"({(completed / total) * 100:.1f}%) | Elapsed: {elapsed_min:.1f} min",
                        flush=True
                    )

    finally:
        # Release the temporary RAM-disk workspace.
        shutil.rmtree(SHM_BASE, ignore_errors=True)

    print(f"[Done] Replicate {rep_id:02d} completed. Results saved to -> {SUMMARY_FILE}\n", flush=True)


# ==========================================
# 5. Final Wrap-Up
# ==========================================
total_hours = (time.time() - global_time_start) / 3600

print("=" * 60)
print(f"All {TOTAL_REPS} permutation replicates finished successfully.")
print(f"Total run time: {total_hours:.2f} hours.")
print("=" * 60, flush=True)
