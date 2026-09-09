"""
Genome-wide nonTR heritability simulation using a fast-extraction pipeline.

This script:
    - Reads .bim files directly to pool all causal candidate variants from
      rLAV and nLAV.
    - Samples N_CAUSAL variants and uses `plink --extract` to generate compact
      genotype matrices for the sampled causal variants.
    - Avoids genome-wide causal-matrix loading and is therefore substantially
      more memory efficient.
    - Uses precomputed manual GRMs for genome-wide REML testing.
"""

import os
import sys
import subprocess
import logging
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed


# ── Paths ─────────────────────────────────────────────────────────────────────
BASE = "/home/s3020226030/1_rSV/01_human_chm13"
BFILE_DIR = f"{BASE}/25_hsq_real_data/bfile"

# Output directory for the fast-extraction simulation
OUT_DIR = f"{BASE}/24_hsq_simulation/real_data_simulation/results_genome_wide_nonTR_precomputed_fast_extract_causal_1k_rep_100"

POOL_BFILE_RLAV = f"{BFILE_DIR}/rLAV_nonTR"
POOL_BFILE_NLAV = f"{BFILE_DIR}/nLAV_nonTR"

# Precomputed GRM prefixes
PRECOMP_GRM_DIR = f"{BASE}/25_hsq_real_data/01_WG_hsq_comparison_ManualGRM_raw_nonTR/GRMs"
GRM_LAV = f"{PRECOMP_GRM_DIR}/grm_manual_LAV"
GRM_RLAV_NLAV = f"{PRECOMP_GRM_DIR}/grm_manual_rLAV_nLAV"
GRM_RLAV = f"{PRECOMP_GRM_DIR}/grm_manual_rLAV"

N_CAUSAL = 1000
HSQ_LEVELS = [0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.8]
N_REPS = 100
THREADS = int(os.environ.get("SLURM_CPUS_PER_TASK", 20))
SEED = 42


for subdir in ["", "grms", "phenos", "remls", "tmp"]:
    os.makedirs(os.path.join(OUT_DIR, subdir), exist_ok=True)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler(f"{OUT_DIR}/simulation.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)


# ── Helper Functions ──────────────────────────────────────────────────────────
def run(cmd, check=True):
    r = subprocess.run(cmd, capture_output=True, text=True)

    if check and r.returncode != 0:
        log.error(f"Command failed: {' '.join(cmd)}\n{r.stderr[-500:]}")
        raise RuntimeError(f"Command failed: {' '.join(cmd)}")

    return r


def extract_and_load_mini_raw(bfile_prefix, snp_list, out_prefix, master_iids):
    """
    Use PLINK to extract a small set of variants, recode them to additive dosage
    format, and align the resulting matrix to the master sample order.
    """
    if len(snp_list) == 0:
        # Return an empty matrix if no variants were sampled from this group.
        return np.empty((len(master_iids), 0))

    snp_file = out_prefix + ".snps"
    with open(snp_file, "w") as f:
        f.write("\n".join(snp_list))

    cmd = [
        "plink",
        "--bfile", bfile_prefix,
        "--extract", snp_file,
        "--recode", "A",
        "--out", out_prefix,
        "--allow-extra-chr",
        "--silent",
    ]
    run(cmd, check=False)

    raw_file = out_prefix + ".raw"
    if not os.path.exists(raw_file):
        return np.empty((len(master_iids), 0))

    # Load the compact PLINK .raw file.
    df = pd.read_csv(raw_file, sep=r"\s+")
    df["IID"] = df["IID"].astype(str)
    df.set_index("IID", inplace=True)

    # Reindex to exactly match the master IID order. Missing samples are set to NaN.
    df_aligned = df.reindex(master_iids)

    # Remove non-genotype metadata columns from the PLINK .raw file.
    meta_cols = ["FID", "PAT", "MAT", "SEX", "PHENOTYPE"]
    snp_cols = [c for c in df_aligned.columns if c not in meta_cols]

    X = df_aligned[snp_cols].values.astype(np.float64)

    # Perform mean imputation for missing genotype values.
    col_means = np.nanmean(X, axis=0)
    nan_mask = np.isnan(X)

    # If a column is entirely missing, impute it with zero.
    if np.isnan(col_means).any():
        col_means[np.isnan(col_means)] = 0.0

    for i in range(X.shape[1]):
        X[nan_mask[:, i], i] = col_means[i]

    return X


def standardize_and_cov(X):
    """
    Standardize the genotype matrix and compute the GRM.

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


def parse_gcta_reml(hsq_file):
    """Parse the heritability estimate and standard error from a GCTA .hsq file."""
    try:
        with open(hsq_file) as f:
            for line in f:
                if line.startswith("V(G)/Vp"):
                    parts = line.split()
                    return float(parts[1]), float(parts[2])
    except Exception:
        pass

    return None, None


def gcta_reml(grm_prefix, phen_file, out_prefix):
    """Run GCTA REML and return the estimated heritability and standard error."""
    cmd = [
        "gcta64",
        "--reml",
        "--grm", grm_prefix,
        "--pheno", phen_file,
        "--out", out_prefix,
        "--thread-num", "1",
    ]
    run(cmd, check=False)

    return parse_gcta_reml(out_prefix + ".hsq")


# ── Main Pipeline ─────────────────────────────────────────────────────────────
def main():
    log.info(f"Fast-extraction simulation started. Threads: {THREADS}")

    # Phase 1: determine the master sample list.
    log.info("Phase 1: Intersecting .fam files to determine the master sample list...")
    fam_r = pd.read_csv(
        POOL_BFILE_RLAV + ".fam",
        sep=r"\s+",
        header=None,
        usecols=[1],
        names=["IID"],
    )
    fam_n = pd.read_csv(
        POOL_BFILE_NLAV + ".fam",
        sep=r"\s+",
        header=None,
        usecols=[1],
        names=["IID"],
    )

    valid_iids = np.intersect1d(fam_r["IID"].astype(str), fam_n["IID"].astype(str))

    # Preserve the rLAV sample order.
    master_iids = [str(i) for i in fam_r["IID"] if str(i) in valid_iids]

    log.info(f"Aligned {len(master_iids)} valid samples shared between rLAV and nLAV.")

    # Phase 2: read BIM files and pool candidate variants.
    log.info("Phase 2: Reading .bim files to pool all causal candidates...")
    bim_r = pd.read_csv(
        POOL_BFILE_RLAV + ".bim",
        sep="\t",
        header=None,
        usecols=[1],
        names=["id"],
    )
    bim_n = pd.read_csv(
        POOL_BFILE_NLAV + ".bim",
        sep="\t",
        header=None,
        usecols=[1],
        names=["id"],
    )

    snps_r_set = set(bim_r["id"])
    snps_n_set = set(bim_n["id"])
    all_candidates = list(snps_r_set.union(snps_n_set))

    log.info(f"Pooled {len(all_candidates)} unique candidate variants from rLAV and nLAV.")

    # Phase 3: map precomputed genome-wide GRMs.
    log.info("Phase 3: Mapping precomputed genome-wide GRMs...")
    grm_global = {
        "rLAV": GRM_RLAV,
        "LAV": GRM_LAV,
        "rLAV+nLAV": GRM_RLAV_NLAV,
    }

    # Phase 4: perform per-replicate fast extraction and phenotype simulation.
    log.info("Phase 4: Running per-replicate PLINK extraction and phenotype simulation...")
    reml_tasks = []
    n_samples = len(master_iids)
    rng = np.random.default_rng(SEED)

    for rep in range(1, N_REPS + 1):
        # 1. Sample causal variants.
        if len(all_candidates) >= N_CAUSAL:
            sampled_snps = rng.choice(all_candidates, N_CAUSAL, replace=False)
        else:
            sampled_snps = np.array(all_candidates)

        # Split sampled variants according to whether they belong to rLAV or nLAV.
        snps_in_r = [s for s in sampled_snps if s in snps_r_set]
        snps_in_n = [s for s in sampled_snps if s in snps_n_set]

        # 2. Extract the sampled variants using PLINK and load compact matrices.
        pfx_r = os.path.join(OUT_DIR, "tmp", f"causal_r_rep{rep}")
        pfx_n = os.path.join(OUT_DIR, "tmp", f"causal_n_rep{rep}")

        X_r = extract_and_load_mini_raw(POOL_BFILE_RLAV, snps_in_r, pfx_r, master_iids)
        X_n = extract_and_load_mini_raw(POOL_BFILE_NLAV, snps_in_n, pfx_n, master_iids)

        # Concatenate the sampled rLAV and nLAV variants into the causal matrix.
        X_causal = np.hstack([X_r, X_n])

        # 3. Build the causal oracle GRM for the current replicate.
        G_causal, P_causal, Z_causal = standardize_and_cov(X_causal)
        causal_grm_pfx = f"{OUT_DIR}/grms/causal_oracle_r{rep}"
        save_grm(G_causal, P_causal, master_iids, causal_grm_pfx)

        # 4. Simulate phenotypes across heritability levels.
        for hsq in HSQ_LEVELS:
            beta = rng.normal(0, 1.0 / np.sqrt(P_causal), size=P_causal)

            G_raw = Z_causal @ beta
            var_G_raw = np.var(G_raw, ddof=1)
            G_true = G_raw * np.sqrt(hsq / var_G_raw) if var_G_raw > 0 else G_raw

            e_raw = rng.normal(0, 1, size=n_samples)
            var_e_raw = np.var(e_raw, ddof=1)
            e = e_raw * np.sqrt((1.0 - hsq) / var_e_raw)

            Y = G_true + e
            emp_hsq = float(np.var(G_true, ddof=1) / np.var(Y, ddof=1))

            phen_path = os.path.join(OUT_DIR, "phenos", f"hsq{int(hsq * 100)}_rep{rep}.phen")
            with open(phen_path, "w") as f:
                for iid, y_val in zip(master_iids, Y):
                    f.write(f"{iid}\t{iid}\t{y_val:.8f}\n")

            grms_to_test = {
                "causal": causal_grm_pfx,
                "rLAV": grm_global["rLAV"],
                "LAV": grm_global["LAV"],
                "rLAV+nLAV": grm_global["rLAV+nLAV"],
            }

            for model, grm_pfx in grms_to_test.items():
                safe_model_name = model.replace("+", "_")
                out_pfx = os.path.join(
                    OUT_DIR,
                    "remls",
                    f"reml_{safe_model_name}_h{int(hsq * 100)}_r{rep}",
                )

                reml_tasks.append({
                    "n_causal": P_causal,
                    "model": model,
                    "hsq": hsq,
                    "rep": rep,
                    "emp_hsq": emp_hsq,
                    "grm_pfx": grm_pfx,
                    "phen_path": phen_path,
                    "out_pfx": out_pfx,
                })

    # Phase 5: execute REML jobs in parallel.
    log.info(f"Phase 5: Executing {len(reml_tasks)} REML models concurrently...")
    all_results = []
    n_done = 0

    def process_reml(task):
        est, se = gcta_reml(task["grm_pfx"], task["phen_path"], task["out_pfx"])

        return {
            "n_causal": task["n_causal"],
            "model": task["model"],
            "true_hsq": task["hsq"],
            "rep": task["rep"],
            "emp_hsq": task["emp_hsq"],
            "est_hsq": est,
            "se": se,
        }

    with ThreadPoolExecutor(max_workers=THREADS) as ex:
        futures = [ex.submit(process_reml, t) for t in reml_tasks]

        for f in as_completed(futures):
            all_results.append(f.result())
            n_done += 1

            if n_done % 50 == 0 or n_done == len(reml_tasks):
                log.info(f"REML progress: {n_done}/{len(reml_tasks)} completed.")

    # Phase 6: export detailed and summarized results.
    out_tsv = os.path.join(OUT_DIR, "results_fast_extract.tsv")
    df = pd.DataFrame(all_results)
    df.to_csv(out_tsv, sep="\t", index=False)
    log.info(f"Saved {len(df)} rows -> {out_tsv}")

    if not df.empty:
        summary = (
            df.dropna(subset=["est_hsq"])
            .groupby(["model", "true_hsq"])
            .agg(
                mean_est=("est_hsq", "mean"),
                mean_emp=("emp_hsq", "mean"),
                n=("est_hsq", "count"),
            )
            .reset_index()
        )
        summary["bias"] = summary["mean_est"] - summary["true_hsq"]

        log.info("\n" + summary.to_string(index=False))
        summary.to_csv(
            os.path.join(OUT_DIR, "summary_fast_extract.tsv"),
            sep="\t",
            index=False,
        )


if __name__ == "__main__":
    main()
