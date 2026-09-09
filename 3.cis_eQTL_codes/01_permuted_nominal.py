#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run 10 phenotype permutations for nominal cis-eQTL mapping.

Optimized for a 40-core job:
  - 10 permutation workers run in parallel.
  - Each worker uses 4 BLAS/OpenMP threads.
  - Output is saved under Revise_QQ/permuted10 to avoid mixing with permuted30.

This script only saves permutation nominal parquet files.
It does not calculate lambda_GC and does not draw QQ plots.

Output layout:
  /fs2/home/xiaxy/cis_eQTL/Revise_QQ/permuted10/
      perm_0/
      perm_1/
      ...
      perm_9/
"""

import glob
import os
from multiprocessing import Pool

# ================= 0. Thread settings =================
NUM_PERMUTATIONS = 10
JOB_CORES = 40
NUM_WORKERS = min(NUM_PERMUTATIONS, JOB_CORES)
THREADS_PER_WORKER = max(1, JOB_CORES // NUM_WORKERS)

# Set thread env before importing numpy/pandas/tensorqtl.
os.environ["OMP_NUM_THREADS"] = str(THREADS_PER_WORKER)
os.environ["MKL_NUM_THREADS"] = str(THREADS_PER_WORKER)
os.environ["OPENBLAS_NUM_THREADS"] = str(THREADS_PER_WORKER)
os.environ["NUMEXPR_NUM_THREADS"] = str(THREADS_PER_WORKER)

import numpy as np
import pandas as pd
from tensorqtl import cis, genotypeio

# ================= 1. Paths =================
plink_prefix = "/fs2/home/xiaxy/cis_eQTL/Revise_ALL/all"
pheno_path = "/fs2/home/xiaxy/cis_eQTL/Revise_ALL/phenotype.parquet"
pos_path = "/fs2/home/xiaxy/cis_eQTL/Revise_ALL/phenotype_pos.parquet"
cov_path = "/fs2/home/xiaxy/cis_eQTL/Revise_ALL/covariates.parquet"
output_dir = "/fs2/home/xiaxy/cis_eQTL/Revise_QQ/permuted10"
os.makedirs(output_dir, exist_ok=True)

CIS_WINDOW = 1_000_000

# ================= 2. Load shared data once =================
print("[1/3] Loading genotype, phenotype, phenotype positions and covariates...")

pr = genotypeio.PlinkReader(plink_prefix)
variant_df = pr.bim.set_index("snp")

phenotype_pos_df = pd.read_parquet(pos_path)
if "__index_level_0__" in phenotype_pos_df.columns:
    phenotype_pos_df = phenotype_pos_df.set_index("__index_level_0__")
elif "phenotype_id" in phenotype_pos_df.columns:
    phenotype_pos_df = phenotype_pos_df.set_index("phenotype_id")
elif "gene_id" in phenotype_pos_df.columns:
    phenotype_pos_df = phenotype_pos_df.set_index("gene_id")
phenotype_pos_df.index.name = "phenotype_id"

if "chr" not in phenotype_pos_df.columns or "pos" not in phenotype_pos_df.columns:
    raise ValueError("phenotype_pos.parquet must contain chr and pos columns.")

phenotype_pos_df = phenotype_pos_df[["chr", "pos"]].copy()
phenotype_pos_df["chr"] = phenotype_pos_df["chr"].astype(str)
phenotype_pos_df["pos"] = phenotype_pos_df["pos"].astype(int)

# Keep chromosome naming consistent between genotype and phenotype position.
geno_has_chr = str(variant_df["chrom"].iloc[0]).startswith("chr")
pheno_has_chr = str(phenotype_pos_df["chr"].iloc[0]).startswith("chr")
if geno_has_chr != pheno_has_chr:
    if pheno_has_chr:
        phenotype_pos_df["chr"] = phenotype_pos_df["chr"].astype(str).str.replace(
            "^chr", "", regex=True
        )
    else:
        phenotype_pos_df["chr"] = "chr" + phenotype_pos_df["chr"].astype(str)

phenotype_df = pd.read_parquet(pheno_path)
if "phenotype_id" in phenotype_df.columns:
    phenotype_df = phenotype_df.set_index("phenotype_id")
elif "gene_id" in phenotype_df.columns:
    phenotype_df = phenotype_df.set_index("gene_id")

for col in ["FID", "IID"]:
    if col in phenotype_df.columns:
        phenotype_df = phenotype_df.drop(columns=[col])

if len(phenotype_df) == len(phenotype_pos_df):
    phenotype_df.index = phenotype_pos_df.index

covariates_df = pd.read_parquet(cov_path)
if "IID" in covariates_df.columns:
    covariates_df = covariates_df.set_index("IID")
if "FID" in covariates_df.columns:
    covariates_df = covariates_df.drop(columns=["FID"])

genotype_df = pr.load_genotypes()

genotype_df.columns = genotype_df.columns.astype(str)
phenotype_df.columns = phenotype_df.columns.astype(str)
covariates_df.index = covariates_df.index.astype(str)

common_genes = phenotype_df.index.intersection(phenotype_pos_df.index)
common_samples = sorted(
    set(phenotype_df.columns)
    & set(covariates_df.index)
    & set(pr.fam["iid"].astype(str))
    & set(genotype_df.columns)
)
if not common_samples:
    raise ValueError("No common samples found across phenotype, covariates, genotype and PLINK fam.")
if len(common_genes) == 0:
    raise ValueError("No common phenotypes found between phenotype and phenotype_pos.")

GLOBAL_PHENO = phenotype_df.loc[common_genes, common_samples]
GLOBAL_COV = covariates_df.loc[common_samples]
GLOBAL_GENO = genotype_df[common_samples]
GLOBAL_POS = phenotype_pos_df.loc[common_genes]
GLOBAL_VAR = variant_df
GLOBAL_SAMPLES = common_samples

print(f"      Common samples: {len(GLOBAL_SAMPLES)}")
print(f"      Variants: {GLOBAL_VAR.shape[0]}")
print(f"      Phenotypes: {GLOBAL_PHENO.shape[0]}")
print(f"      Workers: {NUM_WORKERS}")
print(f"      Threads per worker: {THREADS_PER_WORKER}")

# ================= 3. Worker =================
def run_single_permutation(seed: int):
    """Run one phenotype permutation and save TensorQTL nominal results."""
    worker_dir = os.path.join(output_dir, f"perm_{seed}")
    os.makedirs(worker_dir, exist_ok=True)

    success_marker = os.path.join(worker_dir, "_SUCCESS")
    existing_files = glob.glob(os.path.join(worker_dir, "*.parquet"))
    if existing_files and os.path.exists(success_marker):
        print(f"[skip] perm_{seed}: existing completed output found.")
        return seed, "skipped", len(existing_files)

    if existing_files and not os.path.exists(success_marker):
        print(f"[warn] perm_{seed}: parquet files exist but _SUCCESS is missing; rerunning.")

    rng = np.random.default_rng(seed)
    permuted_idx = rng.permutation(len(GLOBAL_SAMPLES))

    pheno_permuted = GLOBAL_PHENO.copy()
    pheno_permuted.iloc[:, :] = GLOBAL_PHENO.iloc[:, permuted_idx].to_numpy()

    cis.map_nominal(
        genotype_df=GLOBAL_GENO,
        variant_df=GLOBAL_VAR,
        phenotype_df=pheno_permuted,
        phenotype_pos_df=GLOBAL_POS,
        prefix=f"perm_{seed}",
        covariates_df=GLOBAL_COV,
        window=CIS_WINDOW,
        output_dir=worker_dir,
        verbose=False,
    )

    out_files = glob.glob(os.path.join(worker_dir, "*.parquet"))
    if not out_files:
        raise RuntimeError(f"perm_{seed} finished but no parquet output was found.")

    with open(success_marker, "w", encoding="utf-8") as f:
        f.write("completed\n")

    print(f"[done] perm_{seed}: {len(out_files)} parquet files saved.")
    return seed, "done", len(out_files)

# ================= 4. Run permutations =================
if __name__ == "__main__":
    print("[2/3] Starting phenotype permutations...")
    print(f"      Output directory: {output_dir}")
    print(f"      Permutations: {NUM_PERMUTATIONS}")
    print(f"      Job cores requested: {JOB_CORES}")
    print(f"      Worker processes used: {NUM_WORKERS}")
    print(f"      Threads per worker: {THREADS_PER_WORKER}")

    with Pool(NUM_WORKERS) as pool:
        results = pool.map(run_single_permutation, range(NUM_PERMUTATIONS))

    done = sum(1 for _, status, _ in results if status == "done")
    skipped = sum(1 for _, status, _ in results if status == "skipped")
    total_files = sum(n_files for _, _, n_files in results)

    print("[3/3] All requested permutations handled.")
    print(f"      Newly completed: {done}")
    print(f"      Skipped existing: {skipped}")
    print(f"      Total parquet files reported: {total_files}")
    print(f"      Saved under: {output_dir}")
