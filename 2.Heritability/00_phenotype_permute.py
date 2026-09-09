"""
Generate synchronized sample permutations for phenotype and covariate matrices.

The same sample-level permutation is applied to both phenotype and covariate
data, preserving their pairing after permutation.
"""

import pandas as pd
import numpy as np
import os


PHENO_FILE = "16_cis_eQTL/phenotype.parquet"
COV_FILE = "16_cis_eQTL/covariates.parquet"
MAIN_DIR = "25_hsq_real_data"
OUT_DIR = f"{MAIN_DIR}/05_WG_hsq_permutation/synced_inputs"

os.makedirs(OUT_DIR, exist_ok=True)


print("[1/2] Loading original data...", flush=True)
pheno_df = pd.read_parquet(PHENO_FILE)
cov_df = pd.read_parquet(COV_FILE)

# Strictly align samples between phenotype and covariate matrices.
common_samples = pheno_df.columns.intersection(cov_df.index).tolist()
print(f"      -> Aligned {len(common_samples)} samples.")

pheno_df = pheno_df[common_samples]
cov_df = cov_df.loc[common_samples]


print("[2/2] Generating 30 synchronized permutations...", flush=True)
np.random.seed(42)

for i in range(1, 31):
    # 1. Generate one shuffled sample-label order.
    shuffled_samples = np.random.permutation(common_samples)

    # 2. Replace phenotype column labels. Samples are stored as columns.
    pheno_perm = pheno_df.copy()
    pheno_perm.columns = shuffled_samples

    # Reorder columns back to the original sample order for downstream consistency.
    pheno_perm = pheno_perm[common_samples]

    # 3. Replace covariate row labels. Samples are stored as the index.
    cov_perm = cov_df.copy()
    cov_perm.index = shuffled_samples

    # Reorder rows back to the original sample order for downstream consistency.
    cov_perm = cov_perm.loc[common_samples]

    # 4. Save the paired permuted files.
    pheno_out = f"{OUT_DIR}/pheno_perm_{i:02d}.parquet"
    cov_out = f"{OUT_DIR}/covar_perm_{i:02d}.parquet"

    pheno_perm.to_parquet(pheno_out)
    cov_perm.to_parquet(cov_out)

    print(f"  [Done] Saved synchronized permutation pair {i:02d}")

print("\n[Done] All 30 synchronized permutations were generated successfully.")