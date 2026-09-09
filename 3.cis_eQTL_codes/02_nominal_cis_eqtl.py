#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pandas as pd
import tensorqtl
import tensorqtl.genotypeio as genotypeio
import tensorqtl.cis as cis

# ==========================================
# 0. Path configuration
# ==========================================
plink_prefix = "/fs2/home/xiaxy/cis_eQTL/Revise_ALL/all"
pheno_path   = "/fs2/home/xiaxy/cis_eQTL/Revise_ALL/phenotype.parquet"
pos_path     = "/fs2/home/xiaxy/cis_eQTL/Revise_ALL/phenotype_pos.parquet"
cov_path     = "/fs2/home/xiaxy/cis_eQTL/Revise_ALL/covariates.parquet"
output_dir   = "/fs2/home/xiaxy/cis_eQTL/Revise_QQ/nominal"

os.makedirs(output_dir, exist_ok=True)

print(f"TensorQTL Version: {tensorqtl.__version__}")

# ==========================================
# 1. Load phenotypes and covariates
# ==========================================
print("Step 1: Loading Phenotypes...")

phenotype_df = pd.read_parquet(pheno_path)
phenotype_pos_df = pd.read_parquet(pos_path)
covariates_df = pd.read_parquet(cov_path)

print(f"  - Phenotypes: {phenotype_df.shape}")

# ==========================================
# 2. Load PLINK bfile and normalize chromosome names
# ==========================================
print("\nStep 2: Loading PLINK bfile directly...")

def load_plink_as_df_fixed(plink_prefix):
    print(f"  - Loading prefix: {plink_prefix} ...")
    pr = genotypeio.PlinkReader(plink_prefix)
    geno_df = pr.load_genotypes()
    var_df = pr.bim.set_index("snp")[["chrom", "pos"]]

    # Normalize chromosome names: 1 -> chr1
    var_df["chrom"] = var_df["chrom"].astype(str).apply(
        lambda x: f"chr{x}" if not x.startswith("chr") else x
    )

    geno_df.index = var_df.index

    if geno_df.isnull().any().any():
        geno_df = geno_df.fillna(-9.0)

    return geno_df, var_df

genotype_df, variant_df = load_plink_as_df_fixed(plink_prefix)

# ==========================================
# 3. Sort variants by chromosome and genomic position
# ==========================================
print("\nStep 3: Sorting variants by genomic position...")

variant_df["chr_num"] = (
    variant_df["chrom"]
    .str.replace("chr", "", regex=False)
    .replace("X", "23")
    .replace("Y", "24")
    .replace("M", "25")
)
variant_df["chr_num"] = pd.to_numeric(variant_df["chr_num"], errors="coerce").fillna(99)

variant_df = variant_df.sort_values(by=["chr_num", "pos"])
variant_df = variant_df.drop(columns=["chr_num"])

# Synchronize genotype row order
genotype_df = genotype_df.loc[variant_df.index]
genotype_df = genotype_df.astype("float32")

print(f"  - Final Genotype Matrix: {genotype_df.shape}")
print(f"  - Sample Variant IDs (sorted): {variant_df.index[:5].tolist()}")

# ==========================================
# 4. Align samples
# ==========================================
print("\nStep 4: Aligning samples...")

common_samples = sorted(list(set(genotype_df.columns) & set(phenotype_df.columns)))

print(f"  - Aligned samples: {len(common_samples)}")

genotype_df = genotype_df[common_samples]
phenotype_df = phenotype_df[common_samples]
covariates_df = covariates_df.loc[common_samples]

# ==========================================
# 5. Run TensorQTL nominal cis-eQTL mapping
# ==========================================
print("\nStep 5: Running Nominal Pass...")

cis.map_nominal(
    genotype_df=genotype_df,
    variant_df=variant_df,
    phenotype_df=phenotype_df,
    phenotype_pos_df=phenotype_pos_df,
    covariates_df=covariates_df,
    prefix="all_nominal",
    output_dir=output_dir,
    window=1000000,
    verbose=True
)

print("\n[DONE] Nominal cis-eQTL analysis finished.")
print(f"Results saved to: {output_dir}")
