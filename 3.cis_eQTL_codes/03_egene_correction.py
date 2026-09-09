#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run TensorQTL permutation-based eGene discovery, then extract corrected
significant variant-gene pairs using gene-specific nominal thresholds.

This version is safer for long jobs:
  1. It sets common CPU thread env vars for a 40-core job.
  2. It saves cis.map_cis output immediately before q-value calculation.
  3. It patches TensorQTL's post.rfunc import issue when possible.
  4. It can resume from the pre-qvalue checkpoint without rerunning map_cis.

Outputs:
  cis_eQTL_all_before_qvalue.parquet/txt
      Raw TensorQTL map_cis permutation result before q-value calculation.
  cis_eQTL_all.txt
      eGene-level result after q-value and pval_nominal_threshold calculation.
  cis_eQTL_significant.txt
      Significant eGenes with qval <= FDR.
  ALL_significant_variant_gene_pairs.txt/parquet
      Nominal variant-gene pairs passing each eGene's pval_nominal_threshold.
"""

import os

# Set these before importing numpy/tensorqtl so BLAS/OpenMP can see them.
JOB_CORES = 40
os.environ.setdefault("OMP_NUM_THREADS", str(JOB_CORES))
os.environ.setdefault("MKL_NUM_THREADS", str(JOB_CORES))
os.environ.setdefault("OPENBLAS_NUM_THREADS", str(JOB_CORES))
os.environ.setdefault("NUMEXPR_NUM_THREADS", str(JOB_CORES))
os.environ.setdefault("DISABLE_PANDERA_IMPORT_WARNING", "True")

import glob
import traceback

import pandas as pd
import pyarrow.parquet as pq
import tensorqtl
from tensorqtl import cis, genotypeio, post

# ================= 1. Paths =================
plink_prefix = "/fs2/home/xiaxy/cis_eQTL/Revise_ALL/all"
pheno_path = "/fs2/home/xiaxy/cis_eQTL/Revise_ALL/phenotype.parquet"
pos_path = "/fs2/home/xiaxy/cis_eQTL/Revise_ALL/phenotype_pos.parquet"
cov_path = "/fs2/home/xiaxy/cis_eQTL/Revise_ALL/covariates.parquet"

nominal_dir = "/fs2/home/xiaxy/cis_eQTL/Revise_QQ/nominal"
output_dir = "/fs2/home/xiaxy/cis_eQTL/Revise_QQ/eGene"
os.makedirs(output_dir, exist_ok=True)

# ================= 2. Parameters =================
N_PERM = 10000
FDR = 0.05
WINDOW = 1_000_000

# True: run map_cis again. False: load cis_eQTL_all_before_qvalue.parquet.
RUN_PERMUTATION = True
EXTRACT_SIGNIFICANT_PAIRS = True

checkpoint_parquet = os.path.join(output_dir, "cis_eQTL_all_before_qvalue.parquet")
checkpoint_txt = os.path.join(output_dir, "cis_eQTL_all_before_qvalue.txt")
cis_all_file = os.path.join(output_dir, "cis_eQTL_all.txt")
cis_sig_file = os.path.join(output_dir, "cis_eQTL_significant.txt")
sig_pairs_txt = os.path.join(output_dir, "ALL_significant_variant_gene_pairs.txt")
sig_pairs_parquet = os.path.join(output_dir, "ALL_significant_variant_gene_pairs.parquet")
error_log = os.path.join(output_dir, "eGene_error_log.txt")

# ================= 3. Helpers =================
def log(msg):
    print(msg, flush=True)

def load_covariates(path):
    cov_df = pd.read_parquet(path)
    if "IID" in cov_df.columns:
        cov_df = cov_df.set_index("IID")
    if "FID" in cov_df.columns:
        cov_df = cov_df.drop(columns=["FID"])
    cov_df.index = cov_df.index.astype(str)
    return cov_df

def load_phenotype_pos(path):
    pos_df = pd.read_parquet(path)
    if "__index_level_0__" in pos_df.columns:
        pos_df = pos_df.set_index("__index_level_0__")
    elif "phenotype_id" in pos_df.columns:
        pos_df = pos_df.set_index("phenotype_id")
    elif "gene_id" in pos_df.columns:
        pos_df = pos_df.set_index("gene_id")

    pos_df.index = pos_df.index.astype(str)
    pos_df.index.name = "phenotype_id"
    pos_df = pos_df[["chr", "pos"]].copy()
    pos_df["chr"] = pos_df["chr"].astype(str)
    pos_df["pos"] = pos_df["pos"].astype(int)
    return pos_df

def load_phenotypes(path, pos_df):
    pheno_df = pd.read_parquet(path)
    drop_cols = [c for c in ["FID", "IID"] if c in pheno_df.columns]
    if drop_cols:
        pheno_df = pheno_df.drop(columns=drop_cols)

    if "phenotype_id" in pheno_df.columns:
        pheno_df = pheno_df.set_index("phenotype_id")
    elif "gene_id" in pheno_df.columns:
        pheno_df = pheno_df.set_index("gene_id")
    elif len(pheno_df) == len(pos_df):
        pheno_df.index = pos_df.index

    pheno_df.index = pheno_df.index.astype(str)
    pheno_df.columns = pheno_df.columns.astype(str)
    return pheno_df

def normalize_chrom_names(variant_df, phenotype_pos_df):
    geno_has_chr = str(variant_df["chrom"].iloc[0]).startswith("chr")
    pheno_has_chr = str(phenotype_pos_df["chr"].iloc[0]).startswith("chr")
    if geno_has_chr and not pheno_has_chr:
        phenotype_pos_df["chr"] = "chr" + phenotype_pos_df["chr"].astype(str)
    elif not geno_has_chr and pheno_has_chr:
        phenotype_pos_df["chr"] = phenotype_pos_df["chr"].astype(str).str.replace("^chr", "", regex=True)
    return phenotype_pos_df

def load_and_align_inputs():
    log(f"TensorQTL version: {tensorqtl.__version__}")
    log(f"Thread env: OMP/MKL/OPENBLAS/NUMEXPR = {JOB_CORES}")
    log("Loading covariates, phenotypes, phenotype positions, and PLINK genotypes...")

    covariates_df = load_covariates(cov_path)
    phenotype_pos_df = load_phenotype_pos(pos_path)
    phenotype_df = load_phenotypes(pheno_path, phenotype_pos_df)

    pr = genotypeio.PlinkReader(plink_prefix)
    genotype_df = pr.load_genotypes()
    genotype_df.columns = genotype_df.columns.astype(str)

    variant_df = pr.bim.set_index("snp").copy()
    phenotype_pos_df = normalize_chrom_names(variant_df, phenotype_pos_df)

    common_genes = phenotype_df.index.intersection(phenotype_pos_df.index)
    if len(common_genes) == 0:
        raise RuntimeError("No common phenotype IDs between phenotype_df and phenotype_pos_df.")

    fam_iids = set(pr.fam["iid"].astype(str))
    covariates_df.index = covariates_df.index.astype(str)
    common_samples = sorted(set(genotype_df.columns) & set(phenotype_df.columns) & set(covariates_df.index) & fam_iids)
    if len(common_samples) == 0:
        raise RuntimeError("No common samples among genotype, phenotype, covariates, and PLINK fam.")

    phenotype_df = phenotype_df.loc[common_genes, common_samples]
    phenotype_pos_df = phenotype_pos_df.loc[common_genes]
    covariates_df = covariates_df.loc[common_samples]
    genotype_df = genotype_df[common_samples]

    log(f"Aligned phenotypes: {phenotype_df.shape}")
    log(f"Aligned phenotype positions: {phenotype_pos_df.shape}")
    log(f"Aligned covariates: {covariates_df.shape}")
    log(f"Aligned genotypes: {genotype_df.shape}")
    return genotype_df, variant_df, phenotype_df, phenotype_pos_df, covariates_df

def patch_tensorqtl_rfunc():
    """
    Some TensorQTL installs do not bind post.rfunc, causing:
        NameError: name 'rfunc' is not defined
    This imports tensorqtl.rfunc and attaches it to tensorqtl.post.
    """
    try:
        from tensorqtl import rfunc

        post.rfunc = rfunc
        log("TensorQTL post.rfunc patched successfully.")
    except Exception as e:
        log(f"Warning: could not patch tensorqtl.post.rfunc: {e}")

def calculate_qvalues_safely(cis_df):
    log(f"Calculating q-values at FDR={FDR}...")
    patch_tensorqtl_rfunc()
    try:
        post.calculate_qvalues(cis_df, fdr=FDR)
    except NameError as e:
        if "rfunc" not in str(e):
            raise
        log("post.calculate_qvalues still cannot find rfunc; retrying after explicit patch...")
        patch_tensorqtl_rfunc()
        post.calculate_qvalues(cis_df, fdr=FDR)

    required = {"qval", "pval_nominal_threshold"}
    missing = required - set(cis_df.columns)
    if missing:
        raise RuntimeError(f"Q-value calculation finished but missing columns: {sorted(missing)}")

def run_or_load_egene_permutation():
    if RUN_PERMUTATION:
        genotype_df, variant_df, phenotype_df, phenotype_pos_df, covariates_df = load_and_align_inputs()
        log(f"Running TensorQTL cis.map_cis with nperm={N_PERM}, window={WINDOW}...")
        cis_df = cis.map_cis(
            genotype_df,
            variant_df,
            phenotype_df,
            phenotype_pos_df,
            covariates_df,
            nperm=N_PERM,
            window=WINDOW,
            verbose=True,
        )

        log(f"Saving pre-qvalue checkpoint: {checkpoint_parquet}")
        cis_df.to_parquet(checkpoint_parquet)
        log(f"Saving pre-qvalue checkpoint TXT: {checkpoint_txt}")
        cis_df.to_csv(checkpoint_txt, sep="\t")
    else:
        if not os.path.exists(checkpoint_parquet):
            raise FileNotFoundError(f"RUN_PERMUTATION=False but checkpoint not found: {checkpoint_parquet}")
        log(f"Loading pre-qvalue checkpoint: {checkpoint_parquet}")
        cis_df = pd.read_parquet(checkpoint_parquet)

    calculate_qvalues_safely(cis_df)

    log(f"Saving all eGene-level results: {cis_all_file}")
    cis_df.to_csv(cis_all_file, sep="\t")

    sig_df = cis_df[cis_df["qval"] <= FDR].copy()
    log(f"Significant eGenes at FDR<={FDR}: {len(sig_df):,}")
    log(f"Saving significant eGenes: {cis_sig_file}")
    sig_df.to_csv(cis_sig_file, sep="\t")
    return cis_df, sig_df

def get_nominal_files():
    files = sorted(glob.glob(os.path.join(nominal_dir, "all_nominal.cis_qtl_pairs.chr*.parquet")))
    if not files:
        files = sorted(glob.glob(os.path.join(nominal_dir, "*.parquet")))
    if not files:
        raise FileNotFoundError(f"No nominal parquet files found under {nominal_dir}")
    return files

def find_first_existing(columns, candidates, label):
    for col in candidates:
        if col in columns:
            return col
    raise ValueError(f"Cannot find {label} column. Existing columns: {columns}")

def load_significant_egene_thresholds():
    if not os.path.exists(cis_sig_file):
        raise FileNotFoundError(f"Cannot find significant eGene file: {cis_sig_file}")

    sig_df = pd.read_csv(cis_sig_file, sep="\t")
    if "phenotype_id" not in sig_df.columns:
        sig_df = sig_df.rename(columns={sig_df.columns[0]: "phenotype_id"})

    if "pval_nominal_threshold" not in sig_df.columns:
        raise ValueError("Missing pval_nominal_threshold. Q-value calculation did not complete correctly.")

    sig_df["phenotype_id"] = sig_df["phenotype_id"].astype(str)
    sig_df["pval_nominal_threshold"] = pd.to_numeric(sig_df["pval_nominal_threshold"], errors="coerce")
    sig_df = sig_df.dropna(subset=["pval_nominal_threshold"])
    threshold_dict = dict(zip(sig_df["phenotype_id"], sig_df["pval_nominal_threshold"]))
    qval_dict = dict(zip(sig_df["phenotype_id"], sig_df["qval"])) if "qval" in sig_df.columns else {}
    log(f"Loaded significant eGenes with thresholds: {len(threshold_dict):,}")
    return threshold_dict, qval_dict

def extract_significant_variant_gene_pairs():
    threshold_dict, qval_dict = load_significant_egene_thresholds()
    nominal_files = get_nominal_files()
    log(f"Found nominal parquet files: {len(nominal_files):,}")

    significant_chunks = []
    total_rows = 0
    total_candidate_gene_rows = 0

    for i, path in enumerate(nominal_files, start=1):
        cols = pq.ParquetFile(path).schema.names
        phenotype_col = find_first_existing(cols, ["phenotype_id", "gene_id", "phenotype"], "phenotype")
        variant_col = find_first_existing(cols, ["variant_id", "snp", "casual_variant"], "variant")
        p_col = find_first_existing(cols, ["pval_nominal", "pval", "pvalue"], "nominal p-value")

        optional_cols = [c for c in ["af", "slope", "slope_se"] if c in cols]
        read_cols = [phenotype_col, variant_col, p_col] + optional_cols
        df = pd.read_parquet(path, columns=read_cols)
        total_rows += len(df)

        df = df.rename(columns={phenotype_col: "phenotype_id", variant_col: "variant_id", p_col: "pval_nominal"})
        df["phenotype_id"] = df["phenotype_id"].astype(str)
        df["variant_id"] = df["variant_id"].astype(str)
        df["pval_nominal"] = pd.to_numeric(df["pval_nominal"], errors="coerce")
        df = df.dropna(subset=["pval_nominal"])

        df = df[df["phenotype_id"].isin(threshold_dict)].copy()
        total_candidate_gene_rows += len(df)
        if df.empty:
            log(f"[{i}/{len(nominal_files)}] {os.path.basename(path)}: no significant eGene rows")
            continue

        df["gene_threshold"] = df["phenotype_id"].map(threshold_dict)
        if qval_dict:
            df["gene_qval"] = df["phenotype_id"].map(qval_dict)
        df["pass_gene_specific_threshold"] = df["pval_nominal"] <= df["gene_threshold"]
        df_sig = df[df["pass_gene_specific_threshold"]].copy()

        if not df_sig.empty:
            first_cols = ["phenotype_id", "variant_id", "pval_nominal", "gene_threshold"]
            if "gene_qval" in df_sig.columns:
                first_cols.append("gene_qval")
            first_cols.append("pass_gene_specific_threshold")
            remaining_cols = [c for c in df_sig.columns if c not in first_cols]
            significant_chunks.append(df_sig[first_cols + remaining_cols])

        log(
            f"[{i}/{len(nominal_files)}] {os.path.basename(path)}: "
            f"tested rows={len(df):,}, significant pairs={len(df_sig):,}"
        )

    log(f"All nominal rows scanned: {total_rows:,}")
    log(f"Rows belonging to significant eGenes: {total_candidate_gene_rows:,}")

    if not significant_chunks:
        log("No variant-gene pairs passed gene-specific thresholds.")
        return pd.DataFrame()

    all_sig_pairs = pd.concat(significant_chunks, ignore_index=True)
    log(f"Total corrected significant variant-gene pairs: {len(all_sig_pairs):,}")
    log(f"Saving TXT: {sig_pairs_txt}")
    all_sig_pairs.to_csv(sig_pairs_txt, sep="\t", index=False)
    log(f"Saving Parquet: {sig_pairs_parquet}")
    all_sig_pairs.to_parquet(sig_pairs_parquet, index=False)
    return all_sig_pairs

def main():
    try:
        run_or_load_egene_permutation()
        if EXTRACT_SIGNIFICANT_PAIRS:
            extract_significant_variant_gene_pairs()
        log(f"Done. Output directory: {output_dir}")
    except Exception:
        with open(error_log, "a", encoding="utf-8") as f:
            f.write(traceback.format_exc())
            f.write("\n")
        raise

if __name__ == "__main__":
    main()
