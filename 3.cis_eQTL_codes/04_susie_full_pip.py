#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run TensorQTL SuSiE and save both summary results and full SuSiE results.

This version uses:
    summary_only=False

So susie.map returns:
    summary_df, susie_res

Saved outputs per chunk:
    susie_summary_chunk_{i}.parquet
    susie_full_result_chunk_{i}.pkl
    susie_full_pip_chunk_{i}.parquet

Final merged outputs:
    susie_finemapping_ALL.parquet
    susie_full_pip_ALL.parquet

Important:
    TensorQTL's susie.map may still drop phenotypes without credible sets from
    susie_res before returning. The full PIP table saved here is full PIP for
    phenotypes retained in susie_res, not necessarily all phenotypes if the
    TensorQTL internal code drops no-CS phenotypes.
"""

import os
import pickle

import pandas as pd
from tensorqtl import genotypeio, susie
from tqdm import tqdm

# ================= 1. Paths =================
plink_prefix = "/fs2/home/xiaxy/cis_eQTL/Revise_ALL/all"
pheno_path = "/fs2/home/xiaxy/cis_eQTL/Revise_ALL/phenotype.parquet"
pos_path = "/fs2/home/xiaxy/cis_eQTL/Revise_ALL/phenotype_pos.parquet"
cov_path = "/fs2/home/xiaxy/cis_eQTL/Revise_ALL/covariates.parquet"
output_dir = "/fs2/home/xiaxy/cis_eQTL/Revise_SUSIE/Result2"

os.makedirs(output_dir, exist_ok=True)

chunk_size = 500
window = 1000000
L = 10

def normalize_pheno_position_index(pos_df):
    if "__index_level_0__" in pos_df.columns:
        pos_df = pos_df.set_index("__index_level_0__")
    elif "phenotype_id" in pos_df.columns:
        pos_df = pos_df.set_index("phenotype_id")
    elif "gene_id" in pos_df.columns:
        pos_df = pos_df.set_index("gene_id")
    pos_df.index.name = "phenotype_id"
    pos_df = pos_df[["chr", "pos"]].copy()
    pos_df["pos"] = pos_df["pos"].astype(int)
    return pos_df

def normalize_pheno_df(pheno_df, pos_df):
    for col in ["IID", "FID"]:
        if col in pheno_df.columns:
            pheno_df = pheno_df.drop(columns=[col])

    # If phenotype rows are unnamed/range-like, assume the row order matches
    # phenotype_pos.parquet. Otherwise align by shared phenotype IDs.
    if len(pheno_df.index) == len(pos_df.index) and not pheno_df.index.isin(pos_df.index).any():
        pheno_df.index = pos_df.index
    else:
        common_phenotypes = pheno_df.index.intersection(pos_df.index)
        if len(common_phenotypes) == 0:
            raise ValueError(
                "No shared phenotype IDs between phenotype.parquet and phenotype_pos.parquet. "
                "Check phenotype row index."
            )
        pheno_df = pheno_df.loc[common_phenotypes]
        pos_df = pos_df.loc[common_phenotypes]

    return pheno_df, pos_df

def extract_full_pip_table(susie_res):
    """Extract full PIP values from TensorQTL susie_res into a long DataFrame."""
    rows = []

    for phenotype_id, result in susie_res.items():
        if result is None or "pip" not in result:
            continue

        pip_obj = result["pip"]

        if isinstance(pip_obj, pd.Series):
            tmp = pip_obj.rename("pip").reset_index()
            tmp = tmp.rename(columns={tmp.columns[0]: "variant_id"})
        elif isinstance(pip_obj, pd.DataFrame):
            tmp = pip_obj.copy()
            if "pip" not in tmp.columns:
                if tmp.shape[1] == 1:
                    tmp = tmp.rename(columns={tmp.columns[0]: "pip"})
                else:
                    tmp = tmp.reset_index()
                    value_cols = [c for c in tmp.columns if c != "index"]
                    if len(value_cols) == 1:
                        tmp = tmp.rename(columns={"index": "variant_id", value_cols[0]: "pip"})
                    else:
                        raise ValueError(
                            f"Cannot infer pip column for phenotype {phenotype_id}; "
                            f"columns={list(tmp.columns)}"
                        )
            if "variant_id" not in tmp.columns:
                if "snp" in tmp.columns:
                    tmp = tmp.rename(columns={"snp": "variant_id"})
                else:
                    tmp = tmp.reset_index().rename(columns={"index": "variant_id"})
        else:
            tmp = pd.DataFrame({"pip": pip_obj})
            tmp = tmp.reset_index().rename(columns={"index": "variant_id"})

        if "variant_id" not in tmp.columns or "pip" not in tmp.columns:
            raise ValueError(
                f"Cannot extract variant_id/pip for phenotype {phenotype_id}; "
                f"columns={list(tmp.columns)}"
            )

        tmp = tmp[["variant_id", "pip"]].copy()
        tmp.insert(0, "phenotype_id", phenotype_id)
        rows.append(tmp)

    if not rows:
        return pd.DataFrame(columns=["phenotype_id", "variant_id", "pip"])

    full_pip_df = pd.concat(rows, ignore_index=True)
    full_pip_df["variant_id"] = full_pip_df["variant_id"].astype(str)
    full_pip_df["pip"] = pd.to_numeric(full_pip_df["pip"], errors="coerce")
    return full_pip_df.dropna(subset=["pip"])

print("Loading and formatting data...")

# ================= 2. Load covariates / phenotypes =================
cov_df = pd.read_parquet(cov_path)
if "IID" in cov_df.columns:
    cov_df = cov_df.set_index("IID")
if "FID" in cov_df.columns:
    cov_df = cov_df.drop(columns=["FID"])

pos_df = pd.read_parquet(pos_path)
pos_df = normalize_pheno_position_index(pos_df)

pheno_df = pd.read_parquet(pheno_path)
pheno_df, pos_df = normalize_pheno_df(pheno_df, pos_df)

# ================= 3. Load genotypes =================
pr = genotypeio.PlinkReader(plink_prefix)
genotype_df = pr.load_genotypes()
variant_df = pr.bim.set_index("snp")

# ================= 4. Fix chromosome naming =================
pheno_chrom_sample = str(pos_df["chr"].iloc[0])
geno_chrom_sample = str(variant_df["chrom"].iloc[0])

if pheno_chrom_sample.startswith("chr") and not geno_chrom_sample.startswith("chr"):
    pos_df["chr"] = pos_df["chr"].astype(str).str.replace("^chr", "", regex=True)
elif not pheno_chrom_sample.startswith("chr") and geno_chrom_sample.startswith("chr"):
    pos_df["chr"] = "chr" + pos_df["chr"].astype(str)

# ================= 5. Align samples =================
common_samples = sorted(set(pheno_df.columns) & set(cov_df.index) & set(pr.fam["iid"]))
print(f"Aligned {len(common_samples)} common samples.")
if not common_samples:
    raise ValueError("No common samples across phenotype, covariates, and genotype files.")

pheno_df = pheno_df[common_samples]
cov_df = cov_df.loc[common_samples]
genotype_df = genotype_df[common_samples]

# ================= 6. Run SuSiE by chunk =================
gene_list = pheno_df.index.tolist()
all_summary_dfs = []
full_pip_chunk_files = []

print(f"Start SuSiE: {len(gene_list)} genes, chunk_size={chunk_size}.")

for i in tqdm(range(0, len(gene_list), chunk_size), desc="Chunks Processing"):
    chunk_genes = gene_list[i : i + chunk_size]
    pheno_chunk = pheno_df.loc[chunk_genes]
    pos_chunk = pos_df.loc[chunk_genes]

    summary_chunk_file = os.path.join(output_dir, f"susie_summary_chunk_{i}.parquet")
    full_result_chunk_file = os.path.join(output_dir, f"susie_full_result_chunk_{i}.pkl")
    full_pip_chunk_file = os.path.join(output_dir, f"susie_full_pip_chunk_{i}.parquet")

    if (
        os.path.exists(summary_chunk_file)
        and os.path.exists(full_result_chunk_file)
        and os.path.exists(full_pip_chunk_file)
    ):
        print(f"Chunk {i} already exists, skipping.")
        summary_df = pd.read_parquet(summary_chunk_file)
        all_summary_dfs.append(summary_df)
        full_pip_chunk_files.append(full_pip_chunk_file)
        continue

    try:
        summary_df, susie_res = susie.map(
            genotype_df,
            variant_df,
            pheno_chunk,
            pos_chunk,
            cov_df,
            window=window,
            L=L,
            summary_only=False,
        )

        if summary_df is not None and not summary_df.empty:
            summary_df.to_parquet(summary_chunk_file)
            all_summary_dfs.append(summary_df)

        with open(full_result_chunk_file, "wb") as f:
            pickle.dump(susie_res, f, protocol=pickle.HIGHEST_PROTOCOL)

        full_pip_df = extract_full_pip_table(susie_res)
        full_pip_df.to_parquet(full_pip_chunk_file, index=False)
        full_pip_chunk_files.append(full_pip_chunk_file)

        print(
            f"Chunk {i}: saved summary rows={0 if summary_df is None else len(summary_df):,}, "
            f"full PIP rows={len(full_pip_df):,}, "
            f"phenotypes in full result={len(susie_res):,}"
        )

    except Exception as e:
        print(f"\nChunk {i} failed: {e}")
        with open(os.path.join(output_dir, "error_log_full_result.txt"), "a") as f:
            f.write(f"Chunk {i} failed: {e}\n")

# ================= 7. Merge final outputs =================
print("Merging final outputs...")

if all_summary_dfs:
    final_summary_df = pd.concat(all_summary_dfs, ignore_index=True)
    final_summary_path = os.path.join(output_dir, "susie_finemapping_ALL.parquet")
    final_summary_df.to_parquet(final_summary_path, index=False)
    print(f"Saved final summary: {final_summary_path}")
else:
    print("No summary results generated. Check error_log_full_result.txt.")

if full_pip_chunk_files:
    full_pip_dfs = [pd.read_parquet(path) for path in full_pip_chunk_files]
    final_full_pip_df = pd.concat(full_pip_dfs, ignore_index=True)
    final_full_pip_path = os.path.join(output_dir, "susie_full_pip_ALL.parquet")
    final_full_pip_df.to_parquet(final_full_pip_path, index=False)
    print(f"Saved final full PIP table: {final_full_pip_path}")
else:
    print("No full PIP results generated. Check error_log_full_result.txt.")

print("Done.")
