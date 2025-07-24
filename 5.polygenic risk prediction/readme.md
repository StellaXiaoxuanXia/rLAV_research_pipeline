# Prediction Pipeline

## Bfile Preparation

### SV Pruning:
Remove SVs with LD > 0.99 from SNP/indel:

```bash
plink --bfile sindel_pruned --bmerge SV_QCed --make-bed --out sindel+SV
plink --bfile sindel+SV --r2 --out SV_ld
awk '$7>0.99 && ( ($3~/^SV/ && $6!~/^SV/) || ($6~/^SV/ && $3!~/^SV/) ) {
        print ($3~/^SV/?$3:$6)
        }' SV_ld.ld | sort -u > remove_SV.txt
plink --bfile SV_QCed --exclude remove_SV.txt --make-bed --out SV_pruned
````

### rSV Pruning:

Remove rSVs with LD > 0.99 from SNP/indel or SV:

```bash
plink --bfile sindel+SV --bmerge rSV_QCed --make-bed --out sindel+SV+rSV
plink --bfile sindel+SV+rSV --r2 --out rSV_ld
awk '$7>0.99 && ( ($3~/^Group/ && $6!~/^Group/) || ($6~/^Group/ && $3!~/^Group/) ) {
        print ($3~/^Group/?$3:$6)
        }' rSV_ld.ld | sort -u > remove_rSV.txt
plink --bfile rSV_QCed --exclude remove_rSV.txt --make-bed --out rSV_pruned
```

## 1. C+T

Randomly select 10% for testing and 90% for training + validation, and perform 5-fold cross-validation for threshold selection.

### Demo Code for Training:

```python
import os
import subprocess
from joblib import Parallel, delayed

def run_cpt_pipeline(
    trait,
    seed,
    r2=0.2,
    kb=50,
    p1=1,
    threads=1,
    covar="cov/cov.txt",
    pheno_dir="phenotype/",
    folds=range(1, 6),
):
    outdir = f"prediction/random_{seed}"
    bfile_map = {
        1: "prediction/bfile/sindel_pruned",
        2: "prediction/bfile/SV_pruned",
        3: "prediction/bfile/rSV_pruned"
    }
    for i in range(1, 4):
        bfile_path = bfile_map[i]
        suffix = f"_{bfile_path}"

        def run(cmd):
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        def process_fold(fold):
            base = os.path.join(outdir, f"fold{fold}")
            gwas_pref = f"{base}/gwas/trait{trait}{suffix}"
            clump_pref = f"{base}/clump/trait{trait}_clump{suffix}"
            prs_pref = f"{base}/prs/trait{trait}_PRS{suffix}"
            freq_pref = f"{base}/freq/train_freq_fold{fold}{suffix}"
            pheno_file = os.path.join(pheno_dir, f"{trait}_std.txt")

            os.makedirs(os.path.join(base, "gwas"), exist_ok=True)
            os.makedirs(os.path.join(base, "clump"), exist_ok=True)
            os.makedirs(os.path.join(base, "prs"), exist_ok=True)
            os.makedirs(os.path.join(base, "freq"), exist_ok=True)

            freq_file = f"{freq_pref}.afreq"
            range_list_path = "range_list"

            # GWAS
            run([
                "plink2", "--bfile", bfile_path, "--keep", f"{base}/train.txt",
                "--pheno", pheno_file, "--covar", covar,
                "--glm", "hide-covar", "cols=+err",
                "--threads", str(threads),
                "--out", gwas_pref
            ])

            # Clumping
            run([
                "plink2", "--bfile", bfile_path, "--keep", f"{base}/train.txt",
                "--clump", f"{gwas_pref}.PHENO1.glm.linear",
                "--clump-p1", str(p1), "--clump-r2", str(r2), "--clump-kb", str(kb),
                "--threads", str(threads), "--out", clump_pref
            ])

            # Extract leads & weights
            leads = f"{base}/clump/trait{trait}_leads{suffix}.txt"
            pvalue = f"{base}/clump/trait{trait}_pvalue{suffix}.txt"
            weights = f"{base}/clump/trait{trait}_weights{suffix}.txt"

            with open(leads, "w") as fout, open(f"{clump_pref}.clumps") as fin:
                for line in fin:
                    if line.startswith(" ") or line.startswith("CHR") or line.strip() == "":
                        continue
                    fout.write(line.split()[2] + "\n")

            with open(pvalue, "w") as fout, open(f"{clump_pref}.clumps") as fin:
                for line in fin:
                    if line.startswith(" ") or line.startswith("CHR") or line.strip() == "":
                        continue
                    toks = line.split()
                    fout.write(f"{toks[2]}\t{toks[3]}\n")

            gwas_file = f"{gwas_pref}.PHENO1.glm.linear"
            keep_set = set()
            with open(leads) as f:
                for line in f:
                    keep_set.add(line.strip())

            with open(weights, "w") as fout, open(gwas_file) as fin:
                for line in fin:
                    if line.startswith("#") or line.startswith("ID"):
                        continue
                    toks = line.split()
                    if toks[2] in keep_set:
                        fout.write(f"{toks[2]}\t{toks[6]}\t{toks[11]}\n")

            # PRS Calculation
            run([
                "plink2", "--bfile", bfile_path, "--extract", leads,
                "--score", weights, "1", "2", "3",
                "--q-score-range", range_list_path, pvalue,
                "--read-freq", freq_file,
                "--threads", str(threads),
                "--out", prs_pref
            ])

            print(f"fold{fold} completed, PRS output: {prs_pref}.sscore")
            os.remove(f"{gwas_pref}.PHENO1.glm.linear")

        Parallel(n_jobs=5)(
            delayed(process_fold)(f) for f in folds
        )
```

###  Validation

```python
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from pathlib import Path
import time

def pearson_r2(y_true, y_pred):
    r = np.corrcoef(y_true, y_pred)[0, 1]
    return r ** 2

def validate_best_prs_full_grid_0123(
    seed,
    trait,
    folds=range(1,6),
    labels=None,
    n_jobs=-1
):
    if labels is None:
        labels = ["T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8"]

    outdir = f"prediction/random_{seed}"
    covar_path = "cov/cov.txt"
    pheno_dir = "phenotype/"

    threshold_dir = Path(outdir) / "threshold"
    threshold_dir.mkdir(parents=True, exist_ok=True)

    pheno = pd.read_csv(Path(pheno_dir) / f"{trait}_std.txt", sep=r"\s+", names=["FID", "IID", "PHENO"])
    covar = pd.read_csv(covar_path, sep=r"\s+", names=["FID", "IID", "COV1", "COV2", "COV3", "COV4"])

    error_log = Path(outdir) / "vali_error.log"
    now = time.strftime("%Y-%m-%d %H:%M:%S")

    folds_data = {}
    missing_flag = False
    missing_records = []

    # Check and load data
    for f in folds:
        tr_ids = pd.read_csv(f"{outdir}/fold{f}/train.txt", sep=r"\s+", names=["FID", "IID"])
        vd_ids = pd.read_csv(f"{outdir}/fold{f}/validation.txt", sep=r"\s+", names=["FID", "IID"])
        tr0 = tr_ids.merge(pheno, on=["FID", "IID"]).merge(covar, on=["FID", "IID"])
        vd0 = vd_ids.merge(pheno, on=["FID", "IID"]).merge(covar, on=["FID", "IID"])

        folds_data[f] = {'tr0': tr0, 'vd0': vd0, 'prs': {}}

    # Model Evaluation
    tmp0 = []
    for f in folds:
        tr0, vd0 = folds_data[f]['tr0'], folds_data[f]['vd0']
        X_tr, y_tr = tr0.drop(columns=["FID", "IID", "PHENO"]), tr0["PHENO"]
        X_vd, y_vd = vd0.drop(columns=["FID", "IID", "PHENO"]), vd0["PHENO"]
        r2_0 = pearson_r2(y_vd, LinearRegression().fit(X_tr, y_tr).predict(X_vd))
        tmp0.append(r2_0)

    r2_0 = np.nanmean(tmp0)
    print(f"Model0 (covariates only) average R²: {r2_0:.4f}")
```

### Test

The following Python function `run_cpt_test_from_threshold_file` performs the test for the prediction model using the best thresholds stored in the threshold file.

```python
import os
import subprocess
import pandas as pd

def run_cpt_test_from_threshold_file(
    seed,
    trait,
    r2=0.2,
    kb=50,
    p1=1,
    threads=1,
    covar="cov/cov.txt",
    pheno_dir="phenotype/"
):
    outdir = f"prediction/random_{seed}"
    train_file = f"prediction/train_{seed}.txt"
    test_dir = os.path.join(outdir, "test")
    modify_dir = os.path.join(test_dir, "modify")
    os.makedirs(modify_dir, exist_ok=True)

    relax_records = []  # Used to record relaxation situations

    # Read the best thresholds
    threshold_path = f"{outdir}/threshold/{trait}.tsv"
    df = pd.read_csv(threshold_path, sep="\t")
    row = df.iloc[0]

    best_thresholds = {
        (1, 1): row['model1'],
        (2, 1): row['model2'].split(",")[0],
        (2, 2): row['model2'].split(",")[1],
        (3, 1): row['model3'].split(",")[0],
        (3, 2): row['model3'].split(",")[1],
        (3, 3): row['model3'].split(",")[2]
    }

    T_to_P = {
        "T1": 0.90,
        "T2": 0.50,
        "T3": 0.30,
        "T4": 0.20,
        "T5": 0.10,
        "T6": 0.05,
        "T7": 0.01,
        "T8": 0.001
    }
    # Order from strict to relaxed
    T_order = ["T8", "T7", "T6", "T5", "T4", "T3", "T2", "T1"]

    bfile_map = {
        1: "prediction/bfile/sindel_pruned",
        2: "prediction/bfile/SV_pruned",
        3: "prediction/bfile/rSV_pruned"
    }

    pheno_file = os.path.join(pheno_dir, f"{trait}_std.txt")

    for model in range(4):
        for b in range(1, model+1):  # model0: none, model1: 1, model2: 1+2, model3: 1+2+3
            bfile_path = bfile_map[b]
            suffix = f"_m{model}_b{b}"
            gwas_pref = os.path.join(test_dir, f"gwas/trait{trait}{suffix}")
            clump_pref = os.path.join(test_dir, f"clump/trait{trait}_clump{suffix}")
            prs_pref = os.path.join(test_dir, f"prs/trait{trait}_PRS{suffix}")
            freq_pref = os.path.join(test_dir, f"freq/train_freq{suffix}")

            for d in ["gwas", "clump", "prs", "freq"]:
                os.makedirs(os.path.join(test_dir, d), exist_ok=True)

            def run(cmd):
                subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            freq_file = f"{freq_pref}.afreq"

            # GWAS
            run([
                "plink2", "--bfile", bfile_path, "--keep", train_file,
                "--pheno", pheno_file, "--covar", covar,
                "--glm", "hide-covar", "cols=+err",
                "--threads", str(threads),
                "--out", gwas_pref
            ])

            # Clump
            run([
                "plink2", "--bfile", bfile_path, "--keep", train_file,
                "--clump", f"{gwas_pref}.PHENO1.glm.linear",
                "--clump-p1", str(p1), "--clump-r2", str(r2), "--clump-kb", str(kb),
                "--threads", str(threads), "--out", clump_pref
            ])

            # Extract lead variants
            base_T_label = best_thresholds.get((model, b), "T1")  # Default is T1
            start_idx = T_order.index(base_T_label)

            selected_leads = []
            used_T_label = None

            for T_label in T_order[start_idx:]:  # Relax from the specified T to T1
                P_threshold = T_to_P[T_label]
                tmp_leads = []
                with open(f"{clump_pref}.clumps") as fin:
                    for line in fin:
                        toks = line.strip().split()
                        pval = toks[3]
                        if pval == "P":
                            continue
                        elif float(pval) <= P_threshold:
                            tmp_leads.append(toks[2])
                if tmp_leads:
                    selected_leads = tmp_leads
                    used_T_label = T_label
                    if used_T_label != base_T_label:
                        relax_records.append({
                            "trait": trait,
                            "model": model,
                            "bfile": b,
                            "original_T": base_T_label,
                            "relaxed_T": used_T_label,
                            "lead_count": len(selected_leads)
                        })
                    break

            if not selected_leads:
                print(f"Model{model} bfile{b} has no lead variant (even after relaxing to T1)")
            else:
                print(f"Model{model} bfile{b} using T={used_T_label} found {len(selected_leads)} lead variants")

            selected_file = f"{clump_pref}_leads_{used_T_label if used_T_label else base_T_label}.txt"
            with open(selected_file, "w") as fout:
                for snp in selected_leads:
                    fout.write(snp + "\n")

            gwas_file = f"{gwas_pref}.PHENO1.glm.linear"
            weights = f"{clump_pref}_weights.txt"
            keep_set = set(selected_leads)

            with open(weights, "w") as fout, open(gwas_file) as fin:
                for line in fin:
                    if line.startswith("#") or line.startswith("ID"):
                        continue
                    toks = line.strip().split()
                    if toks[2] in keep_set:
                        fout.write(f"{toks[2]}\t{toks[6]}\t{toks[11]}\n")

            # PRS Calculation
            if selected_leads:
                run([
                    "plink2", "--bfile", bfile_path, "--extract", selected_file,
                    "--score", weights, "1", "2", "3",
                    "--read-freq", freq_file,
                    "--threads", str(threads),
                    "--out", prs_pref
                ])
                print(f"Model{model} bfile{b} PRS output: {prs_pref}.sscore")
            else:
                print(f"Model{model} bfile{b} has no PRS output")

            os.remove(f"{gwas_pref}.PHENO1.glm.linear")

    # Write relaxation records
    if relax_records:
        relax_df = pd.DataFrame(relax_records)
        relax_df.to_csv(os.path.join(modify_dir, f"{trait}.tsv"), sep="\t", index=False)
        print(f"Relaxation records written to: {modify_dir}/{trait}.tsv")
````

- Evaluate Test R²

```python
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from pathlib import Path

def pearson_r2(y_true, y_pred):
    r = np.corrcoef(y_true, y_pred)[0, 1]
    return r ** 2

def evaluate_test_r2(
    seed,
    trait,
    covar_path="cov/cov.txt",
    pheno_dir="phenotype/"
):
    outdir = f"prediction/random_{seed}"
    test_dir = os.path.join(outdir, "test")

    pheno = pd.read_csv(Path(pheno_dir) / f"{trait}_std.txt", sep=r"\s+", names=["FID", "IID", "PHENO"])
    covar = pd.read_csv(covar_path, sep=r"\s+", names=["FID", "IID", "COV1", "COV2", "COV3", "COV4"])
    train_ids = pd.read_csv(f"prediction/train_{seed}.txt", sep=r"\s+", names=["FID", "IID"])

    # Identify test samples
    all_ids = set(pheno["FID"])
    train_set = set(train_ids["FID"])
    test_ids = pd.DataFrame(list(all_ids - train_set), columns=["FID"])
    test_ids["IID"] = test_ids["FID"]

    test0 = test_ids.merge(pheno, on=["FID", "IID"]).merge(covar, on=["FID", "IID"])
    X0 = test0.drop(columns=["FID", "IID", "PHENO"])
    y0 = test0["PHENO"]

    results = {}

    # Model 0: Covariates only
    lr0 = LinearRegression().fit(X0, y0)
    y_pred = lr0.predict(X0)
    r2_0 = pearson_r2(y0, y_pred)
    results["model0"] = r2_0

    for model in range(1, 4):
        df = test0.copy()
        for b in range(1, model + 1):
            suffix = f"_m{model}_b{b}"
            prs_path = f"{test_dir}/prs/trait{trait}_PRS{suffix}.sscore"

            prs = pd.read_csv(
                prs_path, delim_whitespace=True, comment=None
            )
            if prs.columns[0].startswith("#"):
                prs.columns = prs.columns.str.replace("#", "")
            prs = prs.rename(columns={"SCORE1_AVG": f"PRS{b}"})[["FID", "IID", f"PRS{b}"]]

            df = df.merge(prs, on=["FID", "IID"])

        X = df.drop(columns=["FID", "IID", "PHENO"])
        y = df["PHENO"]

        lr = LinearRegression().fit(X, y)
        y_pred = lr.predict(X)
        r2 = pearson_r2(y, y_pred)
        results[f"model{model}"] = r2

    return results
```

This script evaluates test results by calculating the R² for different models and saves the predictions.


## 2. Elastic-Net

### Sample Grouping: 
10% for testing, 90% for training and validation.

```r
#!/usr/bin/env Rscript

# 1) Randomly select 10% of the samples for test.txt
# 2) Generate train/valid in results/{SNP,SV,rSV}/fold*/
# 3) Copy the same train/valid in retrain/{SNP,SV,rSV}/fold*/

suppressPackageStartupMessages(library(data.table))

## ---------- Read Parameters ----------
args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 1) stop("Usage: Rscript 01_init.R <samples.txt>")
samples_file <- args[1]

set.seed(42)

## ---------- Read Samples ----------
samples   <- readLines(samples_file)
n_total   <- length(samples)

## ---------- Select 10% for Test ----------
n_test    <- floor(0.10 * n_total)
test_idx  <- sample(seq_len(n_total), n_test)
writeLines(samples[test_idx], "test.txt")

## ---------- 5-Fold Cross Validation ----------
trainval_ids <- samples[-test_idx]
fold_flag    <- sample(rep(1:5, length.out = length(trainval_ids)))
tv_dt        <- data.table(SampleID = trainval_ids, Fold = fold_flag)

types  <- c("SNP", "SV", "rSV")
roots  <- c(results = "results", retrain = "retrain")

for (fold in 1:5) {
  valid_vec <- tv_dt[Fold == fold, SampleID]
  train_vec <- setdiff(trainval_ids, valid_vec)

  for (root in roots) {
    for (tp in types) {
      fold_dir <- file.path(root, tp, sprintf("fold%d", fold))
      dir.create(fold_dir, recursive = TRUE, showWarnings = FALSE)
      writeLines(train_vec,  file.path(fold_dir, "train.txt"))
      writeLines(valid_vec,  file.path(fold_dir, "valid.txt"))
    }
  }
}
```

### First Run of PRS

```bash
#!/usr/bin/env bash

BFILE="$1"           
OUTDIR="$2"          
FOLD="$3"
TRAIT="$4"

PHENO_DIR="/path/to/phenotype/dir"
COV="/path/to/covariates/file"
THREADS=2

TRAIN="${OUTDIR}/fold${FOLD}/train.txt"
VALID="${OUTDIR}/fold${FOLD}/valid.txt"

mkdir -p "${OUTDIR}/fold${FOLD}/elastic" "${OUTDIR}/fold${FOLD}/scores"

# 1) Elastic-Net Fitting
ldak6 \
  --allow-multi YES \
  --elastic "${OUTDIR}/fold${FOLD}/elastic/${TRAIT}" \
  --bfile "${BFILE}" \
  --pheno "${PHENO_DIR}/${TRAIT}.phen" \
  --keep "$TRAIN" \
  --max-threads "${THREADS}" \
  --covar "${COV}" \
  --skip-cv YES \
  --LOCO NO 

# 2) PRS Scoring
ldak6 \
  --allow-multi YES \
  --calc-scores "${OUTDIR}/fold${FOLD}/scores/${TRAIT}" \
  --scorefile  "${OUTDIR}/fold${FOLD}/elastic/${TRAIT}.effects" \
  --bfile      "${BFILE}" \
  --covar      "${COV}" \
  --coeffsfile "${OUTDIR}/fold${FOLD}/elastic/${TRAIT}.coeff" \
  --save-counts YES \
  --power      0
```

### Thresholding

```python
#!/usr/bin/env python3
# 03_thresholding.py   (write per-trait temp files, final concat)
# --------------------------------------------------------------
import sys, itertools, numpy as np, pandas as pd
from pathlib import Path
from joblib import Parallel, delayed
from sklearn.linear_model import LinearRegression
from tqdm import tqdm

# ---------- Configuration ----------
ROOT_RESULTS = Path("results")
PHENO_DIR    = Path("/path/to/phenotype/dir")
COV_FILE     = Path("cov.txt")
FOLDS        = range(1, 6)
THR_DIR      = Path("threshold")               # Output directory for summary files
TMP_DIR      = THR_DIR / "tmp"                 # Temporary files for each trait
OUT_FILE     = THR_DIR / "traits_thres.tsv"
N_JOBS_M2    = 1
N_JOBS_M3    = 1
# -------------------------

# ---------- Tools ----------
def pearson_r2(y, yhat): return np.corrcoef(y, yhat)[0,1]**2
read_ids = lambda p: pd.read_csv(p, sep=r"\s+", header=None, names=["FID","IID"])
def read_cov():
    df = pd.read_csv(COV_FILE, sep=r"\s+", header=None)
    df.columns = ["FID","IID"] + [f"COV{i}" for i in range(1, df.shape[1]-1)]
    return df
COV_DF  = read_cov(); COV_COL = COV_DF.columns[2:]

def read_prof(path:Path, prefix:str):
    df = pd.read_csv(path, sep=r"\s+")
    if {"ID1","ID2"}.issubset(df.columns): df.rename(columns={"ID1":"FID","ID2":"IID"}, inplace=True)
    df.rename(columns={c:f"{prefix}_{c.split('_')[1]}" for c in df if c.startswith("Profile_")}, inplace=True)
    return df[["FID","IID",* [f"{prefix}_{k}" for k in range(1,11)]]]

def load_fold(trait, fold):
    ids_dir = ROOT_RESULTS/"SNP"/f"fold{fold}"
    tr_ids, vd_ids = read_ids(ids_dir/"train.txt"), read_ids(ids_dir/"valid.txt")
    pheno = pd.read_csv(PHENO_DIR/f"{trait}.phen", sep=r"\s+", header=None,
                        names=["FID","IID","PHENO"])
    prs1 = read_prof(ROOT_RESULTS/"SNP"/f"fold{fold}/scores/{trait}.profile","PRS1")
    prs2 = read_prof(ROOT_RESULTS/"SV" /f"fold{fold}/scores/{trait}.profile","PRS2")
    prs3 = read_prof(ROOT_RESULTS/"rSV"/f"fold{fold}/scores/{trait}.profile","PRS3")
    def assemble(ids):
        df = ids.merge(pheno).merge(COV_DF)
        for p in (prs1,prs2,prs3): df = df.merge(p,on=["FID","IID"])
        return df
    return assemble(tr_ids), assemble(vd_ids)

ALL_K = range(1,11)
GRID2 = list(itertools.product(ALL_K, ALL_K))
GRID3 = list(itertools.product(ALL_K, ALL_K, ALL_K))

# ---------- Threshold Search ----------
def best_k1(folds):
    r2=np.zeros(10)
    for tr,vd in folds:
        for k in ALL_K:
            cols=[f"PRS1_{k}",*COV_COL]
            r2[k-1]+=pearson_r2(vd["PHENO"],
                                LinearRegression().fit(tr[cols],tr["PHENO"]).predict(vd[cols]))
    return int(r2.argmax()+1)

def best_k12(folds):
    def score(pair):
        acc=[]
        for tr,vd in folds:
            cols=[f"PRS1_{pair[0]}",f"PRS2_{pair[1]}",*COV_COL]
            acc.append(pearson_r2(vd["PHENO"],
                     LinearRegression().fit(tr[cols],tr["PHENO"]).predict(vd[cols])))
        return pair, np.mean(acc)
    return max(Parallel(n_jobs=N_JOBS_M2)(delayed(score)(p) for p in GRID2),
               key=lambda x:x[1])[0]

def best_k123(folds):
    def score(triple):
        acc=[]
        for tr,vd in folds:
            cols=[f"PRS1_{triple[0]}",f"PRS2_{triple[1]}",f"PRS3_{triple[2]}",*COV_COL]
            acc.append(pearson_r2(vd["PHENO"],
                     LinearRegression().fit(tr[cols],tr["PHENO"]).predict(vd[cols])))
        return triple, np.mean(acc)
    return max(Parallel(n_jobs=N_JOBS_M3)(delayed(score)(t) for t in GRID3),
               key=lambda x:x[1])[0]

# ---------- Single Trait ----------
def run_trait(trait:int):
    folds=[load_fold(trait,f) for f in FOLDS]
    k1          = best_k1(folds)
    (k1m2,k2)   = best_k12(folds)
    (a,b,c)     = best_k123(folds)
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    (TMP_DIR/f"{trait}.tsv").write_text(
        f"{trait}\t{k1}\t{k1m2},{k2}\t{a},{b},{c}\n"
    )
    print(f"[Trait {trait}] done ({k1}) ({k1m2},{k2}) ({a},{b},{c})", flush=True)

# ---------- Main Entry ----------
def main(start:int,end:int,out_jobs:int):
    traits=list(range(start,end+1))
    Parallel(n_jobs=out_jobs)(
        delayed(run_trait)(t) for t in tqdm(traits,desc="Traits")
    )
    # Combine all tmp/*.tsv
    rows=[]
    for p in TMP_DIR.glob("*.tsv"):
        trait=int(p.stem)    # File name -> trait id
        k1, m2, m3 = p.read_text().strip().split("\t")[1:]
        rows.append((trait,k1,m2,m3))
    df=pd.DataFrame(rows, columns=["trait","model1","model2","model3"])
    df.sort_values("trait",inplace=True)
    THR_DIR.mkdir(exist_ok=True, parents=True)
    df.to_csv(OUT_FILE, sep="\t", index=False)
    print(f"✓ thresholds written to {OUT_FILE}")
```

### Recalculate PRS

```bash
#!/usr/bin/env bash

BFILE="$1"           
OUTDIR="$2"          
FOLD="$3"
TRAIT="$4"

PHENO_DIR="/path/to/phenotype/dir"
COV="cov.txt"
THREADS=2

TRAIN="${OUTDIR}/fold${FOLD}/train.txt"
VALID="${OUTDIR}/fold${FOLD}/valid.txt"

TV="${OUTDIR}/fold${FOLD}/tv.txt"
if [ ! -f "$TV" ]; then
  cat "${TRAIN}" "${VALID}" | sort -u > "$TV"
fi

mkdir -p "${OUTDIR}/fold${FOLD}/elastic" "${OUTDIR}/fold${FOLD}/scores"

# 1) Elastic-Net Fitting
ldak6 \
  --allow-multi YES \
  --elastic "${OUTDIR}/fold${FOLD}/elastic/${TRAIT}" \
  --bfile "${BFILE}" \
  --pheno "${PHENO_DIR}/${TRAIT}.phen" \
  --keep "$TV" \
  --max-threads "${THREADS}" \
  --covar "${COV}" \
  --skip-cv YES \
  --LOCO NO 

# 2) PRS Scoring
ldak6 \
  --allow-multi YES \
  --calc-scores "${OUTDIR}/fold${FOLD}/scores/${TRAIT}" \
  --scorefile  "${OUTDIR}/fold${FOLD}/elastic/${TRAIT}.effects" \
  --bfile      "${BFILE}" \
  --covar      "${COV}" \
  --coeffsfile "${OUTDIR}/fold${FOLD}/elastic/${TRAIT}.coeff" \
  --save-counts YES \
  --power      0
```

### Evaluation

```python
#!/usr/bin/env python3
# 05_evaluation.py  ——  Evaluate retrained PRS on the test set, output ONE file
# ---------------------------------------------------------------
import sys, numpy as np, pandas as pd
from pathlib import Path
from joblib import Parallel, delayed
from sklearn.linear_model import LinearRegression
from tqdm import tqdm

# ========= Path Constants =========
ROOT_RETRAIN = Path("retrain")                # retrain/SNP/...
PHENO_DIR    = Path("/path/to/phenotype/dir")
COV_FILE     = Path("cov.txt")
THR_FILE     = Path("threshold/traits_thres.tsv")   # Threshold summary table
TEST_IDS     = pd.read_csv("test.txt", sep=r"\s+", header=None,
                           names=["FID","IID"])

OUT_FILE     = ROOT_RETRAIN / "test_results.tsv"

# ========= Covariates =========
COV_DF = pd.read_csv(COV_FILE, sep=r"\s+", header=None)
COV_DF.columns = ["FID","IID"] + [f"COV{i}" for i in range(1, COV_DF.shape[1]-1)]
COV_COLS = COV_DF.columns[2:]

# ========= Threshold Load =========
THR_DF = pd.read_csv(THR_FILE, sep="\t").set_index("trait")

# ========= Tools =========
def pearson_r2(y, yhat): return np.corrcoef(y, yhat)[0,1] ** 2

def read_profile(trait:int, vtype:str):
    prof = ROOT_RETRAIN / vtype / "fold1" / "scores" / f"{trait}.profile"
    df   = pd.read_csv(prof, sep=r"\s+")
    if {"ID1","ID2"}.issubset(df.columns):
        df.rename(columns={"ID1":"FID","ID2":"IID"}, inplace=True)
    prefix = {"SNP":"PRS1","SV":"PRS2","rSV":"PRS3"}[vtype]
    df.rename(columns={c:f"{prefix}_{c.split('_')[1]}" for c in df if c.startswith("Profile_")},
              inplace=True)
    return df[["FID","IID", *[f"{prefix}_{k}" for k in range(1,11)]]]

def build_test_df(trait:int):
    ph  = pd.read_csv(PHENO_DIR/f"{trait}.phen", sep=r"\s+", header=None,
                      names=["FID","IID","PHENO"])
    df = TEST_IDS.merge(ph).merge(COV_DF)
    for vt in ("SNP","SV","rSV"):
        df = df.merge(read_profile(trait, vt), on=["FID","IID"])
    return df

# ========= Single Trait =========
def evaluate(trait:int):
    if trait not in THR_DF.index:
        return None               # Skip if no threshold
    row   = THR_DF.loc[trait]
    k1          = int(row["model1"])
    k1m2,k2     = map(int, row["model2"].split(","))
    a,b,c       = map(int, row["model3"].split(","))

    df  = build_test_df(trait)
    y   = df["PHENO"].values
    Xc  = df[COV_COLS].values

    r0 = pearson_r2(y, LinearRegression().fit(Xc, y).predict(Xc))
    r1 = pearson_r2(y, LinearRegression().fit(np.c_[Xc, df[f"PRS1_{k1}"]], y)
                                               .predict(np.c_[Xc, df[f"PRS1_{k1}"]]))
    r2 = pearson_r2(y, LinearRegression().fit(np.c_[Xc, df[f"PRS1_{k1m2}"], df[f"PRS2_{k2}"]], y)
                                               .predict(np.c_[Xc, df[f"PRS1_{k1m2}"], df[f"PRS2_{k2}"]]))
    X3 = np.c_[Xc, df[f"PRS1_{a}"], df[f"PRS2_{b}"], df[f"PRS3_{c}"]]
    lr3= LinearRegression().fit(X3, y)
    r3 = pearson_r2(y, lr3.predict(X3))
    betas = lr3.coef_[-3:]   # sindel, SV, rSV

    return {"trait":trait, "model0":r0, "model1":r1, "model2":r2, "model3":r3,
            "b_sindel":betas[0], "b_SV":betas[1], "b_rSV":betas[2]}

# ========= Main Entry =========
def main(begin:int, end:int, n_jobs:int):
    tasks = range(begin, end+1)
    rows  = Parallel(n_jobs=n_jobs)(
        delayed(evaluate)(t) for t in tqdm(tasks, desc="evaluate")
    )
    rows  = [r for r in rows if r]          # Remove None values
    df    = pd.DataFrame(rows).sort_values("trait")
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_FILE, sep="\t", index=False)
    print(f"✓ results written to {OUT_FILE} ({len(df)} traits)")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python 05_evaluation.py <start_trait> <end_trait> [n_jobs]")
        sys.exit(1)
    s, e = map(int, sys.argv[1:3])
    nj   = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    main(s, e, nj)
```


