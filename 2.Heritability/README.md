# GRM Construction, Permutation, and Simulation Pipeline

This repository contains the scripts used to build genetic relationship matrices (GRMs), run phenotype-permutation REML analyses, and perform causal-variant simulations for the heritability analyses.

The code is organized into three parts:

1. GRM construction
2. Synchronized phenotype permutation
3. Heritability simulation

Before running the scripts, check the path variables at the top of each file and adjust them to the local environment.

---

## Requirements

The scripts require:

```bash
python >= 3.8
plink
gcta64
```

Required Python packages:

```bash
pandas
numpy
psutil
```

Some scripts also use standard Python modules such as `os`, `subprocess`, `threading`, `logging`, `glob`, and `concurrent.futures`.

---

## Script Overview

| Script | Purpose |
|---|---|
| `00_GRM_TR.py` | Builds manual GRMs for rLAV, LAV, and rLAV+nLAV models. The variant category is controlled by `TR_TYPE`, such as `STR`, `VNTR`, or `nonTR`. |
| `00_GRM_3TRcaller_raw.py` | Builds GRMs from raw classical TR-caller outputs, including STR, adVNTR, and VNTRseek. |
| `00_GRM_CN.py` | Builds copy-number GRMs for LAV_STR and LAV_VNTR features. |
| `00_GRM_CN_rLAV_nLAV.py` | Builds joint GRMs by combining copy-number features with rLAV+nLAV dosage features. |
| `00_phenotype_permute.py` | Generates synchronized phenotype and covariate permutations. |
| `00_permutation_hsq_fixed.py` | Runs synchronized permutation REML analysis for the main GRM set. |
| `00_permutation_hsq_fixed_3TRcaller_raw.py` | Runs permutation REML analysis for the three raw TR-caller GRMs. |
| `00_permutation_hsq_fixed_CN_rLAV_nLAV.py` | Runs permutation REML analysis for the pruned joint CN+rLAV+nLAV GRMs. |
| `00_nonTR_simu_rLAV_causal.py` | Runs nonTR heritability simulations using sampled causal rLAV/nLAV features and precomputed GRMs. |
| `00_STR_simu_CN_rLAV_nLAV_causal.py` | Runs STR mixed-causal simulations using CN and pruned rLAV+nLAV features. |
| `00_VNTR_simu_CN_rLAV_nLAV_causal.py` | Runs VNTR mixed-causal simulations using CN and pruned rLAV+nLAV features. |

---

## Recommended Running Order

### 1. Build GRMs

Run the GRM construction scripts first:

```bash
python 00_GRM_TR.py
python 00_GRM_3TRcaller_raw.py
python 00_GRM_CN.py
python 00_GRM_CN_rLAV_nLAV.py
```

For `00_GRM_TR.py`, edit `TR_TYPE` before running if multiple variant categories are needed:

```python
TR_TYPE = "STR"
TR_TYPE = "VNTR"
TR_TYPE = "nonTR"
```

Each GRM is saved in GCTA binary GRM format:

```text
*.grm.id
*.grm.bin
*.grm.N.bin
```

---

### 2. Generate synchronized permutations

Run:

```bash
python 00_phenotype_permute.py
```

This script applies the same sample permutation to phenotype and covariate matrices. The resulting paired permutation files are used by the downstream REML permutation scripts.

---

### 3. Run permutation REML analyses

Run the relevant permutation scripts:

```bash
python 00_permutation_hsq_fixed.py
python 00_permutation_hsq_fixed_3TRcaller_raw.py
python 00_permutation_hsq_fixed_CN_rLAV_nLAV.py
```

These scripts estimate heritability after synchronized phenotype permutation and are used to evaluate empirical null behavior and Type I error.

The scripts include checkpoint-aware resume support, so interrupted runs can usually continue from previously completed phenotypes.

---

### 4. Run simulation analyses

Run the simulation scripts as needed:

```bash
python 00_nonTR_simu_rLAV_causal.py
python 00_STR_simu_CN_rLAV_nLAV_causal.py
python 00_VNTR_simu_CN_rLAV_nLAV_causal.py
```

The simulation scripts generate phenotypes under specified heritability levels and compare REML estimates from oracle and prebuilt GRMs.

---

## Notes

- Most scripts use precomputed GRMs, so the GRM construction step should be completed first.
- The simulation scripts use a fixed random seed for reproducibility.
- Temporary files are generated during PLINK and GCTA runs and are removed automatically where applicable.
- For cluster runs, adjust thread settings such as `WORKER_THREADS`, `GCTA_THREADS`, or `THREADS` according to available resources.
