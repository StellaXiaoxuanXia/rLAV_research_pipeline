# LAVrefiner prediction pipeline

This repository contains code for evaluating gene-level prediction performance
from LAVrefiner-derived variants and external STR/VNTR callers.

The pipeline uses residualized molecular phenotypes as prediction targets,
fits penalized linear models from genotype features, and evaluates prediction
accuracy in held-out samples. Prediction performance is reported as test-set
`R2`, defined as the squared Pearson correlation between observed and predicted
residualized phenotypes.

## Overview

The workflow has five main steps:

1. Prepare genotype matrices in RDS format.
2. Generate repeated train-test splits.
3. Fit prediction models for each gene and feature set.
4. Run permutation analyses using permuted residualized phenotypes.
5. Summarize per-gene prediction accuracy across repeated splits.

## Inputs

The main required inputs are:

- A gene list.
- Residualized phenotype files, one file per gene.
- Repeated train-test split files.
- Genotype matrices converted to RDS format.

Each residualized phenotype file contains sample identifiers and the
residualized phenotype value. Samples with missing phenotype values or PLINK
missing phenotype code `-9` are excluded before model fitting.

## Genotype Preparation

`00_prepare_rds.R` converts genotype inputs into the RDS format used by the
prediction scripts.

Supported input types include:

- PLINK bfiles for biallelic variants.
- Copy-number matrices.
- STR/VNTR matrices from external callers.

For external STR/VNTR calls, allele-pair values are converted to diploid
reference-relative repeat dosage by summing the two allele-level copy-number
changes.

Example usage:

```bash
Rscript 00_prepare_rds.R --bfile input_prefix --out output_name
Rscript 00_prepare_rds.R --cn copy_number_matrix.tsv --out output_name
Rscript 00_prepare_rds.R --tr external_TR_matrix.tsv --out output_name
```

## Train-Test Splits

`00_init_splits.R` generates repeated random train-test splits. In the current
analysis, each repeat assigns 80% of samples to the training set and 20% to the
test set. The same split files are used across all model classes to enable
paired comparison of prediction accuracy.

## Prediction Models

For each gene, repeat, and feature set, the pipeline fits a LASSO model using
`bigstatsr::big_spLinReg` with `alpha = 1`. The regularization parameter is
selected by internal cross-validation within the training samples. The fitted
marker effects are then used to calculate a genetic prediction score for all
samples.

Prediction is evaluated only in held-out test samples. To place scores on the
phenotype scale, a linear model of residualized phenotype on genetic score is
fitted in the training samples and applied to the test samples. Prediction
accuracy is reported as squared Pearson correlation between observed and
predicted residualized phenotypes in the test set.

Sample matching is performed using `FID` and `IID` against each RDS object's
sample table before model fitting.

## Model Classes

STR models:

```text
M1  LAV biallelic variants
M2  nLAV + rLAV biallelic variants
M3  LAV copy number
M4  nLAV + rLAV biallelic variants + copy number
S1  STRling copy number
```

VNTR models:

```text
M1  LAV biallelic variants
M2  nLAV + rLAV biallelic variants
M3  LAV copy number
M4  nLAV + rLAV biallelic variants + copy number
S1  adVNTR copy number
S2  VNTRseek copy number
```

non-TR models:

```text
M1  LAV biallelic variants
M2  nLAV + rLAV biallelic variants
```

## Running the Pipeline

Master scripts run prediction models across a gene list:

```bash
Rscript 01_master_VNTR.raw.R gene_list.txt
Rscript 01_master_VNTR.QCed.R gene_list.txt

Rscript 02_master_STR.raw.R gene_list.txt
Rscript 02_master_STR.QCed.R gene_list.txt

Rscript 03_master_nonTR.R gene_list.txt
```

The raw and QCed scripts are used to compare external caller results before and
after caller-specific filtering. Internal LAVrefiner-derived models are run
under the same samples, split files, and model settings.

## Permutation Analysis

Permutation scripts use the same genotypes, train-test splits, and model
settings as the main analysis, but replace the observed residualized phenotype
with permuted residualized phenotypes.

Each permutation can be run separately:

```bash
Rscript 01_master_VNTR.raw.permutation.R gene_list.txt 1
Rscript 01_master_VNTR.QCed.permutation.R gene_list.txt 1

Rscript 02_master_STR.raw.permutation.R gene_list.txt 1
Rscript 02_master_STR.QCed.permutation.R gene_list.txt 1

Rscript 03_master_nonTR.permutation.R gene_list.txt 1
```

The second argument is the permutation ID.

## Outputs

For each gene, the prediction scripts write a table with one row per model and
one column per repeat:

```text
Model  Rep1  Rep2  ...  Rep10
M1     ...
M2     ...
```

Missing values are kept as `NA` and should not be treated as zero.

For downstream summaries, prediction accuracy can be averaged across
non-missing repeats within each gene and model, then reshaped to one row per
gene and one column per model.
