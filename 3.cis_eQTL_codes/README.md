# cis-eQTL Code Availability Pipeline

This folder contains the scripts used for cis-eQTL QQ-plot preparation, eGene-level correction, and SuSiE fine-mapping/PIP extraction.

Before running, edit the hard-coded input and output paths in each Python script to match the local file system. The scripts assume TensorQTL-compatible PLINK genotype files, phenotype matrices, phenotype position files, and covariate matrices.

## Recommended running order

1. `01_permuted_nominal.py`  
   Runs 10 phenotype permutations for nominal cis-eQTL mapping. These outputs can be used for QQ-plot/null-distribution checks.

2. `02_nominal_cis_eqtl.py`  
   Runs the nominal cis-eQTL scan with TensorQTL and saves chromosome-level nominal association files.

3. `03_egene_correction.py`  
   Runs permutation-based eGene discovery, calculates q-values, and extracts corrected significant variant-gene pairs using gene-specific nominal thresholds.

4. `04_susie_full_pip.py`  
   Runs TensorQTL SuSiE fine-mapping by phenotype chunks and exports both SuSiE summaries and full PIP tables for retained phenotypes.

## SLURM wrappers

Each Python script has a matching `.sbatch` file:

- `01_permuted_nominal.sbatch`
- `02_nominal_cis_eqtl.sbatch`
- `03_egene_correction.sbatch`
- `04_susie_full_pip.sbatch`

Submit them in order, for example:

```bash
sbatch 01_permuted_nominal.sbatch
sbatch 02_nominal_cis_eqtl.sbatch
sbatch 03_egene_correction.sbatch
sbatch 04_susie_full_pip.sbatch
```

The SLURM settings are templates and may need to be adjusted for the available cluster partition, CPU allocation, and environment.
