# eQTL Analysis

## 1. GWAS and Threshold Filtering

Demo: Use `plink2` to filter variants with p < 1/N. The bfile used is `SV_QCed` and `rSV_prune`.

```bash
plink2 \
    --bfile "${bfile}" \
    --pheno "${pheno}" \
    --covar "${covar}" \
    --glm hide-covar \
    --out "${tmp_prefix}"
````

## 2. Clumping and LD Filtering for eQTL

Clump all significant variants within 50kb of each other. After clumping, remove clusters with fewer than two variants. The variant with the lowest p-value in each cluster is considered the lead variant. Extract all lead variants for the same trait and calculate r². If the r² between two lead variants is greater than 0.2, retain only the cluster with the smallest p-value. The final eQTL results will be obtained.

## 3. Hotspot Analysis

Extract the gene locations corresponding to all lead variants in the eQTL and perform hotspot analysis using the Hotscan software. This will identify hotspot locations.

* [Hotscan GitHub](https://github.com/itojal/hot_scan)

## 4. GO Enrichment Analysis

Use the `clusterProfiler` R package (v.3.10.1) to perform GO enrichment annotation.

### Example for Human:

```r
library(stringr)
library(clusterProfiler)
library(org.Hs.eg.db)

ego <- enrichGO(
    gene           = <gene_list>,
    OrgDb          = org.Hs.eg.db,
    keyType        = "ENSEMBL",
    ont            = "ALL",
    pAdjustMethod  = "BH",
    pvalueCutoff   = 0.01,
    qvalueCutoff   = 0.01,
    readable       = FALSE
)
```
