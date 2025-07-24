# QC

Remove low-quality variants with MAF < 0.01 and call rate < 0.8 using plink:
```bash
plink --bfile test --keep-allele-order --maf 0.01 --geno 0.2 --make-bed --out test_QCed
````

# rSV Call

Perform rSV calling on the `SV_QCed` file, followed by QC and pruning for the rSV calls.

## Step 1: Convert bfile to vcf

```bash
plink --bfile SV_QCed --keep-allele-order --recode-vcf --out SV
```

## Step 2: Compress and index the vcf

```bash
bgzip SV.vcf
bcftools index SV.vcf.gz
```

## Step 3: Call rSV
See https://github.com/StellaXiaoxuanXia/SVrefiner for details.
```bash
SVrefiner --vcf SV.vcf.gz --ref ref.fasta --threads 10 --out .
```

## Step 4: QC rSV

```bash
plink --vcf rSV.vcf --make-bed --maf 0.01 --geno 0.2 --keep-allele-order --out rSV_QCed
plink --bfile rSV_QCed --indep-pairwise 100 1 0.999 --out rSV_ld
plink --bfile rSV_QCed --extract rSV_ld.prune.in --make-bed --out rSV_temp
```

Finally, delete rSV variants with r² = 1 within a 50kb window, using the `oSV.vcf` file, to obtain `rSV_prune`.

```bash
# Custom pruning steps for rSVs based on r² = 1
```
