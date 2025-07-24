# Hsq Estimation

## Covariance of 4 PCs Creation

Prune the SNP/indel (sindel) variants:

```bash
plink --bfile sindel_QCed --indep-pairwise 100 1 0.025 --out sindel_ld
plink --bfile sindel_QCed --extract sindel_ld.prune.in --make-bed --out sindel_pruned
plink --bfile sindel_pruned --pca 4 --out cov
````

## 1. GCTA

Website: [GCTA Software](https://yanglab.westlake.edu.cn/software/gcta/)

After pruning other bfiles, generate GRMs:

```bash
gcta64 --bfile test1 --make-grm --out test1_grm
gcta64 --bfile test2 --make-grm --out test2_grm
gcta64 --bfile test3 --make-grm --out test3_grm
```

Prepare the MGRM file:

* test1\_grm
* test2\_grm
* test3\_grm

Perform Hsq Estimation:

```bash
gcta64 --bfile test_all --mgrm mgrm.txt --reml --pheno pheno.phen --qcovar cov.txt --out hsq_results
```

## 2. LDAK

Website: [LDAK Software](https://dougspeed.com/)

Directly operate on all QCed files. Below is a demo:

### Make GRM

```bash
bfile=SV
model=ldak
mkdir -p hsq/ldak/${bfile}
dir=hsq/ldak/${bfile}
for j in {1..22}; do
    ldak6 --bfile ${bfile}_QC/${bfile}_QCed \
        --cut-weights ${dir}/sections$j \
        --chr $j \
        --allow-multi YES
    ldak6 --calc-weights-all ${dir}/sections$j \
        --bfile ${bfile}_QC/${bfile}_QCed \
        --chr $j \
        --allow-multi YES
done
cat ${dir}/sections{1..22}/weights.short > ${dir}/weights.short
```

Calculate Kinship:

```bash
ldak6 --bfile ${bfile}_QC/${bfile}_QCed \
    --power -0.25 \
    --weights hsq/${model}/${bfile}/weights.short \
    --allow-multi YES \
    --calc-kins-direct hsq/${model}/${bfile}
```

### Hsq Estimation

```bash
ldak6 --bfile test_all --mgrm mgrm.txt --reml hsq_results --pheno pheno.phen --covar cov.txt --constrain YES
```

# 3. Group Research Pipeline

This repository contains the code for conducting group research and processing genetic data related to structural variants (SVs) and refined SVs (rSVs). It includes scripts for selecting and preparing the data, generating genetic relationship matrices (GRMs), estimating heritability, and visualizing results.

## Group Research Sampling

This section gathers filenames that match the criteria and groups them based on chromosomes.

### Step 1: Collect Data
```python
import pandas as pd
import os
from tqdm import tqdm
import random
from collections import defaultdict

target_folder = "rSV_call/matrix_results"

all_groups = []

for filename in os.listdir(target_folder):
    if filename.endswith("D_matrix.csv"):
        parts = filename.split("_")
        if len(parts) >= 3:
            chrom = parts[1]
            group_number = parts[2]
            all_groups.append({
                "filename": filename,
                "chrom": chrom,
                "group_number": group_number
            })

random.seed(42)
selected_groups = random.sample(all_groups, min(100, len(all_groups)))

chrom_group_dict = defaultdict(list)
for group in selected_groups:
    chrom_group_dict[group["chrom"]].append(group["filename"])

with open("group_research/groups.txt", "w") as f:
    for item in selected_groups:
        parts = item["filename"].split("_")
        chrom = parts[1]
        group_number = parts[2]
        pos = parts[3]
        group_name = f"Group_{chrom}_{group_number}_{pos}"
        f.write(group_name + "\n")
````

### Step 2: Data Preparation with PLINK


```bash
# oSV and rSV processing with PLINK
dir=group_research
mkdir -p ${dir}
bfile=oSV
plink --vcf rSV_call/${bfile}.vcf --keep-allele-order --make-bed --out ${dir}/${bfile}
awk '{$2 = "SV_" NR; print}' OFS='\t' ${dir}/${bfile}.bim > tmp.bim && mv tmp.bim ${dir}/${bfile}.bim

bfile=rSV
plink --vcf rSV_call/${bfile}.vcf --keep-allele-order --make-bed --out ${dir}/${bfile}
```

### Step 3: Contract Groups and Generate GRMs

```python
def sv_id_contract(filename):
    parts = filename.split("_")
    chrom = parts[1]
    group_number = parts[2]
    pos = parts[3]
    d_matrix_path = f"rSV_call/matrix_results/{filename}"
    d_matrix = pd.read_csv(d_matrix_path, index_col=0)
    rSV_number = d_matrix.shape[1]
    SV_number = d_matrix.shape[0]
    rSV_name_list = [f"Group_{chrom}_{group_number}_{pos}_rSV_{i}" for i in range(1, rSV_number + 1)]
    bim_path = "group_research/oSV.bim"
    bim = pd.read_csv(bim_path, sep="\t", header=None)
    target_rows = bim[(bim[0].astype(str) == chrom) & (bim[3] == int(pos))]
    if target_rows.empty:
        print("No matching pos found in oSV.bim!")
    else:
        start_index = target_rows.index[0]
        sv_ids = bim.loc[start_index:start_index + SV_number - 1, 1].tolist()
    return chrom, group_number, pos
```

### Step 4: PLINK Command for Group Processing

```python
def plink_cmd_contract(filename):
    # Process the input file to get the GRM for each group
    ...
    # Run PLINK commands for both oSV and rSV datasets
    subprocess.run(plink_sv_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    subprocess.run(plink_rsv_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return group_dir
```

## 4. Heritability Estimation (GCTA)

### Parallel Heritability Estimation

```bash
# Parallelized heritability estimation using GCTA
phen_num=200
threads=1
export threads

run_gcta() {
    group="$1"
    bfile="$2"
    i="$3"
    out="group_research/${group}/${bfile}/hsq_results/pheno_${i}_${bfile}_group"
    mkdir -p "group_research/${group}/${bfile}/hsq_results"
    gcta64 --grm "group_research/${group}/${bfile}_group_grm" \
           --pheno "phenotype/${i}_std.txt" \
           --qcovar cov/cov.txt \
           --reml \
           --thread-num "$threads" \
           --out "$out" \
           > /dev/null 2>&1
}

for group in $(cat group_research/groups.txt); do
    echo "Processing ${group}..."
    for bfile in "rSV+oSV"; do
        rm -rf "group_research/${group}/${bfile}/hsq_results"
    done
    parallel --jobs 40 --tmpdir /home/s3020226030/tmp \
             run_gcta ::: "${group}" ::: "rSV+oSV" ::: $(seq 1 $phen_num)
done
```

## 5. Data Collection and Visualization

### Collecting Results for Visualization

```python
import pandas as pd

def collect_hsq_results(num_pheno=10):
    with open("group_research/groups.txt", "r") as f:
        for line in f:
            line = line.strip()  # Remove newline and spaces
            print(f"processing {line}")
            parts = line.split("_")
            chrom = parts[1]
            group_number = parts[2]
            pos = parts[3]
            hsq_dir_oSV = f"group_research/{line}/oSV/hsq_results"
            hsq_dir_merged = f"group_research/{line}/rSV+oSV/hsq_results"
            summary_data = []
            for i in range(1, num_pheno + 1):
                pheno_id = i
                hsq_oSV_file = os.path.join(hsq_dir_oSV, f"pheno_{i}_oSV_group.hsq")
                hsq_merged_file = os.path.join(hsq_dir_merged, f"pheno_{i}_rSV+oSV_group.hsq")
                oSV_hsq = extract_hsq_value(hsq_oSV_file)
                merged_hsq = extract_hsq_value(hsq_merged_file)
                summary_data.append({
                    "phenotype_index": pheno_id,
                    "oSV_hsq": oSV_hsq,
                    "rSV+oSV_hsq": merged_hsq
                })
            # Convert to DataFrame and save to file
            df = pd.DataFrame(summary_data)
            output_path = f"group_research/hsq_summary_{chrom}_{group_number}_{pos}.tsv"
            df.to_csv(output_path, sep="\t", index=False)

collect_hsq_results(num_pheno=200)
```

