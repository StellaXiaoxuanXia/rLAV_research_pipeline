# Variant Call Pipeline

This pipeline describes the steps to call variants (SNPs, indels, and structural variants) using various tools. It includes mapping reads to the pangenome, calling SNPs and indels with DeepVariant, filtering, and performing structural variant (SV) calls using Paragraph. 

## 1. Map to Pangenome

First, align the sample reads to the pangenome reference:

```bash
FQ_DIR="/path/to/WGS"
REF_DIR="/path/to/pangenome/dir"

vg giraffe -p -t ${SLURM_CPUS_PER_TASK} \
    --sample <sample_ID> \
    -m $REF_DIR/<pangenome>.k39.w15.N16.min \
    -d $REF_DIR/<pangenome>.dist \
    -H $REF_DIR/<pangenome>.N16.gbwt \
    -x $REF_DIR/<pangenome>.xg \
    -N <sample_ID> \
    -f $FQ_DIR/<FASTQ_1> \
    -f $FQ_DIR/<FASTQ_2> \
    -o bam > <sample_ID>.bam
````

After generating the BAM file, sort it using `samtools`:

```bash
samtools sort <sample_ID>.bam -o <sample_ID>_sorted.bam
```

## 2. SNP and Indel Calling

### Using DeepVariant

To call SNPs and indels, the pipeline utilizes DeepVariant:

```bash
#!/bin/bash

ID=$1    # Sample ID

BAM_DIR="/path/to/BAM_files"
REF_DIR="/path/to/pangenome"
OUTPUT_DIR="/path/to/output"
SIF_IMAGE="/path/to/deepvariant/image"

mkdir -p ${OUTPUT_DIR}

singularity exec -B ${BAM_DIR}:/bam_input \
                 -B ${REF_DIR}:/ref_input \
                 -B ${OUTPUT_DIR}:/output \
                 $SIF_IMAGE \
    /opt/deepvariant/bin/run_deepvariant \
    --model_type=WGS \
    --ref=/ref_input/<pangenome>.fasta \
    --reads=/bam_input/<BAM_file>.bam \
    --output_vcf=/output/${ID}.vcf.gz \
    --output_gvcf=/output/${ID}.gvcf.gz \
    --intermediate_results_dir=/output/deepvariant_tmp_output/$ID \
    --vcf_stats_report=false \
    --sample_name $ID \
    --num_shards=16
```

After calling variants for each sample, merge the VCF files:

```bash
bcftools merge -l <file_list> -Oz -o <output>.vcf.gz
```

Apply depth filtering and quality checks using WGS:

```bash
WGS --model vcf --type depthFilterDP --minDepth 810 --maxDepth 3124 --file 01_raw.vcf.gz --out 02_depth.vcf
bgzip 02_depth.vcf

WGS --model vcf --type qualityFilter --threshold 20 --file 02_depth.vcf.gz --out 03_quality.vcf
bgzip 03_quality.vcf

WGS --model vcf --type inDel_len --file 03_quality.vcf.gz --out 04_indel_size.vcf
bgzip 04_indel_size.vcf

plink --vcf 04_indel_size.vcf.gz --biallelic-only --make-bed --vcf-half-call --out 05_plink
```

## 3. Structural Variant (SV) Calling

Structural variants (SVs) are called using Paragraph, a tool designed for SV genotyping:


Genotypes of SVs were called by Paragraph using default parameters (https://github.com/Illumina/paragraph). Here is an example:
```bash
python3 bin/multigrmpy.py -i <pangenome_candidate>.vcf \
                          -m samples.txt \
                          -r <pangenome>.fa \
                          -o test
```

The above script processes the candidate SVs and outputs the results for further analysis.
