#!/bin/bash
#SBATCH -J eGene
#SBATCH -n 1
#SBATCH -N 1
#SBATCH -p cps4
#SBATCH -c 42
#SBATCH -o eGene.out
#SBATCH -e eGene.err

set -euo pipefail

START_TIME=$(date "+%Y-%m-%d %H:%M:%S")
START_TIMESTAMP=$(date +%s)
echo "=========================================================="
echo "Job started: ${START_TIME}"
echo "=========================================================="

python 03_egene_correction.py

END_TIME=$(date "+%Y-%m-%d %H:%M:%S")
END_TIMESTAMP=$(date +%s)
DURATION=$((END_TIMESTAMP - START_TIMESTAMP))
HOURS=$((DURATION / 3600))
MINUTES=$(((DURATION % 3600) / 60))
SECONDS=$((DURATION % 60))

echo "=========================================================="
echo "Job finished: ${END_TIME}"
echo "Total run time: ${HOURS} h ${MINUTES} min ${SECONDS} s"
echo "=========================================================="
