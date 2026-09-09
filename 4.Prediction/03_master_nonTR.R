library(parallel)

num_cores <- 50
rscript <- "/fs2/software/bioinfo/miniforge3/envs/r_v4.5/bin/Rscript"

args <- commandArgs(trailingOnly = TRUE)
gene_list <- if (length(args) >= 1) args[1] else "gene_list.txt"

message("Using gene list: ", gene_list)

if (file.exists(gene_list)) {
  genes <- readLines(gene_list)
  genes <- trimws(genes[genes != ""])
} else {
  stop("Gene list was not found: ", gene_list)
}

message("Genes loaded: ", length(genes))

Sys.setenv(
  OMP_NUM_THREADS = "1",
  MKL_NUM_THREADS = "1",
  OPENBLAS_NUM_THREADS = "1",
  VECLIB_MAXIMUM_THREADS = "1",
  NUMEXPR_NUM_THREADS = "1"
)

bigparallelr::set_blas_ncores(1)

results <- mclapply(genes, function(gene) {
  message(sprintf("Processing nonTR: %s", gene))

  cmd <- sprintf(
    "%s prediction_pipeline.nonTR.wg.final.R --split_dir splits_10rep --n_reps 10 --trait %s --out_dir results/nonTR/10rep",
    shQuote(rscript),
    shQuote(gene)
  )

  exit_code <- system(cmd)

  if (exit_code == 0) {
    return(paste("Finished", gene))
  } else {
    return(paste("Error in", gene))
  }
}, mc.cores = num_cores)

print(unlist(results))
