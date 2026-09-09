#!/usr/bin/env Rscript

# Generate repeated Monte Carlo train/test splits.
# Input samples can be provided as either one column (IID only) or two columns
# (FID and IID). Output files are written without headers for direct use by the
# prediction pipeline.

set.seed(42)

input_file <- "all.sample.ID.txt"
out_dir <- "splits_10rep"
n_reps <- 10
test_ratio <- 0.2

if (!dir.exists(out_dir)) {
  dir.create(out_dir, recursive = TRUE)
}

samples <- read.table(input_file, header = FALSE, stringsAsFactors = FALSE)

if (ncol(samples) == 1) {
  samples <- data.frame(
    FID = samples[[1]],
    IID = samples[[1]],
    stringsAsFactors = FALSE
  )
} else {
  samples <- samples[, 1:2]
  colnames(samples) <- c("FID", "IID")
}

n_samples <- nrow(samples)
n_test <- round(n_samples * test_ratio)
n_train <- n_samples - n_test

cat(sprintf("Total samples loaded: %d\n", n_samples))
cat(sprintf("Split per repeat: train=%d | test=%d\n", n_train, n_test))
cat(sprintf("Saving %d repeats to: %s/\n", n_reps, out_dir))

for (rep_id in seq_len(n_reps)) {
  test_indices <- sample(seq_len(n_samples), size = n_test, replace = FALSE)

  test_samples <- samples[test_indices, c("FID", "IID")]
  train_samples <- samples[-test_indices, c("FID", "IID")]

  write.table(
    test_samples,
    file = file.path(out_dir, sprintf("rep%d_test.txt", rep_id)),
    sep = "\t",
    row.names = FALSE,
    col.names = FALSE,
    quote = FALSE
  )

  write.table(
    train_samples,
    file = file.path(out_dir, sprintf("rep%d_train.txt", rep_id)),
    sep = "\t",
    row.names = FALSE,
    col.names = FALSE,
    quote = FALSE
  )

  if (rep_id == 1 || rep_id %% 10 == 0) {
    cat(sprintf("  - Repeat %d generated successfully.\n", rep_id))
  }
}

cat("All repeats completed.\n")
