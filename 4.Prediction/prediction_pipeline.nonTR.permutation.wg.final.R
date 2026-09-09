library(optparse)
library(data.table)
library(bigstatsr)
library(bigsnpr)
library(bigparallelr)

Sys.setenv(OMP_NUM_THREADS = 1)
Sys.setenv(MKL_NUM_THREADS = 1)
Sys.setenv(OPENBLAS_NUM_THREADS = 1)
Sys.setenv(VECLIB_MAXIMUM_THREADS = 1)
Sys.setenv(NUMEXPR_NUM_THREADS = 1)

bigparallelr::set_blas_ncores(1)

ncores <- 1
cat(sprintf("[%s] Initializing... Using %d core for LASSO regression\n", Sys.time(), ncores))

option_list <- list(
  make_option(c("--trait"), type = "character", default = NULL),
  make_option(c("--out_dir"), type = "character", default = NULL),
  make_option(c("--work_dir"), type = "character",
              default = "/fs2/home/xiaxy/gzz/projects/2024/SVrefiner/human/prediction/bigstatsr2"),
  make_option(c("--perm_root"), type = "character",
              default = "/fs2/home/xiaxy/gzz/projects/2024/SVrefiner/human/phenotypes_residual_perm"),
  make_option(c("--perm_start"), type = "integer", default = 1),
  make_option(c("--n_perms"), type = "integer", default = 1),
  make_option(c("--split_dir"), type = "character", default = "splits_10rep"),
  make_option(c("--n_reps"), type = "integer", default = 10),
  make_option(c("--alphas"), type = "character", default = "1"),
  make_option(c("--nfolds_lasso"), type = "integer", default = 5)
)

rds_dir <- "/fs2/home/xiaxy/gzz/projects/2024/SVrefiner/human/prediction/bigstatsr2/rds"

lav_bi_rds <- file.path(rds_dir, "LAV.nonTR.rds")
nlav_rlav_bi_rds <- file.path(rds_dir, "nLAV+rLAV.nonTR.rds")

opt <- parse_args(OptionParser(option_list = option_list))
for (x in c("trait", "out_dir")) {
  if (is.null(opt[[x]])) stop("Missing --", x)
}

setwd(opt$work_dir)
trait <- opt$trait
alphas <- as.numeric(strsplit(opt$alphas, ",", fixed = TRUE)[[1]])

split_dir <- file.path(opt$work_dir, opt$split_dir)
out_root <- file.path(opt$work_dir, opt$out_dir)
dir.create(out_root, recursive = TRUE, showWarnings = FALSE)

load_obj <- function(path) {
  cat(sprintf("[%s] Loading RDS: %s\n", Sys.time(), path))
  if (!file.exists(path)) stop("Input file does not exist: ", path)
  obj <- tryCatch(
    snp_attach(path),
    error = function(e1) {
      tryCatch(
        readRDS(path),
        error = function(e2) {
          stop(
            "Failed to load input as bigSNP or ordinary RDS: ", path,
            "\n  snp_attach error: ", e1$message,
            "\n  readRDS error: ", e2$message,
            "\nCheck that this path points to a .rds file, not .bk/.bed/.bim/.fam or a text file."
          )
        }
      )
    }
  )
  if (is.null(obj$genotypes) || is.null(obj$fam) || is.null(obj$map)) {
    stop("RDS object must contain genotypes, fam, and map: ", path)
  }
  obj
}

as_iid <- function(fam) {
  if ("sample.ID" %in% names(fam)) return(as.character(fam$sample.ID))
  if ("IID" %in% names(fam)) return(as.character(fam$IID))
  stop("fam must contain sample.ID or IID")
}

as_fid <- function(fam) {
  if ("family.ID" %in% names(fam)) return(as.character(fam$family.ID))
  if ("FID" %in% names(fam)) return(as.character(fam$FID))
  as_iid(fam)
}

sample_key <- function(fid, iid) paste(fid, iid, sep = "\r")

fam_key <- function(fam) sample_key(as_fid(fam), as_iid(fam))

resolve_fit_cols <- function(fit_cols, cols) {
  if (is.null(fit_cols)) return(seq_along(cols))
  if (all(fit_cols %in% cols)) return(match(fit_cols, cols))
  if (length(fit_cols) > 0 && max(fit_cols) <= length(cols)) return(fit_cols)
  stop("Could not align big_spLinReg retained columns with input columns")
}

train_score <- function(fs, set_name, perm_idx, rep_id, all_data, train_data, y_train) {
  obj <- fs$obj
  cols <- fs$cols
  if (length(cols) < 2) {
    return(list(score = rep(0, nrow(all_data))))
  }

  keys <- fam_key(obj$fam)
  rows_all <- match(sample_key(all_data$FID, all_data$IID), keys)
  if (anyNA(rows_all)) stop("Some samples are missing from RDS object for ", set_name)
  ind_train <- match(sample_key(train_data$FID, train_data$IID), keys)
  if (anyNA(ind_train)) stop("Some training samples are missing from RDS object for ", set_name)

  G <- obj$genotypes
  p <- length(cols)
  set.seed(100000L * perm_idx + rep_id)

  fit <- tryCatch({
    big_spLinReg(
      X = G,
      y.train = y_train,
      ind.train = ind_train,
      ind.col = seq_len(p),
      alphas = alphas,
      K = opt$nfolds_lasso,
      ncores = ncores,
      warn = FALSE
    )
  }, error = function(e) {
    cat(sprintf("Perm %d | Rep %d | %s ERROR: %s\n", perm_idx, rep_id, set_name, e$message))
    NULL
  })

  if (is.null(fit)) {
    return(list(score = rep(0, nrow(all_data))))
  }

  sm <- summary(fit)
  best <- which.min(sm$validation_loss)
  fit_cols <- resolve_fit_cols(attr(fit, "ind.col"), cols)
  beta <- numeric(p)
  beta_fit <- sm$beta[[best]][seq_along(fit_cols)]
  beta_fit[!is.finite(beta_fit)] <- 0
  beta[fit_cols] <- beta_fit
  score <- as.numeric(G[rows_all, cols] %*% beta)
  score[!is.finite(score)] <- NA_real_

  list(score = score)
}

fit_r2 <- function(formula_text, train_data, test_data) {
  f <- as.formula(formula_text)
  vars <- all.vars(f)
  train_cc <- complete.cases(train_data[, ..vars])
  test_cc <- complete.cases(test_data[, ..vars])
  if (sum(train_cc) < 2 || sum(test_cc) < 2) {
    cat(sprintf("  Evaluation skipped: %s | train complete=%d | test complete=%d\n",
                formula_text, sum(train_cc), sum(test_cc)))
    return(NA_real_)
  }
  fit <- lm(f, data = train_data[train_cc])
  pred <- predict(fit, newdata = test_data[test_cc])
  cor(test_data$Pheno[test_cc], pred, use = "complete.obs")^2
}

lav_bi <- load_obj(lav_bi_rds)
nlav_rlav_bi <- load_obj(nlav_rlav_bi_rds)

fam_ref <- data.table(
  FID = as_fid(lav_bi$fam),
  IID = as_iid(lav_bi$fam),
  Index = seq_len(nrow(lav_bi$fam))
)

cols <- list(
  LAV_bi = seq_len(ncol(lav_bi$genotypes)),
  nLAV_rLAV_bi = seq_len(ncol(nlav_rlav_bi$genotypes))
)

feature_sets <- list(
  LAV_bi = list(obj = lav_bi, cols = cols$LAV_bi),
  nLAV_rLAV_bi = list(obj = nlav_rlav_bi, cols = cols$nLAV_rLAV_bi)
)

score_defs <- list(
  S_LAV_bi = "LAV_bi",
  S_nLAV_rLAV_bi = "nLAV_rLAV_bi"
)

formulas <- c(
  M1 = "Pheno ~ S_LAV_bi",
  M2 = "Pheno ~ S_nLAV_rLAV_bi"
)

run_one_permutation <- function(perm_idx) {
  pheno_file <- file.path(opt$perm_root, sprintf("perm_%d", perm_idx), sprintf("%s.phen", trait))
  if (!file.exists(pheno_file)) stop("Missing permuted phenotype file: ", pheno_file)

  cat("\nPermutation ", perm_idx, "\n", sep = "")
  cat("Phenotype file: ", pheno_file, "\n", sep = "")

  phen <- fread(pheno_file, header = FALSE, col.names = c("FID", "IID", "Pheno"))
  phen[, Pheno := as.numeric(as.character(Pheno))]
  base <- phen[!is.na(Pheno) & Pheno != -9]
  base <- merge(fam_ref, base, by = c("FID", "IID"), sort = FALSE)
  setorder(base, Index)
  if (nrow(base) == 0) stop("No samples left after merging phenotype with genotype fam for perm ", perm_idx)

  r2_rows <- list()
  for (rep_id in seq_len(opt$n_reps)) {
    cat(sprintf("Rep %d\n", rep_id))

    train_ids <- fread(file.path(split_dir, sprintf("rep%d_train.txt", rep_id)),
                       header = FALSE,
                       col.names = c("FID", "IID"))
    test_ids <- fread(file.path(split_dir, sprintf("rep%d_test.txt", rep_id)),
                      header = FALSE,
                      col.names = c("FID", "IID"))

    train_data <- merge(base, train_ids, by = c("FID", "IID"), sort = FALSE)
    test_data <- merge(base, test_ids, by = c("FID", "IID"), sort = FALSE)
    setorder(train_data, Index)
    setorder(test_data, Index)
    cat(sprintf("  Samples after merge: train=%d | test=%d\n", nrow(train_data), nrow(test_data)))
    if (nrow(train_data) == 0 || nrow(test_data) == 0) {
      stop("No samples left after merging permuted phenotype with split IDs")
    }

    fold_data <- copy(base)
    y_train <- train_data$Pheno

    for (score_name in names(score_defs)) {
      fs <- feature_sets[[score_defs[[score_name]]]]
      res <- train_score(fs, score_name, perm_idx, rep_id, base, train_data, y_train)
      fold_data[, (score_name) := res$score]
    }

    train_eval <- fold_data[Index %in% train_data$Index]
    test_eval <- fold_data[Index %in% test_data$Index]
    setorder(train_eval, Index)
    setorder(test_eval, Index)

    r2 <- vapply(formulas, fit_r2, numeric(1), train_data = train_eval, test_data = test_eval)
    r2_rows[[length(r2_rows) + 1]] <- data.table(Model = names(r2), Rep = rep_id, R2 = as.numeric(r2))
  }

  r2_long <- rbindlist(r2_rows)
  r2_wide <- dcast(r2_long, Model ~ Rep, value.var = "R2")
  setnames(r2_wide, as.character(seq_len(opt$n_reps)), paste0("Rep", seq_len(opt$n_reps)))

  out_dir <- file.path(out_root, sprintf("perm_%d", perm_idx))
  dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
  fwrite(r2_wide, file.path(out_dir, sprintf("trait%s_model_r2.tsv", trait)), sep = "\t")
  cat(sprintf("[%s] Finished permutation %d. Results saved in %s\n", Sys.time(), perm_idx, out_dir))
}

perm_ids <- seq.int(opt$perm_start, length.out = opt$n_perms)
invisible(lapply(perm_ids, run_one_permutation))
