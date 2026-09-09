library(optparse)
library(data.table)
library(bigstatsr)
library(bigsnpr)

option_list <- list(
  make_option(c("--bfile"), type = "character", default = NULL,
              help = "PLINK bfile prefix."),
  make_option(c("--cn"), type = "character", default = NULL,
              help = "Copy-number TSV."),
  make_option(c("--tr"), type = "character", default = NULL,
              help = "TR matrix with SampleID in the first column and allele pairs such as 0/24."),
  make_option(c("--out"), type = "character", default = NULL,
              help = "Output prefix or .rds path."),
  make_option(c("--missing"), type = "character", default = "",
              help = "Optional comma-separated missing-value codes for CN table. -9 is real unless listed here."),
  make_option(c("--chunk_size"), type = "integer", default = 2000,
              help = "Columns per chunk when copying/imputing matrices.")
)

opt <- parse_args(OptionParser(option_list = option_list))
input_modes <- c(bfile = !is.null(opt$bfile), cn = !is.null(opt$cn), tr = !is.null(opt$tr))
if (sum(input_modes) < 1) stop("Provide at least one input: --bfile, --cn, or --tr")
if (is.null(opt$out)) stop("Missing --out")

out_prefix_from_path <- function(path) {
  sub("[.]rds$", "", path, ignore.case = TRUE)
}

opt$out <- out_prefix_from_path(opt$out)
dir.create(dirname(opt$out), recursive = TRUE, showWarnings = FALSE)

normalize_chr <- function(x) {
  x <- as.character(x)
  sub("^chr", "", x, ignore.case = TRUE)
}

chr_to_int <- function(x) {
  suppressWarnings(as.integer(normalize_chr(x)))
}

parse_region_id <- function(id, pos) {
  m <- regexec("^([^:]+):([0-9]+)-([0-9]+)$", id)
  p <- regmatches(id, m)
  start <- suppressWarnings(as.integer(pos))
  end <- start
  ok <- lengths(p) == 4
  if (any(ok)) {
    start[ok] <- as.integer(vapply(p[ok], `[`, character(1), 3))
    end[ok] <- as.integer(vapply(p[ok], `[`, character(1), 4))
  }
  list(start = start, end = end)
}

load_obj <- function(path) {
  obj <- tryCatch(snp_attach(path), error = function(e) readRDS(path))
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

save_obj <- function(G, fam, map, out_prefix) {
  obj <- list(genotypes = G, fam = fam, map = map)
  saveRDS(obj, paste0(out_prefix, ".rds"))
  cat("Saved: ", paste0(out_prefix, ".rds"), " and .bk\n", sep = "")
}

standardize_map <- function(map, source) {
  map <- as.data.frame(map, stringsAsFactors = FALSE)
  if (!"chromosome" %in% names(map)) map$chromosome <- NA_integer_
  if (!"marker.ID" %in% names(map)) map$marker.ID <- seq_len(nrow(map))
  if (!"physical.pos" %in% names(map)) map$physical.pos <- seq_len(nrow(map))
  if (!"allele1" %in% names(map)) map$allele1 <- NA_character_
  if (!"allele2" %in% names(map)) map$allele2 <- NA_character_
  data.frame(
    chromosome = map$chromosome,
    marker.ID = paste(source, map$marker.ID, sep = ":"),
    physical.pos = map$physical.pos,
    allele1 = map$allele1,
    allele2 = map$allele2,
    source = source,
    original.marker.ID = map$marker.ID,
    stringsAsFactors = FALSE
  )
}

impute_copy_fbm <- function(G_in, out_prefix, rows = NULL) {
  if (is.null(rows)) rows <- seq_len(nrow(G_in))
  n <- length(rows)
  p <- ncol(G_in)
  G_out <- FBM(nrow = n, ncol = p, type = "double", backingfile = out_prefix)
  chunks <- split(seq_len(p), ceiling(seq_len(p) / opt$chunk_size))
  total_missing <- 0

  for (k in seq_along(chunks)) {
    jj <- chunks[[k]]
    X <- as.matrix(G_in[rows, jj])
    miss <- is.na(X)
    n_miss <- sum(miss)
    total_missing <- total_missing + n_miss
    if (n_miss > 0) {
      mu <- colMeans(X, na.rm = TRUE)
      mu[is.nan(mu)] <- 0
      for (j in seq_along(jj)) {
        idx <- is.na(X[, j])
        if (any(idx)) X[idx, j] <- mu[j]
      }
    }
    G_out[, jj] <- X
    if (k %% 20 == 0 || k == length(chunks)) {
      cat(sprintf("  copied chunk %d / %d | missing imputed=%d\n",
                  k, length(chunks), total_missing))
    }
  }
  G_out
}

make_biallelic_rds <- function(bfile, out_prefix) {
  bedfile <- paste0(bfile, ".bed")
  if (!file.exists(bedfile)) stop("BED file not found: ", bedfile)
  tmp_prefix <- tempfile(pattern = "raw_bed_", tmpdir = dirname(out_prefix))
  raw_rds <- snp_readBed(bedfile, backingfile = tmp_prefix)
  raw <- snp_attach(raw_rds)

  cat("Imputing biallelic RDS: ", out_prefix, "\n", sep = "")
  G <- impute_copy_fbm(raw$genotypes, out_prefix)
  fam <- raw$fam
  map <- raw$map
  save_obj(G, fam, map, out_prefix)
  unlink(c(raw_rds, paste0(tmp_prefix, ".bk")), force = TRUE)
  load_obj(paste0(out_prefix, ".rds"))
}

make_cn_rds <- function(cn_file, out_prefix) {
  missing_codes <- unlist(strsplit(opt$missing, ",", fixed = TRUE))
  missing_codes <- missing_codes[nzchar(missing_codes)]
  cn <- fread(cn_file, na.strings = c("NA", "NaN", "", missing_codes))
  anno_cols <- c("#CHROM", "POS", "ID", "REPEAT_UNIT", "REPEAT_LENGTH")
  if (!all(anno_cols %in% names(cn))) {
    stop("CN file must contain columns: ", paste(anno_cols, collapse = ", "))
  }
  sample_cols <- setdiff(names(cn), anno_cols)
  if (length(sample_cols) == 0) stop("No sample columns found in CN table")

  reg <- parse_region_id(cn$ID, cn$POS)
  map <- data.frame(
    chromosome = chr_to_int(cn[["#CHROM"]]),
    marker.ID = as.character(cn[["ID"]]),
    physical.pos = as.integer(cn[["POS"]]),
    start = reg$start,
    end = reg$end,
    allele1 = "CNdiff",
    allele2 = "REF",
    repeat.unit = as.character(cn[["REPEAT_UNIT"]]),
    repeat.length = cn[["REPEAT_LENGTH"]],
    stringsAsFactors = FALSE
  )
  fam <- data.frame(
    family.ID = sample_cols,
    sample.ID = sample_cols,
    FID = sample_cols,
    IID = sample_cols,
    stringsAsFactors = FALSE
  )

  G <- FBM(nrow = length(sample_cols), ncol = nrow(cn), type = "double", backingfile = out_prefix)
  chunks <- split(seq_len(nrow(cn)), ceiling(seq_len(nrow(cn)) / opt$chunk_size))
  total_missing <- 0

  cat("Creating CN RDS: ", out_prefix, "\n", sep = "")
  for (k in seq_along(chunks)) {
    ii <- chunks[[k]]
    X <- as.matrix(cn[ii, ..sample_cols])
    storage.mode(X) <- "double"
    miss <- is.na(X)
    n_miss <- sum(miss)
    total_missing <- total_missing + n_miss
    if (n_miss > 0) {
      mu <- rowMeans(X, na.rm = TRUE)
      mu[is.nan(mu)] <- 0
      for (i in seq_along(ii)) {
        idx <- is.na(X[i, ])
        if (any(idx)) X[i, idx] <- mu[i]
      }
    }
    G[, ii] <- t(X)
    if (k %% 20 == 0 || k == length(chunks)) {
      cat(sprintf("  copied CN chunk %d / %d | missing imputed=%d\n",
                  k, length(chunks), total_missing))
    }
  }

  save_obj(G, fam, map, out_prefix)
  load_obj(paste0(out_prefix, ".rds"))
}

make_tr_matrix_rds <- function(matrix_file, out_prefix) {
  tr <- fread(matrix_file, sep = "\t", header = TRUE, data.table = TRUE,
              colClasses = "character", na.strings = c("NA", "NaN", ""))
  if (ncol(tr) < 2) stop("TR matrix must contain SampleID and at least one variant column")

  sample_col <- names(tr)[1]
  sample_ids <- tr[[sample_col]]
  if (anyDuplicated(sample_ids)) stop("Duplicated sample IDs in first column: ", sample_col)

  variant_ids <- names(tr)[-1]
  n <- length(sample_ids)
  p <- length(variant_ids)
  G <- FBM(nrow = n, ncol = p, type = "double", backingfile = out_prefix)
  chunks <- split(seq_len(p), ceiling(seq_len(p) / opt$chunk_size))
  total_missing <- 0

  encode_pair <- function(x) {
    x <- trimws(x)
    x[!nzchar(x)] <- NA_character_
    if (all(is.na(x))) return(rep(NA_real_, length(x)))
    ok <- is.na(x) | grepl("^-?[0-9]+/-?[0-9]+$", x)
    if (!all(ok)) {
      bad <- unique(x[!ok])[1]
      stop("TR genotype values must look like allele1/allele2; first bad value: ", bad)
    }
    alleles <- tstrsplit(x, "/", fixed = TRUE, fill = NA_character_)
    a1 <- suppressWarnings(as.numeric(alleles[[1]]))
    a2 <- suppressWarnings(as.numeric(alleles[[2]]))
    a1 + a2
  }

  cat("Creating TR matrix RDS: ", out_prefix, "\n", sep = "")
  for (k in seq_along(chunks)) {
    jj <- chunks[[k]]
    X <- matrix(NA_real_, nrow = n, ncol = length(jj))
    for (j in seq_along(jj)) {
      vals <- encode_pair(tr[[jj[j] + 1]])
      miss <- is.na(vals)
      total_missing <- total_missing + sum(miss)
      if (any(miss)) {
        mu <- mean(vals, na.rm = TRUE)
        if (is.nan(mu)) mu <- 0
        vals[miss] <- mu
      }
      X[, j] <- vals
    }
    G[, jj] <- X
    if (k %% 20 == 0 || k == length(chunks)) {
      cat(sprintf("  copied TR chunk %d / %d | missing imputed=%d\n",
                  k, length(chunks), total_missing))
    }
  }

  fam <- data.frame(
    family.ID = sample_ids,
    sample.ID = sample_ids,
    FID = sample_ids,
    IID = sample_ids,
    stringsAsFactors = FALSE
  )
  map <- data.frame(
    chromosome = NA_integer_,
    marker.ID = variant_ids,
    physical.pos = seq_along(variant_ids),
    allele1 = NA_character_,
    allele2 = NA_character_,
    stringsAsFactors = FALSE
  )

  save_obj(G, fam, map, out_prefix)
  load_obj(paste0(out_prefix, ".rds"))
}

make_joint_rds <- function(parts, out_prefix) {
  iid_list <- lapply(parts, function(x) as_iid(x$obj$fam))
  common_iid <- Reduce(intersect, iid_list)
  if (length(common_iid) == 0) stop("No overlapping samples across inputs")

  rows_list <- Map(function(part, iids) match(common_iid, iids), parts, iid_list)
  ncols <- vapply(parts, function(x) ncol(x$obj$genotypes), integer(1))
  G <- FBM(nrow = length(common_iid), ncol = sum(ncols), type = "double", backingfile = out_prefix)

  cat("Creating joint RDS: ", out_prefix, "\n", sep = "")
  col_offset <- 0
  for (i in seq_along(parts)) {
    obj <- parts[[i]]$obj
    rows <- rows_list[[i]]
    p <- ncol(obj$genotypes)
    chunks <- split(seq_len(p), ceiling(seq_len(p) / opt$chunk_size))
    for (k in seq_along(chunks)) {
      jj <- chunks[[k]]
      G[, col_offset + jj] <- obj$genotypes[rows, jj]
    }
    col_offset <- col_offset + p
  }

  fam <- data.frame(
    family.ID = as_fid(parts[[1]]$obj$fam)[rows_list[[1]]],
    sample.ID = common_iid,
    FID = as_fid(parts[[1]]$obj$fam)[rows_list[[1]]],
    IID = common_iid,
    stringsAsFactors = FALSE
  )
  map <- do.call(rbind, lapply(parts, function(x) standardize_map(x$obj$map, x$name)))

  save_obj(G, fam, map, out_prefix)
  load_obj(paste0(out_prefix, ".rds"))
}

if (sum(input_modes) == 1) {
  if (input_modes[["bfile"]]) {
    make_biallelic_rds(opt$bfile, opt$out)
  } else if (input_modes[["cn"]]) {
    make_cn_rds(opt$cn, opt$out)
  } else if (input_modes[["tr"]]) {
    make_tr_matrix_rds(opt$tr, opt$out)
  }
} else {
  parts <- list()
  tmp_prefixes <- character()

  if (input_modes[["bfile"]]) {
    tmp <- paste0(opt$out, ".tmp_bfile")
    tmp_prefixes <- c(tmp_prefixes, tmp)
    parts[["bfile"]] <- list(name = "bfile", obj = make_biallelic_rds(opt$bfile, tmp))
  }
  if (input_modes[["cn"]]) {
    tmp <- paste0(opt$out, ".tmp_cn")
    tmp_prefixes <- c(tmp_prefixes, tmp)
    parts[["cn"]] <- list(name = "cn", obj = make_cn_rds(opt$cn, tmp))
  }
  if (input_modes[["tr"]]) {
    tmp <- paste0(opt$out, ".tmp_tr")
    tmp_prefixes <- c(tmp_prefixes, tmp)
    parts[["tr"]] <- list(name = "tr", obj = make_tr_matrix_rds(opt$tr, tmp))
  }

  make_joint_rds(parts, opt$out)
  unlink(c(paste0(tmp_prefixes, ".rds"), paste0(tmp_prefixes, ".bk")), force = TRUE)
}

cat("Done. Final RDS written to: ", paste0(opt$out, ".rds"), "\n", sep = "")
