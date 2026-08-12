# ---------------------------------------------------------------------------
# 14 - Dentate gyrus pseudobulk under an annotation-confidence threshold
#
# Beads assigned to DG are kept only above a minimum MapMyCells subclass
# correlation coefficient, and the whole analysis is repeated at each cutoff.
#
# Thresholding also removes counts, and a change in depth moves p-values on its
# own, so p-values alone cannot say whether a threshold improves the analysis.
# The criterion used here is whether the real animal split separates further
# from the nine alternative splits, tested at every threshold with the same
# exact permutation as script 10.
#
# Thresholds stop at 0.30. At 0.35 J12 keeps only 50 beads, which recreates the
# imbalance that made CA1 untestable.
# ---------------------------------------------------------------------------

suppressPackageStartupMessages({
  library(Seurat)
  library(Matrix)
  library(DESeq2)
  library(clusterProfiler)
  library(org.Mm.eg.db)
})

set.seed(1234)

root    <- "path/to/slide_seq_project"
obj_dir <- file.path(root, "out", "objects")
out_dir <- file.path(root, "out", "dg_threshold")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

samples    <- c("J11", "J12", "J21", "J22", "J23")
SED        <- c("J11", "J12")
THRESHOLDS <- c(0, 0.15, 0.20, 0.25, 0.30)

all_genes <- sort(unique(unlist(lapply(samples, function(s)
  readLines(file.path(root, tolower(s), paste0(s, "_genes.tsv")))))))

# --- Aggregate DG counts per animal at every threshold, in one pass ----------
pb <- lapply(THRESHOLDS, function(x) matrix(0, nrow = length(all_genes),
                                            ncol = length(samples),
                                            dimnames = list(all_genes, samples)))
names(pb) <- as.character(THRESHOLDS)
nbeads <- matrix(0L, nrow = length(THRESHOLDS), ncol = length(samples),
                 dimnames = list(as.character(THRESHOLDS), samples))

for (s in samples) {
  message("aggregating ", s)
  obj <- readRDS(file.path(obj_dir, paste0(s, "_seurat.rds")))
  keep <- obj$hippocampal_region == "DG"
  counts <- LayerData(obj[["Spatial"]], "counts")[, keep, drop = FALSE]
  corr <- obj$subclass_correlation_coefficient[keep]

  for (thr in THRESHOLDS) {
    idx <- which(corr >= thr)
    nbeads[as.character(thr), s] <- length(idx)
    v <- Matrix::rowSums(counts[, idx, drop = FALSE])
    pb[[as.character(thr)]][names(v), s] <- v
  }
  rm(obj, counts); gc(verbose = FALSE)
}

for (thr in THRESHOLDS) {
  write.csv(pb[[as.character(thr)]],
            file.path(out_dir, sprintf("pseudobulk_DG_corr%.2f.csv", thr)))
}
write.csv(nbeads, file.path(out_dir, "beads_per_threshold.csv"))

message("\n===== beads per animal =====");  print(nbeads)
message("\n===== total counts per animal =====")
print(round(t(sapply(as.character(THRESHOLDS), function(t) colSums(pb[[t]])))))

# --- DESeq2 + GSEA over all ten animal splits, at each threshold ------------
splits <- combn(samples, 2, simplify = FALSE)
rows <- list()

for (thr in THRESHOLDS) {
  cts_all <- round(pb[[as.character(thr)]])
  mode(cts_all) <- "integer"
  message("\n################ threshold ", thr, " ################")

  for (i in seq_along(splits)) {
    small <- splits[[i]]
    is_true <- setequal(small, SED)
    coldata <- data.frame(
      row.names = samples,
      group = factor(ifelse(samples %in% small, "A", "B"), levels = c("A", "B")))

    keep <- rowSums(cts_all >= 5) >= 2
    cts <- cts_all[keep, ]

    res <- tryCatch({
      dds <- DESeqDataSetFromMatrix(cts, coldata, design = ~ group)
      dds <- DESeq(dds, quiet = TRUE)
      results(dds, contrast = c("group", "B", "A"))
    }, error = function(e) tryCatch({
      dds <- DESeqDataSetFromMatrix(cts, coldata, design = ~ group)
      dds <- DESeq(dds, fitType = "local", quiet = TRUE)
      results(dds, contrast = c("group", "B", "A"))
    }, error = function(e2) NULL))

    if (is.null(res)) next
    df <- as.data.frame(res); df$gene <- rownames(df); df <- df[!is.na(df$stat), ]

    map <- suppressMessages(
      AnnotationDbi::select(org.Mm.eg.db, keys = df$gene, keytype = "SYMBOL",
                            columns = "ENTREZID"))
    map <- map[!is.na(map$ENTREZID) & !duplicated(map$SYMBOL), ]
    df <- merge(df, map, by.x = "gene", by.y = "SYMBOL")
    df <- df[!duplicated(df$ENTREZID), ]
    rk <- sort(setNames(df$stat, df$ENTREZID), decreasing = TRUE)

    kg <- tryCatch(as.data.frame(
      gseKEGG(rk, organism = "mmu", minGSSize = 15, maxGSSize = 500,
              pvalueCutoff = 1, eps = 0, seed = TRUE, verbose = FALSE)),
      error = function(e) NULL)

    g <- function(id, col) if (!is.null(kg) && any(kg$ID == id)) kg[kg$ID == id, col] else NA

    rows[[length(rows) + 1]] <- data.frame(
      threshold = thr, split = i, is_true_design = is_true,
      min_beads = min(nbeads[as.character(thr), ]),
      min_depth = min(colSums(cts_all)),
      genes_tested = nrow(cts),
      n_genes_padj005 = sum(df$padj < 0.05, na.rm = TRUE),
      n_kegg_padj005 = if (is.null(kg)) NA else sum(kg$p.adjust < 0.05),
      LTP_NES = g("mmu04720", "NES"), LTP_padj = g("mmu04720", "p.adjust"),
      cGMP_NES = g("mmu04022", "NES"), cGMP_padj = g("mmu04022", "p.adjust"))
  }
  message("threshold ", thr, " done")
}

perm <- do.call(rbind, rows)
write.csv(perm, file.path(out_dir, "DG_threshold_permutation.csv"), row.names = FALSE)

# --- Summary: real split versus the nine alternatives, per threshold --------
message("\n\n===== REAL DESIGN AT EACH THRESHOLD =====")
real <- perm[perm$is_true_design, c("threshold", "min_beads", "min_depth",
                                    "genes_tested", "n_genes_padj005",
                                    "n_kegg_padj005", "LTP_NES", "LTP_padj",
                                    "cGMP_NES", "cGMP_padj")]
print(real, row.names = FALSE, digits = 3)

message("\n===== PERMUTATION RANK OF THE REAL SPLIT (1 = best of 10) =====")
summ <- do.call(rbind, lapply(THRESHOLDS, function(thr) {
  d <- perm[perm$threshold == thr, ]
  r <- d[d$is_true_design, ]
  data.frame(
    threshold = thr,
    rank_LTP  = sum(d$LTP_NES  >= r$LTP_NES,  na.rm = TRUE),
    rank_cGMP = sum(d$cGMP_NES >= r$cGMP_NES, na.rm = TRUE),
    rank_genes = sum(d$n_genes_padj005 >= r$n_genes_padj005, na.rm = TRUE),
    rank_kegg  = sum(d$n_kegg_padj005  >= r$n_kegg_padj005,  na.rm = TRUE),
    n_splits_with_any_sig_gene = sum(d$n_genes_padj005 > 0, na.rm = TRUE))
}))
print(summ, row.names = FALSE)
