# ---------------------------------------------------------------------------
# 05 - Differential expression, RUN vs SED, animal as unit of replication
#
# DESeq2 on the pseudobulk matrices from script 04. Design: ~ group, with SED
# as reference. n = 2 SED vs 3 RUN.
#
# With two and three replicates the dispersion estimate is unstable, and an
# exact test at animal level bottoms out at p = 0.1, because only ten label
# assignments exist. Per-gene significance is therefore not the objective: the
# output is a gene ranking for the enrichment analysis in script 06.
#
# Genes are kept if they reach at least 5 counts in at least 2 animals.
# Categories with too little depth to fit a dispersion trend are skipped and
# reported as such.
# ---------------------------------------------------------------------------

suppressPackageStartupMessages({
  library(DESeq2)
})

root   <- "path/to/slide_seq_project"
pb_dir <- file.path(root, "out", "pseudobulk")
de_dir <- file.path(root, "out", "de_pseudobulk")
dir.create(de_dir, recursive = TRUE, showWarnings = FALSE)

samples <- c("J11", "J12", "J21", "J22", "J23")
coldata <- data.frame(
  row.names = samples,
  group = factor(c("SED", "SED", "RUN", "RUN", "RUN"), levels = c("SED", "RUN"))
)

MIN_COUNT   <- 5
MIN_SAMPLES <- 2
# Categories below this depth in any animal cannot support a dispersion fit.
MIN_TOTAL_PER_ANIMAL <- 20000

summary_rows <- list()

for (var in c("FinalID", "hippocampal_region")) {
  pb_list <- readRDS(file.path(pb_dir, paste0("pseudobulk_", var, ".rds")))

  for (cat in names(pb_list)) {
    cts <- round(pb_list[[cat]][, samples])
    mode(cts) <- "integer"
    depth <- colSums(cts)

    label <- paste0(var, " / ", cat)
    message("\n===== ", label, " =====")
    message("counts per animal: ", paste(depth, collapse = ", "))

    if (min(depth) < MIN_TOTAL_PER_ANIMAL) {
      message("SKIPPED: minimum depth ", min(depth), " < ", MIN_TOTAL_PER_ANIMAL)
      summary_rows[[label]] <- data.frame(
        variable = var, category = cat, status = "skipped_low_depth",
        min_depth = min(depth), genes_tested = NA, n_padj_005 = NA,
        min_padj = NA, n_lfc_gt1 = NA
      )
      next
    }

    keep <- rowSums(cts >= MIN_COUNT) >= MIN_SAMPLES
    cts <- cts[keep, ]
    message("genes tested: ", nrow(cts))

    dds <- DESeqDataSetFromMatrix(cts, coldata, design = ~ group)
    ok <- TRUE
    res <- tryCatch({
      dds <- DESeq(dds, quiet = TRUE)
      results(dds, contrast = c("group", "RUN", "SED"))
    }, error = function(e) {
      message("parametric fit failed (", conditionMessage(e), "), retrying local")
      tryCatch({
        dds <- DESeq(dds, fitType = "local", quiet = TRUE)
        results(dds, contrast = c("group", "RUN", "SED"))
      }, error = function(e2) { ok <<- FALSE; NULL })
    })

    if (!ok || is.null(res)) {
      message("SKIPPED: DESeq2 could not fit")
      summary_rows[[label]] <- data.frame(
        variable = var, category = cat, status = "skipped_fit_failed",
        min_depth = min(depth), genes_tested = nrow(cts), n_padj_005 = NA,
        min_padj = NA, n_lfc_gt1 = NA
      )
      next
    }

    df <- as.data.frame(res)
    df$gene <- rownames(df)
    df <- df[order(-df$log2FoldChange), ]
    df <- df[, c("gene", "baseMean", "log2FoldChange", "lfcSE", "stat",
                 "pvalue", "padj")]

    write.csv(df, file.path(de_dir, paste0("DE_", var, "_", cat, ".csv")),
              row.names = FALSE)

    n_sig <- sum(df$padj < 0.05, na.rm = TRUE)
    message("genes with padj < 0.05: ", n_sig,
            " | smallest padj: ", signif(min(df$padj, na.rm = TRUE), 3))

    summary_rows[[label]] <- data.frame(
      variable = var, category = cat, status = "analysed",
      min_depth = min(depth), genes_tested = nrow(cts), n_padj_005 = n_sig,
      min_padj = signif(min(df$padj, na.rm = TRUE), 3),
      n_lfc_gt1 = sum(abs(df$log2FoldChange) > 1, na.rm = TRUE)
    )
  }
}

summ <- do.call(rbind, summary_rows)
write.csv(summ, file.path(de_dir, "DE_summary.csv"), row.names = FALSE)

message("\n===== SUMMARY =====")
print(summ, row.names = FALSE)
