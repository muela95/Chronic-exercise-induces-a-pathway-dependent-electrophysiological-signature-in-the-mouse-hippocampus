# ---------------------------------------------------------------------------
# 10 - Exact animal-level permutation test
#
# With five animals there are exactly choose(5,2) = 10 ways to split them into a
# group of two and a group of three. One of those ten is the real design
# (SED = J11, J12). The other nine are label assignments with no biological
# meaning.
#
# For every split the full pipeline is repeated: DESeq2 on the pseudobulk, then
# GSEA. If the real split produces stronger results than the nine fake ones, the
# effect is attributable to exercise. If it sits in the middle of the pack, the
# signal is between-animal variation and nothing else.
#
# Because only ten assignments exist, the smallest attainable permutation
# p-value is 0.1. Results are therefore reported as the rank of the real design
# among the ten splits rather than as a significance claim.
# ---------------------------------------------------------------------------

suppressPackageStartupMessages({
  library(DESeq2)
  library(clusterProfiler)
  library(org.Mm.eg.db)
})

set.seed(1234)

root   <- "path/to/slide_seq_project"
pb_dir <- file.path(root, "out", "pseudobulk")
pm_dir <- file.path(root, "out", "permutation")
dir.create(pm_dir, recursive = TRUE, showWarnings = FALSE)

samples <- c("J11", "J12", "J21", "J22", "J23")
TRUE_SMALL_GROUP <- c("J11", "J12")   # the real SED pair

TARGET_KEGG <- c("mmu04720" = "Long-term potentiation",
                 "mmu04022" = "cGMP-PKG signaling pathway")

targets <- list(
  list(var = "hippocampal_region", cat = "DG"),
  list(var = "FinalID",            cat = "Glutamatergic_Neurons"),
  list(var = "FinalID",            cat = "Astrocytes")
)

splits <- combn(samples, 2, simplify = FALSE)

results <- list()

for (t in targets) {
  pb_list <- readRDS(file.path(pb_dir, paste0("pseudobulk_", t$var, ".rds")))
  cts_all <- round(pb_list[[t$cat]][, samples])
  mode(cts_all) <- "integer"
  label <- paste0(t$var, "_", t$cat)
  message("\n################ ", label, " ################")

  for (i in seq_along(splits)) {
    small <- splits[[i]]
    is_true <- setequal(small, TRUE_SMALL_GROUP)
    tag <- paste0(paste(small, collapse = "+"), " vs ",
                  paste(setdiff(samples, small), collapse = "+"))
    message("\n--- split ", i, "/10: ", tag, if (is_true) "   <== REAL DESIGN" else "")

    coldata <- data.frame(
      row.names = samples,
      group = factor(ifelse(samples %in% small, "A", "B"), levels = c("A", "B")))

    keep <- rowSums(cts_all >= 5) >= 2
    cts <- cts_all[keep, ]

    dds <- DESeqDataSetFromMatrix(cts, coldata, design = ~ group)
    res <- tryCatch({
      dds <- DESeq(dds, quiet = TRUE)
      results(dds, contrast = c("group", "B", "A"))
    }, error = function(e) {
      dds <- DESeq(dds, fitType = "local", quiet = TRUE)
      results(dds, contrast = c("group", "B", "A"))
    })

    df <- as.data.frame(res)
    df$gene <- rownames(df)
    df <- df[!is.na(df$stat), ]
    n_sig <- sum(df$padj < 0.05, na.rm = TRUE)

    map <- suppressMessages(
      AnnotationDbi::select(org.Mm.eg.db, keys = df$gene, keytype = "SYMBOL",
                            columns = "ENTREZID"))
    map <- map[!is.na(map$ENTREZID) & !duplicated(map$SYMBOL), ]
    df <- merge(df, map, by.x = "gene", by.y = "SYMBOL")
    df <- df[!duplicated(df$ENTREZID), ]
    rank_ent <- sort(setNames(df$stat, df$ENTREZID), decreasing = TRUE)

    kg <- tryCatch(
      gseKEGG(rank_ent, organism = "mmu", minGSSize = 15, maxGSSize = 500,
              pvalueCutoff = 1, eps = 0, seed = TRUE, verbose = FALSE),
      error = function(e) NULL)

    row <- data.frame(category = label, split = i, groups = tag,
                      is_true_design = is_true, genes_tested = nrow(cts),
                      n_genes_padj005 = n_sig,
                      n_kegg_padj005 = NA_integer_,
                      LTP_NES = NA_real_, LTP_padj = NA_real_,
                      cGMP_NES = NA_real_, cGMP_padj = NA_real_)

    if (!is.null(kg)) {
      out <- as.data.frame(kg)
      row$n_kegg_padj005 <- sum(out$p.adjust < 0.05)
      for (id in names(TARGET_KEGG)) {
        r <- out[out$ID == id, ]
        if (nrow(r)) {
          if (id == "mmu04720") { row$LTP_NES <- r$NES;  row$LTP_padj <- r$p.adjust }
          if (id == "mmu04022") { row$cGMP_NES <- r$NES; row$cGMP_padj <- r$p.adjust }
        }
      }
    }

    message("genes padj<0.05: ", n_sig,
            " | KEGG padj<0.05: ", row$n_kegg_padj005,
            " | LTP NES: ", round(row$LTP_NES, 3),
            " | cGMP NES: ", round(row$cGMP_NES, 3))

    results[[length(results) + 1]] <- row
  }
}

perm <- do.call(rbind, results)
write.csv(perm, file.path(pm_dir, "animal_permutation_results.csv"), row.names = FALSE)

message("\n\n===================== PERMUTATION SUMMARY =====================")
for (lab in unique(perm$category)) {
  d <- perm[perm$category == lab, ]
  real <- d[d$is_true_design, ]
  message("\n########## ", lab, " ##########")
  print(d[order(-d$LTP_NES),
          c("groups", "is_true_design", "n_genes_padj005", "n_kegg_padj005",
            "LTP_NES", "cGMP_NES")], row.names = FALSE)

  # Permutation p-value: the fraction of splits reaching at least the observed
  # value. The real split is included, so the minimum is 1/10.
  for (metric in c("LTP_NES", "cGMP_NES", "n_genes_padj005", "n_kegg_padj005")) {
    v <- d[[metric]]; obs <- real[[metric]]
    if (all(is.na(v)) || is.na(obs)) next
    p <- mean(v >= obs, na.rm = TRUE)
    message(sprintf("%-18s observed = %8.3f | rank %d of %d | permutation p = %.2f",
                    metric, obs, sum(v >= obs, na.rm = TRUE), sum(!is.na(v)), p))
  }
}
