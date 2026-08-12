# ---------------------------------------------------------------------------
# 11 - Leave-one-out robustness check for the dentate gyrus result
#
# The permutation test (script 10) put the real design first of ten in the
# dentate gyrus. This checks whether that depends on a single animal.
#
# Each RUN animal is dropped in turn, leaving a 2 vs 2 comparison. J21 matters
# most: it carries 12.2M UMI against 7.4M for the shallowest library, so a
# result driven by J21 would be a sequencing-depth artefact rather than an
# effect of exercise.
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

pb <- readRDS(file.path(pb_dir, "pseudobulk_hippocampal_region.rds"))
cts_all <- round(pb[["DG"]])
mode(cts_all) <- "integer"

sed <- c("J11", "J12")
run <- c("J21", "J22", "J23")

runs <- c(list(list(name = "all 5 animals (2 vs 3)", keep = c(sed, run))),
          lapply(run, function(d)
            list(name = paste0("drop ", d, " (2 vs 2)"),
                 keep = c(sed, setdiff(run, d)))))

out_rows <- list()

for (r in runs) {
  s <- r$keep
  cts <- cts_all[, s]
  keep <- rowSums(cts >= 5) >= 2
  cts <- cts[keep, ]

  coldata <- data.frame(
    row.names = s,
    group = factor(ifelse(s %in% sed, "SED", "RUN"), levels = c("SED", "RUN")))

  dds <- DESeqDataSetFromMatrix(cts, coldata, design = ~ group)
  res <- tryCatch({
    dds <- DESeq(dds, quiet = TRUE); results(dds, contrast = c("group", "RUN", "SED"))
  }, error = function(e) {
    dds <- DESeq(dds, fitType = "local", quiet = TRUE)
    results(dds, contrast = c("group", "RUN", "SED"))
  })

  df <- as.data.frame(res); df$gene <- rownames(df); df <- df[!is.na(df$stat), ]
  n_sig <- sum(df$padj < 0.05, na.rm = TRUE)

  map <- suppressMessages(
    AnnotationDbi::select(org.Mm.eg.db, keys = df$gene, keytype = "SYMBOL",
                          columns = "ENTREZID"))
  map <- map[!is.na(map$ENTREZID) & !duplicated(map$SYMBOL), ]
  df <- merge(df, map, by.x = "gene", by.y = "SYMBOL")
  df <- df[!duplicated(df$ENTREZID), ]
  rank_ent <- sort(setNames(df$stat, df$ENTREZID), decreasing = TRUE)

  kg <- as.data.frame(gseKEGG(rank_ent, organism = "mmu", minGSSize = 15,
                              maxGSSize = 500, pvalueCutoff = 1, eps = 0,
                              seed = TRUE, verbose = FALSE))

  get <- function(id, col) { x <- kg[kg$ID == id, col]; if (length(x)) x else NA }

  out_rows[[r$name]] <- data.frame(
    analysis = r$name, n_animals = length(s), genes_tested = nrow(cts),
    n_genes_padj005 = n_sig,
    LTP_NES  = round(get("mmu04720", "NES"), 3),
    LTP_padj = signif(get("mmu04720", "p.adjust"), 3),
    cGMP_NES  = round(get("mmu04022", "NES"), 3),
    cGMP_padj = signif(get("mmu04022", "p.adjust"), 3))

  message(r$name, ": genes padj<0.05 = ", n_sig,
          " | LTP NES = ", round(get("mmu04720", "NES"), 3),
          " | cGMP NES = ", round(get("mmu04022", "NES"), 3))
}

res_tab <- do.call(rbind, out_rows)
write.csv(res_tab, file.path(pm_dir, "leave_one_out_DG.csv"), row.names = FALSE)

message("\n===== LEAVE-ONE-OUT, DENTATE GYRUS =====")
print(res_tab, row.names = FALSE)
