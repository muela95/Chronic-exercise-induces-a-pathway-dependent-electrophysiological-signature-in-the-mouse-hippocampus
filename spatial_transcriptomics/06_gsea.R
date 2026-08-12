# ---------------------------------------------------------------------------
# 06 - GO and KEGG enrichment on the animal-level ranking (GSEA)
#
# An over-representation test needs a list of significant genes, which this
# design cannot produce. GSEA is used instead: it runs on the complete ranked
# gene list and requires no significance cutoff.
#
# Ranking metric: the DESeq2 Wald statistic.
#
# Two KEGG pathways are tracked explicitly across all categories, long-term
# potentiation (mmu04720) and cGMP-PKG signalling (mmu04022), together with GO
# terms covering synaptic maturation, calcium handling and dephosphorylation.
# ---------------------------------------------------------------------------

suppressPackageStartupMessages({
  library(clusterProfiler)
  library(org.Mm.eg.db)
  library(DOSE)
})

set.seed(1234)

root    <- "path/to/slide_seq_project"
de_dir  <- file.path(root, "out", "de_pseudobulk")
gs_dir  <- file.path(root, "out", "gsea")
dir.create(gs_dir, recursive = TRUE, showWarnings = FALSE)

de_files <- list.files(de_dir, pattern = "^DE_.*\\.csv$", full.names = TRUE)
de_files <- de_files[!grepl("DE_summary", de_files)]

# Pathways tracked explicitly across every category.
TARGET_KEGG <- c("mmu04720" = "Long-term potentiation",
                 "mmu04022" = "cGMP-PKG signaling pathway")
TARGET_GO_PATTERN <- "assembly|maturation|morphogenesis|differentiation|synap|calcium|dephosphorylation"

tracker <- list()

for (f in de_files) {
  label <- sub("^DE_", "", tools::file_path_sans_ext(basename(f)))
  message("\n========== ", label, " ==========")

  df <- read.csv(f)
  df <- df[!is.na(df$stat), ]

  map <- suppressMessages(
    AnnotationDbi::select(org.Mm.eg.db, keys = df$gene, keytype = "SYMBOL",
                          columns = "ENTREZID"))
  map <- map[!is.na(map$ENTREZID) & !duplicated(map$SYMBOL), ]
  df <- merge(df, map, by.x = "gene", by.y = "SYMBOL")
  df <- df[!duplicated(df$ENTREZID), ]
  message("genes with ENTREZID: ", nrow(df))

  rank_sym <- setNames(df$stat, df$gene)
  rank_sym <- sort(rank_sym, decreasing = TRUE)
  rank_ent <- setNames(df$stat, df$ENTREZID)
  rank_ent <- sort(rank_ent, decreasing = TRUE)

  # --- GO -------------------------------------------------------------------
  for (ont in c("BP", "MF", "CC")) {
    r <- tryCatch(
      gseGO(rank_sym, OrgDb = org.Mm.eg.db, keyType = "SYMBOL", ont = ont,
            minGSSize = 15, maxGSSize = 500, pvalueCutoff = 1,
            eps = 0, seed = TRUE, verbose = FALSE),
      error = function(e) { message("GO ", ont, " failed: ", conditionMessage(e)); NULL })

    if (!is.null(r) && nrow(as.data.frame(r))) {
      out <- as.data.frame(r)
      write.csv(out, file.path(gs_dir, paste0("GSEA_GO", ont, "_", label, ".csv")),
                row.names = FALSE)
      sig <- sum(out$p.adjust < 0.05)
      message("GO ", ont, ": ", nrow(out), " terms, ", sig, " with p.adjust < 0.05")
      if (sig) {
        top <- head(out[out$p.adjust < 0.05, c("Description", "NES", "p.adjust")], 8)
        print(top, row.names = FALSE)
      }
      hits <- out[grepl(TARGET_GO_PATTERN, out$Description, ignore.case = TRUE) &
                    out$p.adjust < 0.05, ]
      if (nrow(hits)) {
        tracker[[paste0(label, "_GO", ont)]] <-
          data.frame(category = label, source = paste0("GO:", ont),
                     term = hits$Description, NES = round(hits$NES, 3),
                     p.adjust = signif(hits$p.adjust, 3))
      }
    } else message("GO ", ont, ": no terms")
  }

  # --- KEGG -----------------------------------------------------------------
  r <- tryCatch(
    gseKEGG(rank_ent, organism = "mmu", minGSSize = 15, maxGSSize = 500,
            pvalueCutoff = 1, eps = 0, seed = TRUE, verbose = FALSE),
    error = function(e) { message("KEGG failed: ", conditionMessage(e)); NULL })

  if (!is.null(r) && nrow(as.data.frame(r))) {
    out <- as.data.frame(r)
    write.csv(out, file.path(gs_dir, paste0("GSEA_KEGG_", label, ".csv")),
              row.names = FALSE)
    sig <- sum(out$p.adjust < 0.05)
    message("KEGG: ", nrow(out), " pathways, ", sig, " with p.adjust < 0.05")
    if (sig) print(head(out[out$p.adjust < 0.05, c("Description", "NES", "p.adjust")], 8),
                   row.names = FALSE)

    for (id in names(TARGET_KEGG)) {
      row <- out[out$ID == id, ]
      if (nrow(row)) {
        message(">>> ", TARGET_KEGG[[id]], " (", id, "): NES = ",
                round(row$NES, 3), ", p = ", signif(row$pvalue, 3),
                ", p.adjust = ", signif(row$p.adjust, 3))
        tracker[[paste0(label, "_", id)]] <-
          data.frame(category = label, source = "KEGG", term = row$Description,
                     NES = round(row$NES, 3), p.adjust = signif(row$p.adjust, 3))
      } else {
        message(">>> ", TARGET_KEGG[[id]], " (", id, "): not returned")
      }
    }
  } else message("KEGG: no pathways")
}

if (length(tracker)) {
  tr <- do.call(rbind, tracker)
  write.csv(tr, file.path(gs_dir, "TRACKED_pathways.csv"), row.names = FALSE)
  message("\n===== TRACKED PATHWAYS =====")
  print(tr, row.names = FALSE)
}

message("\ndone")
