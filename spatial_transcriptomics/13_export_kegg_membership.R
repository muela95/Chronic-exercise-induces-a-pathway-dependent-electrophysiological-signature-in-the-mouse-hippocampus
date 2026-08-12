# ---------------------------------------------------------------------------
# 13 - Export full KEGG gene-set membership and the exact ranked lists
#
# The GSEA output only carries core_enrichment (the leading edge). Redrawing the
# running enrichment score faithfully needs two more things:
#
#   1. the complete membership of each gene set
#   2. the exact ranked vector that GSEA walked down, including the genes that
#      were dropped when symbols were mapped to Entrez IDs and duplicates were
#      removed
#
# Both are written here, for mmu04720 (long-term potentiation) and mmu04022
# (cGMP-PKG signalling).
#
# The running score at position i is the standard Kolmogorov-Smirnov style walk:
#   hits   -> running += |stat_i|^1 / sum(|stat_j|) over j in set
#   misses -> running -= 1 / (N - Nh)
# with N the length of the ranked list and Nh the number of set members in it.
# ---------------------------------------------------------------------------

suppressPackageStartupMessages({
  library(clusterProfiler)
  library(org.Mm.eg.db)
})

root   <- "path/to/slide_seq_project"
de_dir <- file.path(root, "out", "de_pseudobulk")
gs_dir <- file.path(root, "out", "gsea")
out_dir <- file.path(gs_dir, "kegg_membership")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

TARGETS <- c("mmu04720" = "Long-term potentiation",
             "mmu04022" = "cGMP-PKG signaling pathway")

# --- Full membership of both pathways ---------------------------------------
kegg <- download_KEGG("mmu")
p2g  <- kegg$KEGGPATHID2EXTID   # pathway -> Entrez
colnames(p2g) <- c("pathway", "entrez")

for (id in names(TARGETS)) {
  members <- p2g$entrez[p2g$pathway == id]
  sym <- suppressMessages(
    AnnotationDbi::select(org.Mm.eg.db, keys = members, keytype = "ENTREZID",
                          columns = "SYMBOL"))
  sym <- sym[!duplicated(sym$ENTREZID), ]
  df <- data.frame(pathway = id, pathway_name = TARGETS[[id]],
                   entrez = sym$ENTREZID, symbol = sym$SYMBOL)
  write.csv(df, file.path(out_dir, paste0("KEGG_", id, "_full_membership.csv")),
            row.names = FALSE)
  message(id, " (", TARGETS[[id]], "): ", nrow(df), " genes in the KEGG set")
}

ltp_set  <- p2g$entrez[p2g$pathway == "mmu04720"]
cgmp_set <- p2g$entrez[p2g$pathway == "mmu04022"]

# --- Exact ranked list per category, with set flags -------------------------
de_files <- list.files(de_dir, pattern = "^DE_.*\\.csv$", full.names = TRUE)
de_files <- de_files[!grepl("DE_summary", de_files)]

check <- list()

for (f in de_files) {
  label <- sub("^DE_", "", tools::file_path_sans_ext(basename(f)))

  df <- read.csv(f)
  df <- df[!is.na(df$stat), ]

  # Same mapping and de-duplication as script 06, in the same order, so the
  # ranked vector is identical to the one GSEA used.
  map <- suppressMessages(
    AnnotationDbi::select(org.Mm.eg.db, keys = df$gene, keytype = "SYMBOL",
                          columns = "ENTREZID"))
  map <- map[!is.na(map$ENTREZID) & !duplicated(map$SYMBOL), ]
  df <- merge(df, map, by.x = "gene", by.y = "SYMBOL")
  df <- df[!duplicated(df$ENTREZID), ]
  df <- df[order(-df$stat), ]

  out <- data.frame(
    rank        = seq_len(nrow(df)),
    symbol      = df$gene,
    entrez      = df$ENTREZID,
    stat        = df$stat,
    log2FC      = df$log2FoldChange,
    pvalue      = df$pvalue,
    padj        = df$padj,
    in_mmu04720 = df$ENTREZID %in% ltp_set,
    in_mmu04022 = df$ENTREZID %in% cgmp_set
  )

  write.csv(out, file.path(out_dir, paste0("ranked_list_", label, ".csv")),
            row.names = FALSE)

  check[[label]] <- data.frame(
    category = label, ranked_list_length = nrow(out),
    n_in_mmu04720 = sum(out$in_mmu04720),
    n_in_mmu04022 = sum(out$in_mmu04022))
}

chk <- do.call(rbind, check)

# Cross-check against the setSize reported by GSEA.
for (i in seq_len(nrow(chk))) {
  lab <- chk$category[i]
  f <- file.path(gs_dir, paste0("GSEA_KEGG_", lab, ".csv"))
  if (!file.exists(f)) next
  k <- read.csv(f)
  chk$gsea_setSize_04720[i] <- if (any(k$ID == "mmu04720")) k$setSize[k$ID == "mmu04720"] else NA
  chk$gsea_setSize_04022[i] <- if (any(k$ID == "mmu04022")) k$setSize[k$ID == "mmu04022"] else NA
}

write.csv(chk, file.path(out_dir, "ranked_list_summary.csv"), row.names = FALSE)

message("\n===== membership present in each ranked list vs GSEA setSize =====")
print(chk, row.names = FALSE)
