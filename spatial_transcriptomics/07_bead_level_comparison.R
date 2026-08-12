# ---------------------------------------------------------------------------
# 07 - Bead-level differential expression, for comparison only
#
# Beads from all animals are pooled by condition, merged, reprocessed with
# SCTransform, and compared with FindMarkers (Wilcoxon, no log-fold-change
# threshold, Benjamini-Hochberg correction). Every spatial barcode is treated as
# an independent observation.
#
# It provides the comparison against which the animal-level analysis of
# scripts 04 to 06 is read.
#
# A negative control is included. For the dentate gyrus the same test is
# repeated with the animals regrouped so that the split ignores treatment
# (J11 + J21 vs J12 + J22 + J23). Genes called significant under that split
# reflect between-animal variation alone, and give the false-positive floor of
# the bead-level approach.
# ---------------------------------------------------------------------------

suppressPackageStartupMessages({
  library(Seurat)
})

set.seed(1234)
options(future.globals.maxSize = 16 * 1024^3)

root    <- "path/to/slide_seq_project"
obj_dir <- file.path(root, "out", "objects")
bl_dir  <- file.path(root, "out", "de_bead_level")
dir.create(bl_dir, recursive = TRUE, showWarnings = FALSE)

samples <- c("J11", "J12", "J21", "J22", "J23")
group   <- c(J11 = "SED", J12 = "SED", J21 = "RUN", J22 = "RUN", J23 = "RUN")
# Control split: ignores treatment, keeps the 2 vs 3 structure.
sham    <- c(J11 = "A",   J12 = "B",   J21 = "A",   J22 = "B",   J23 = "B")

targets <- list(
  list(var = "FinalID",            cat = "Astrocytes"),
  list(var = "FinalID",            cat = "GABAergic_Neurons"),
  list(var = "FinalID",            cat = "Glutamatergic_Neurons"),
  list(var = "FinalID",            cat = "Oligodendrocytes"),
  list(var = "FinalID",            cat = "Ependymal"),
  # CA1 is included although the animal-level analysis cannot test it: J12
  # contributes 5,571 counts against 284,148 in J11. The bead-level test
  # returns results for it regardless.
  list(var = "hippocampal_region", cat = "CA1"),
  list(var = "hippocampal_region", cat = "CA3"),
  list(var = "hippocampal_region", cat = "DG")
)

# Load once, subset per target.
message("loading objects...")
objs <- lapply(samples, function(s) readRDS(file.path(obj_dir, paste0(s, "_seurat.rds"))))
names(objs) <- samples

run_test <- function(merged, ident_col, id1, id2, tag) {
  Idents(merged) <- ident_col
  res <- FindMarkers(merged, ident.1 = id1, ident.2 = id2,
                     logfc.threshold = 0, min.pct = 0, min.diff.pct = 0,
                     min.cells.feature = 1, test.use = "wilcox",
                     verbose = FALSE)
  res$gene <- rownames(res)
  write.csv(res, file.path(bl_dir, paste0("beadDE_", tag, ".csv")), row.names = FALSE)
  res
}

summary_rows <- list()

for (t in targets) {
  var <- t$var; cat_ <- t$cat
  tag <- paste0(var, "_", cat_)
  message("\n========== ", tag, " ==========")

  subs <- lapply(samples, function(s) {
    o <- objs[[s]]
    cells <- colnames(o)[o[[var]][, 1] == cat_]
    if (length(cells) < 3) return(NULL)
    sub <- subset(o, cells = cells)
    DefaultAssay(sub) <- "Spatial"
    sub <- DietSeurat(sub, assays = "Spatial")
    sub
  })
  names(subs) <- samples
  subs <- subs[!vapply(subs, is.null, logical(1))]

  n_per <- vapply(subs, function(x) as.integer(ncol(x)), integer(1))
  message("beads per animal: ", paste(names(n_per), n_per, sep = "=", collapse = ", "))

  merged <- merge(subs[[1]], y = subs[-1], add.cell.ids = names(subs))
  merged <- JoinLayers(merged)
  merged$group <- unname(group[merged$sample])
  merged$sham  <- unname(sham[merged$sample])

  merged <- SCTransform(merged, assay = "Spatial", vst.flavor = "v1",
                        verbose = FALSE)

  n_sed <- sum(merged$group == "SED"); n_run <- sum(merged$group == "RUN")
  message("SED beads: ", n_sed, " | RUN beads: ", n_run)

  res <- run_test(merged, "group", "RUN", "SED", tag)
  n_sig <- sum(res$p_val_adj < 0.05, na.rm = TRUE)
  message("RUN vs SED  -> genes with p_val_adj < 0.05: ", n_sig, " of ", nrow(res))

  n_sham <- NA
  if (cat_ == "DG") {
    res_s <- run_test(merged, "sham", "B", "A", paste0(tag, "_SHAM"))
    n_sham <- sum(res_s$p_val_adj < 0.05, na.rm = TRUE)
    message("CONTROL (treatment-blind split) -> genes with p_val_adj < 0.05: ",
            n_sham, " of ", nrow(res_s))
  }

  summary_rows[[tag]] <- data.frame(
    variable = var, category = cat_,
    beads_SED = n_sed, beads_RUN = n_run, beads_total = n_sed + n_run,
    genes_tested = nrow(res),
    n_padj_005_beadlevel = n_sig,
    n_padj_005_control_split = n_sham
  )

  rm(merged, subs, res); gc(verbose = FALSE)
}

summ <- do.call(rbind, summary_rows)
write.csv(summ, file.path(bl_dir, "bead_level_summary.csv"), row.names = FALSE)

message("\n===== BEAD-LEVEL SUMMARY =====")
print(summ, row.names = FALSE)
