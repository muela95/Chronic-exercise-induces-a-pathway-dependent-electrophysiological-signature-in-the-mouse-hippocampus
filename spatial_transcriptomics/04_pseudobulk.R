# ---------------------------------------------------------------------------
# 04 - Pseudobulk aggregation with the animal as the unit of replication
#
# For every cell type (FinalID) and every hippocampal subfield
# (hippocampal_region), raw counts are summed across all beads of an animal.
# The result is a genes x animal matrix with five columns: two SED, three RUN.
#
# The biological unit of replication is therefore the animal, not the spatial
# barcode.
#
# Gene space: each Seeker library reports only the genes it detected, so the
# per-sample gene lists differ (27,113 to 29,697; union 36,086). Counts are
# placed in the union space and missing genes are filled with zero, which is
# their true value: undetected in that library. Filtering happens at the DE
# step, not here.
# ---------------------------------------------------------------------------

suppressPackageStartupMessages({
  library(Seurat)
  library(Matrix)
})

root    <- "path/to/slide_seq_project"
out_dir <- file.path(root, "out")
obj_dir <- file.path(out_dir, "objects")
pb_dir  <- file.path(out_dir, "pseudobulk")
dir.create(pb_dir, recursive = TRUE, showWarnings = FALSE)

samples <- c("J11", "J12", "J21", "J22", "J23")
group   <- c(J11 = "SED", J12 = "SED", J21 = "RUN", J22 = "RUN", J23 = "RUN")

# Union gene space, taken from the Seeker gene lists.
all_genes <- sort(unique(unlist(lapply(samples, function(s)
  readLines(file.path(root, tolower(s), paste0(s, "_genes.tsv")))))))
message("union gene space: ", length(all_genes))

# Collect per-animal sums for each grouping variable.
collect <- list(FinalID = list(), hippocampal_region = list())
n_beads <- list(FinalID = list(), hippocampal_region = list())

for (s in samples) {
  message("\n===== ", s, " (", group[[s]], ") =====")
  obj <- readRDS(file.path(obj_dir, paste0(s, "_seurat.rds")))
  counts <- LayerData(obj[["Spatial"]], "counts")

  for (var in c("FinalID", "hippocampal_region")) {
    lab <- obj[[var]][, 1]
    mats <- list(); nb <- list()
    for (lv in sort(unique(lab))) {
      idx <- which(lab == lv)
      v <- Matrix::rowSums(counts[, idx, drop = FALSE])
      full <- setNames(numeric(length(all_genes)), all_genes)
      full[names(v)] <- v
      mats[[lv]] <- full
      nb[[lv]] <- length(idx)
    }
    collect[[var]][[s]] <- mats
    n_beads[[var]][[s]] <- unlist(nb)
    message(var, ": ", length(mats), " categories")
  }

  rm(obj, counts); gc(verbose = FALSE)
}

# Assemble one genes x animal matrix per category and save.
for (var in c("FinalID", "hippocampal_region")) {
  cats <- sort(unique(unlist(lapply(collect[[var]], names))))
  pb_list <- list()

  for (cat in cats) {
    m <- sapply(samples, function(s) {
      v <- collect[[var]][[s]][[cat]]
      if (is.null(v)) setNames(numeric(length(all_genes)), all_genes) else v
    })
    rownames(m) <- all_genes
    pb_list[[cat]] <- m
  }

  saveRDS(pb_list, file.path(pb_dir, paste0("pseudobulk_", var, ".rds")))

  # Bead counts per category and animal, for the supplementary table.
  nb <- sapply(samples, function(s) {
    v <- n_beads[[var]][[s]]
    setNames(as.integer(v[cats]), cats)
  })
  nb[is.na(nb)] <- 0L
  rownames(nb) <- cats
  write.csv(nb, file.path(pb_dir, paste0("beads_per_", var, "_per_animal.csv")))

  message("\n===== ", var, ": total pseudobulk counts per animal =====")
  tot <- sapply(cats, function(cat) colSums(pb_list[[cat]]))
  print(round(t(tot)))

  message("\n===== ", var, ": genes with >0 counts, per category and animal =====")
  det <- sapply(cats, function(cat) colSums(pb_list[[cat]] > 0))
  print(t(det))
}

message("\ndone")
