# ---------------------------------------------------------------------------
# 03 - Attach MapMyCells annotation to the Seurat objects
#
# Input:  MapMyCells output (10x Whole Mouse Brain CCN20230722, Correlation
#         Mapping, cell_type_mapper v1.6.0) for the combined h5ad.
# Output: per-bead annotation tables and Seurat objects carrying FinalID.
#
# The 318 Allen subclasses are consolidated into eleven broad categories
# (FinalID). The mapping is written out subclass by subclass rather than derived
# from name patterns, so that every assignment can be audited.
#
# A second column, hippocampal_region, isolates the hippocampal principal-cell
# subclasses. CA1, CA2, CA3 and DG are therefore cell-type assignments, not
# dissected regions, and beads carrying them are distributed across the section.
# ---------------------------------------------------------------------------

suppressPackageStartupMessages({
  library(Seurat)
})

root    <- "path/to/slide_seq_project"
out_dir <- file.path(root, "out")
obj_dir <- file.path(out_dir, "objects")
ann_dir <- file.path(out_dir, "annotation")
dir.create(ann_dir, recursive = TRUE, showWarnings = FALSE)

samples <- c("J11", "J12", "J21", "J22", "J23")

mmc_file <- list.files(out_dir, pattern = "CorrelationMapping.*\\.csv$",
                       full.names = TRUE)[1]
message("MapMyCells file: ", basename(mmc_file))
mmc <- read.csv(mmc_file, comment.char = "#")
message("rows: ", nrow(mmc))

# --- FinalID consolidation, by Allen subclass -------------------------------
subclass_to_final <- c(
  # Astro-Epen
  "316 Bergmann NN"       = "Astrocytes",
  "317 Astro-CB NN"       = "Astrocytes",
  "318 Astro-NT NN"       = "Astrocytes",
  "319 Astro-TE NN"       = "Astrocytes",
  "320 Astro-OLF NN"      = "Astrocytes",
  "321 Astroependymal NN" = "Ependymal",
  "322 Tanycyte NN"       = "Ependymal",
  "323 Ependymal NN"      = "Ependymal",
  "324 Hypendymal NN"     = "Ependymal",
  "325 CHOR NN"           = "Ependymal",
  # OPC-Oligo
  "326 OPC NN"            = "Oligodendrocytes",
  "327 Oligo NN"          = "Oligodendrocytes",
  # OEC
  "328 OEC NN"            = "OEC",
  # Vascular
  "329 ABC NN"            = "Vascular_cells",
  "330 VLMC NN"           = "Vascular_cells",
  "331 Peri NN"           = "Vascular_cells",
  "332 SMC NN"            = "Vascular_cells",
  "333 Endo NN"           = "Endothelial_cells",
  # Immune
  "334 Microglia NN"      = "Microglia",
  "335 BAM NN"            = "Immune_cells",
  "336 Monocytes NN"      = "Immune_cells",
  "337 DC NN"             = "Immune_cells",
  "338 Lymphoid NN"       = "Immune_cells"
)

mmc$FinalID <- unname(subclass_to_final[mmc$subclass_name])

# Neuronal classes are resolved from the class label suffix.
neuronal <- is.na(mmc$FinalID)
mmc$FinalID[neuronal & grepl("Glut$", mmc$class_name)] <- "Glutamatergic_Neurons"
mmc$FinalID[neuronal & grepl("GABA",  mmc$class_name)] <- "GABAergic_Neurons"
mmc$FinalID[is.na(mmc$FinalID) &
              grepl("(Dopa|Sero|Chol|Nora|Hist)$", mmc$class_name)] <- "Other_Neurons"

stopifnot(!any(is.na(mmc$FinalID)))

# --- Hippocampal principal-cell subclasses ----------------------------------
hippo <- c(
  "016 CA1-ProS Glut" = "CA1",
  "025 CA2-FC-IG Glut" = "CA2",
  "017 CA3 Glut"       = "CA3",
  "037 DG Glut"        = "DG",
  "036 HPF CR Glut"    = "HPF_CR"
)
mmc$hippocampal_region <- unname(hippo[mmc$subclass_name])
mmc$hippocampal_region[is.na(mmc$hippocampal_region)] <- "Other"

mmc$sample  <- sub("_.*$", "", mmc$cell_id)
mmc$barcode <- sub("^[^_]+_", "", mmc$cell_id)

write.csv(mmc, file.path(ann_dir, "mapmycells_annotation_all_beads.csv"),
          row.names = FALSE)

message("\n===== FinalID x sample =====")
print(table(mmc$FinalID, mmc$sample))

message("\n===== hippocampal_region x sample =====")
print(table(mmc$hippocampal_region, mmc$sample))

message("\n===== subclass correlation coefficient, median per sample =====")
print(round(tapply(mmc$subclass_correlation_coefficient, mmc$sample, median), 4))

message("\n===== subclass correlation coefficient, median per FinalID =====")
print(round(tapply(mmc$subclass_correlation_coefficient, mmc$FinalID, median), 4))

# --- Attach to the Seurat objects -------------------------------------------
for (s in samples) {
  f <- file.path(obj_dir, paste0(s, "_seurat.rds"))
  obj <- readRDS(f)
  a <- mmc[mmc$sample == s, ]
  rownames(a) <- a$barcode
  a <- a[colnames(obj), ]
  stopifnot(identical(rownames(a), colnames(obj)))

  for (cn in c("class_name", "subclass_name", "supertype_name", "cluster_name",
               "class_correlation_coefficient", "subclass_correlation_coefficient",
               "FinalID", "hippocampal_region")) {
    obj[[cn]] <- a[[cn]]
  }

  saveRDS(obj, f)
  message(s, ": annotation attached (", ncol(obj), " beads, ",
          length(unique(obj$FinalID)), " FinalID categories)")
  rm(obj, a); gc(verbose = FALSE)
}

message("\ndone")
