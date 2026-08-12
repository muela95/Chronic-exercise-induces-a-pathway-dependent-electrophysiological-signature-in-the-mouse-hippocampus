# ---------------------------------------------------------------------------
# 01 - Build Seurat objects from Curio Seeker primary output
#
# SCTransform normalisation, PCA on 20 components, SNN graph, UMAP and t-SNE,
# Louvain clustering at resolution 0.4.
#
# Five animals, two conditions:
#   SED (sedentary)          = J11, J12
#   RUN (moderate exercise)  = J21, J22, J23
#
# Gene symbols are kept in their original case so they map to org.Mm.eg.db
# downstream. Raw counts are retained in the "Spatial" assay for the
# animal-level pseudobulk analysis.
# ---------------------------------------------------------------------------

suppressPackageStartupMessages({
  library(Seurat)
  library(Matrix)
})

set.seed(1234)

root    <- "path/to/slide_seq_project"
out_dir <- file.path(root, "out")
obj_dir <- file.path(out_dir, "objects")
dir.create(obj_dir, recursive = TRUE, showWarnings = FALSE)

samples <- c("J11", "J12", "J21", "J22", "J23")
group   <- c(J11 = "SED", J12 = "SED", J21 = "RUN", J22 = "RUN", J23 = "RUN")

# Preprocessing parameters ---------------------------------------------------
N_PCS      <- 20
RESOLUTION <- 0.4
# Seurat 5 defaults to SCTransform v2. v1 is pinned so that normalisation
# matches the Seurat 4.1.3 behaviour used when the dataset was first processed.
VST_FLAVOR <- "v1"

qc <- list()

for (s in samples) {
  message("\n========== ", s, " (", group[[s]], ") ==========")
  d <- file.path(root, tolower(s))

  counts <- readMM(file.path(d, paste0(s, "_MoleculesPerMatchedBead.mtx")))
  rownames(counts) <- readLines(file.path(d, paste0(s, "_genes.tsv")))
  colnames(counts) <- readLines(file.path(d, paste0(s, "_barcodes.tsv")))
  counts <- as(counts, "CsparseMatrix")

  positions <- read.csv(file.path(d, paste0(s, "_MatchedBeadLocation.csv")),
                        row.names = 1)
  colnames(positions) <- c("x", "y")
  positions <- positions[colnames(counts), ]

  obj <- CreateSeuratObject(counts = counts, project = paste0("SlideSeq_", s),
                            assay = "Spatial")
  obj[["image"]] <- new("SlideSeq", assay = "Spatial", coordinates = positions)

  obj$sample <- s
  obj$group  <- group[[s]]
  obj$percent.mt <- PercentageFeatureSet(obj, pattern = "^mt-", assay = "Spatial")

  # Beads with zero counts carry no information and break normalisation.
  n_before <- ncol(obj)
  obj <- subset(obj, subset = nCount_Spatial > 0)
  message("beads: ", n_before, " -> ", ncol(obj), " (zero-count removed: ",
          n_before - ncol(obj), ")")

  qc[[s]] <- data.frame(
    sample            = s,
    group             = group[[s]],
    beads_input       = n_before,
    beads_retained    = ncol(obj),
    genes_detected    = nrow(obj),
    median_umi        = median(obj$nCount_Spatial),
    median_genes      = median(obj$nFeature_Spatial),
    median_percent_mt = round(median(obj$percent.mt), 2),
    mean_percent_mt   = round(mean(obj$percent.mt), 2),
    total_umi         = sum(obj$nCount_Spatial)
  )

  obj <- SCTransform(obj, assay = "Spatial", vst.flavor = VST_FLAVOR,
                     verbose = FALSE)
  obj <- RunPCA(obj, npcs = N_PCS, verbose = FALSE)
  obj <- RunUMAP(obj, dims = 1:N_PCS, verbose = FALSE)
  obj <- RunTSNE(obj, dims = 1:N_PCS, check_duplicates = FALSE)
  obj <- FindNeighbors(obj, dims = 1:N_PCS, verbose = FALSE)
  obj <- FindClusters(obj, resolution = RESOLUTION, verbose = FALSE)

  message("clusters found: ", nlevels(obj$seurat_clusters))

  saveRDS(obj, file.path(obj_dir, paste0(s, "_seurat.rds")))
  rm(obj, counts); gc(verbose = FALSE)
}

qc_tab <- do.call(rbind, qc)
write.csv(qc_tab, file.path(out_dir, "QC_table_per_sample.csv"), row.names = FALSE)

message("\n===== QC SUMMARY =====")
print(qc_tab, row.names = FALSE)

message("\nsessionInfo:")
print(sessionInfo())
