# ---------------------------------------------------------------------------
# 08 - Supplementary figures
#
# Outputs, per sample and combined:
#   F1  UMAP coloured by Louvain cluster
#   F2  UMAP coloured by MapMyCells cell type (FinalID)
#   F3  spatial map of clusters on the tissue
#   F4  spatial map of cell types on the tissue
#   F5  dotplot of canonical markers by FinalID
#   F6  spatial map of hippocampal subfields
# ---------------------------------------------------------------------------

suppressPackageStartupMessages({
  library(Seurat)
  library(ggplot2)
  library(patchwork)
})

root    <- "path/to/slide_seq_project"
obj_dir <- file.path(root, "out", "objects")
fig_dir <- file.path(root, "out", "figures")
dir.create(fig_dir, recursive = TRUE, showWarnings = FALSE)

samples <- c("J11", "J12", "J21", "J22", "J23")
group   <- c(J11 = "SED", J12 = "SED", J21 = "RUN", J22 = "RUN", J23 = "RUN")

# Canonical markers for each annotated category.
markers <- list(
  Glutamatergic_Neurons = c("Slc17a7", "Camk2a"),
  GABAergic_Neurons     = c("Gad1", "Gad2"),
  Astrocytes            = c("Aqp4", "Gja1"),
  Microglia             = c("Cx3cr1", "P2ry12"),
  Oligodendrocytes      = c("Mbp", "Plp1"),
  Endothelial_cells     = c("Cldn5", "Pecam1"),
  Vascular_cells        = c("Pdgfrb", "Acta2"),
  Ependymal             = c("Foxj1", "Ttr"),
  Immune_cells          = c("Ptprc", "Cd74")
)
marker_vec <- unique(unlist(markers))

# Order cell types so the dotplot diagonal is readable.
final_order <- c("Glutamatergic_Neurons", "GABAergic_Neurons", "Other_Neurons",
                 "Astrocytes", "Oligodendrocytes", "OEC", "Ependymal",
                 "Microglia", "Immune_cells", "Endothelial_cells",
                 "Vascular_cells")

dot_data <- list()

for (s in samples) {
  message("\n===== ", s, " =====")
  obj <- readRDS(file.path(obj_dir, paste0(s, "_seurat.rds")))
  ttl <- paste0(s, " (", group[[s]], ")")

  p1 <- DimPlot(obj, reduction = "umap", group.by = "seurat_clusters",
                label = TRUE, label.size = 3, raster = FALSE) +
    ggtitle(paste0(ttl, " - Louvain clusters (res 0.4)")) + NoLegend()
  ggsave(file.path(fig_dir, paste0("F1_UMAP_clusters_", s, ".png")), p1,
         width = 6, height = 6, dpi = 300)

  p2 <- DimPlot(obj, reduction = "umap", group.by = "FinalID", raster = FALSE) +
    ggtitle(paste0(ttl, " - MapMyCells cell types"))
  ggsave(file.path(fig_dir, paste0("F2_UMAP_celltypes_", s, ".png")), p2,
         width = 8, height = 6, dpi = 300)

  p3 <- SpatialDimPlot(obj, group.by = "seurat_clusters", stroke = 0,
                       pt.size.factor = 1.2) +
    ggtitle(paste0(ttl, " - clusters on tissue"))
  ggsave(file.path(fig_dir, paste0("F3_spatial_clusters_", s, ".png")), p3,
         width = 7, height = 6, dpi = 300)

  p4 <- SpatialDimPlot(obj, group.by = "FinalID", stroke = 0,
                       pt.size.factor = 1.2) +
    ggtitle(paste0(ttl, " - cell types on tissue"))
  ggsave(file.path(fig_dir, paste0("F4_spatial_celltypes_", s, ".png")), p4,
         width = 8, height = 6, dpi = 300)

  p6 <- SpatialDimPlot(obj, group.by = "hippocampal_region", stroke = 0,
                       pt.size.factor = 1.2) +
    ggtitle(paste0(ttl, " - hippocampal subfields"))
  ggsave(file.path(fig_dir, paste0("F6_spatial_subfields_", s, ".png")), p6,
         width = 7, height = 6, dpi = 300)

  # Per-sample marker dotplot, on SCT normalised data.
  DefaultAssay(obj) <- "SCT"
  present <- intersect(marker_vec, rownames(obj))
  obj$FinalID_ord <- factor(obj$FinalID,
                            levels = rev(intersect(final_order, unique(obj$FinalID))))
  Idents(obj) <- "FinalID_ord"
  p5 <- DotPlot(obj, features = present) +
    RotatedAxis() +
    ggtitle(paste0(ttl, " - canonical markers by assigned cell type")) +
    theme(axis.text.x = element_text(size = 8))
  ggsave(file.path(fig_dir, paste0("F5_markers_dotplot_", s, ".png")), p5,
         width = 10, height = 5, dpi = 300)

  # Mean expression per cell type, pooled across samples for the combined panel.
  avg <- AverageExpression(obj, features = present, group.by = "FinalID",
                           assays = "SCT", layer = "data")$SCT
  dot_data[[s]] <- avg

  message("figures written")
  rm(obj); gc(verbose = FALSE)
}

# Combined marker heatmap: mean SCT expression per cell type, averaged over the
# five animals.
cats <- Reduce(intersect, lapply(dot_data, colnames))
genes <- Reduce(intersect, lapply(dot_data, rownames))
comb <- Reduce(`+`, lapply(dot_data, function(m) m[genes, cats])) / length(dot_data)

# Scale each gene across cell types so the panel shows specificity.
z <- t(scale(t(comb)))
df <- expand.grid(gene = rownames(z), celltype = colnames(z),
                  stringsAsFactors = FALSE)
df$z <- as.vector(z)
df$gene <- factor(df$gene, levels = intersect(marker_vec, rownames(z)))
df$celltype <- factor(df$celltype, levels = intersect(final_order, colnames(z)))

ph <- ggplot(df, aes(gene, celltype, fill = z)) +
  geom_tile(colour = "white") +
  scale_fill_gradient2(low = "#2166ac", mid = "white", high = "#b2182b",
                       midpoint = 0, name = "z-score") +
  labs(x = NULL, y = NULL,
       title = "Canonical marker expression by MapMyCells cell type",
       subtitle = "mean SCT expression across the five animals, scaled per gene") +
  theme_minimal(base_size = 11) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

ggsave(file.path(fig_dir, "F5_COMBINED_marker_heatmap.png"), ph,
       width = 9, height = 5, dpi = 300)
write.csv(comb, file.path(fig_dir, "F5_marker_mean_expression.csv"))

message("\ndone")
