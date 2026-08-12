# ---------------------------------------------------------------------------
# 08b - Combined canonical-marker heatmap (rebuild)
#
# Validation of the MapMyCells assignments against canonical marker genes.
#
# AverageExpression returns cell-type names with underscores replaced by dots.
# Names are normalised here before plotting, otherwise the factor releveling in
# script 08 drops every category whose name contains an underscore.
# ---------------------------------------------------------------------------

suppressPackageStartupMessages({
  library(ggplot2)
})

root    <- "path/to/slide_seq_project"
fig_dir <- file.path(root, "out", "figures")

comb <- as.matrix(read.csv(file.path(fig_dir, "F5_marker_mean_expression.csv"),
                           row.names = 1, check.names = FALSE))
colnames(comb) <- gsub("\\.", "_", colnames(comb))

marker_order <- c("Slc17a7", "Camk2a", "Gad1", "Gad2", "Aqp4", "Gja1",
                  "Mbp", "Plp1", "Foxj1", "Ttr", "Cx3cr1", "P2ry12",
                  "Ptprc", "Cldn5", "Pecam1", "Pdgfrb", "Acta2")
final_order <- c("Glutamatergic_Neurons", "GABAergic_Neurons", "Other_Neurons",
                 "Astrocytes", "Oligodendrocytes", "OEC", "Ependymal",
                 "Microglia", "Immune_cells", "Endothelial_cells",
                 "Vascular_cells")

stopifnot(all(final_order %in% colnames(comb)))

z <- t(scale(t(comb)))
df <- expand.grid(gene = rownames(z), celltype = colnames(z),
                  stringsAsFactors = FALSE)
df$z <- as.vector(z)
df$gene     <- factor(df$gene, levels = intersect(marker_order, rownames(z)))
df$celltype <- factor(df$celltype, levels = rev(final_order))
df <- df[!is.na(df$gene) & !is.na(df$celltype), ]

ph <- ggplot(df, aes(gene, celltype, fill = z)) +
  geom_tile(colour = "white", linewidth = 0.4) +
  scale_fill_gradient2(low = "#2166ac", mid = "white", high = "#b2182b",
                       midpoint = 0, name = "z-score") +
  labs(x = NULL, y = NULL,
       title = "Canonical marker expression by MapMyCells cell type",
       subtitle = "mean SCT expression across the five animals, scaled per gene") +
  theme_minimal(base_size = 11) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1, face = "italic"),
        panel.grid = element_blank())

ggsave(file.path(fig_dir, "F5_COMBINED_marker_heatmap.png"), ph,
       width = 9, height = 5, dpi = 300)

# Report, per marker, which cell type shows the highest scaled expression.
expected <- c(Slc17a7 = "Glutamatergic_Neurons", Camk2a = "Glutamatergic_Neurons",
              Gad1 = "GABAergic_Neurons", Gad2 = "GABAergic_Neurons",
              Aqp4 = "Astrocytes", Gja1 = "Astrocytes",
              Mbp = "Oligodendrocytes", Plp1 = "Oligodendrocytes",
              Foxj1 = "Ependymal", Ttr = "Ependymal",
              Cx3cr1 = "Microglia", P2ry12 = "Microglia",
              Ptprc = "Immune_cells", Cldn5 = "Endothelial_cells",
              Pecam1 = "Endothelial_cells", Pdgfrb = "Vascular_cells",
              Acta2 = "Vascular_cells")

res <- do.call(rbind, lapply(names(expected), function(g) {
  if (!g %in% rownames(z)) return(NULL)
  top <- colnames(z)[which.max(z[g, ])]
  data.frame(marker = g, expected = expected[[g]], highest_in = top,
             z_top = round(max(z[g, ]), 2),
             z_expected = round(z[g, expected[[g]]], 2),
             match = ifelse(top == expected[[g]], "yes", "no"))
}))

write.csv(res, file.path(fig_dir, "F5_marker_validation.csv"), row.names = FALSE)
print(res, row.names = FALSE)
message("\nmarkers highest in the expected cell type: ",
        sum(res$match == "yes"), " of ", nrow(res))
