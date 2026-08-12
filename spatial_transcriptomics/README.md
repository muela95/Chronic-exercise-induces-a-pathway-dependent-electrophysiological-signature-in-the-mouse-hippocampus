# Spatial transcriptomics (Slide-seq)

Code for the spatial transcriptomics section. The cohort is independent of the
electrophysiology cohort: five animals, one dentate gyrus section each, two
conditions (SED = J11, J12; RUN = J21, J22, J23).

The analysis starts from the primary output of Curio Seeker and treats the
animal, not the spatial barcode, as the unit of replication.

## Pipeline

Scripts run in numerical order. Each writes to `out/` under the project root.

| Script | Description |
|---|---|
| `01_build_objects.R` | Builds one Seurat object per animal from the Seeker matrices. SCTransform, 20 PCs, SNN graph, UMAP and t-SNE, Louvain clustering at resolution 0.4. Writes the per-sample quality control table |
| `02_export_h5ad_for_mapmycells.py` | Exports h5ad files for cell type annotation, one combined and five per-sample |
| `03_attach_annotation.R` | Attaches the MapMyCells output, consolidates the 318 Allen subclasses into eleven categories and flags the hippocampal principal-cell subclasses |
| `04_pseudobulk.R` | Sums raw counts across all beads of an animal, within each cell type and each hippocampal subfield |
| `05_deseq2.R` | Differential expression, RUN versus SED, on the pseudobulk matrices |
| `06_gsea.R` | GO and KEGG enrichment by GSEA on the ranked gene lists |
| `07_bead_level_comparison.R` | Repeats the comparison at bead level, including a treatment-blind control split |
| `08_figures.R`, `08b_marker_heatmap.R` | UMAPs, spatial maps, and validation of the annotation against canonical markers |
| `09_supplementary_enrichment_table.py` | Assembles the enrichment results into a single table |
| `10_animal_permutation.R` | Exact permutation over the ten possible ways of splitting five animals into groups of two and three |
| `11_leave_one_out_DG.R` | Repeats the dentate gyrus analysis dropping each exercised animal in turn |
| `12_entorhinal_check.py` | Tests whether beads assigned to entorhinal cortex form a compact territory or are scattered |
| `13_export_kegg_membership.R` | Exports full KEGG gene set membership and the exact ranked lists, for redrawing the running enrichment score |
| `14_DG_confidence_threshold.R` | Repeats the dentate gyrus analysis at five annotation-confidence thresholds |

## Requirements

R 4.4.1 with Seurat 5.1.0, sctransform 0.4.1, DESeq2, edgeR, limma,
clusterProfiler, org.Mm.eg.db, fgsea, enrichplot, Matrix, ggplot2 and patchwork.

Python 3.11 with anndata, scipy, pandas, numpy, h5py, matplotlib and openpyxl.

Cell type annotation is performed with MapMyCells (Allen Institute,
cell_type_mapper 1.6.0), taxonomy 10x Whole Mouse Brain CCN20230722, algorithm
Correlation Mapping. Script `02` writes the input; the mapping itself runs on
the Allen portal and its output is read back by script `03`.

## Usage

Set `root` at the top of each script to the folder holding one subdirectory per
animal (`j11`, `j12`, `j21`, `j22`, `j23`), each containing the four Seeker
files: `*_MoleculesPerMatchedBead.mtx`, `*_genes.tsv`, `*_barcodes.tsv` and
`*_MatchedBeadLocation.csv`. Then:

```
Rscript 01_build_objects.R
python  02_export_h5ad_for_mapmycells.py
Rscript 03_attach_annotation.R
Rscript 04_pseudobulk.R
Rscript 05_deseq2.R
Rscript 06_gsea.R
```

Scripts `07` to `14` are independent of each other and read the objects and
pseudobulk matrices written by `01` to `04`.

## Analysis

Gene symbols are kept in the case reported by Seeker and are passed to
MapMyCells as symbols rather than Ensembl identifiers, which maps 35,940 of the
36,086 genes.

Each Seeker library reports only the genes it detected, so the per-sample gene
lists differ (27,113 to 29,697 genes; union 36,086). Pseudobulk counts are
placed in the union space, with undetected genes set to zero. Genes are filtered
at the differential expression step, keeping those with at least 5 counts in at
least 2 animals. Categories with fewer than 20,000 counts in any animal are not
tested, since a dispersion trend cannot be fitted from them.

With two and three animals per group, only ten label assignments exist, so the
smallest attainable permutation p-value is 0.1. Per-gene significance is not the
objective. Results are read as the rank of the real design among the ten splits,
and enrichment is assessed by GSEA over the complete ranked list rather than by
an over-representation test on a list of significant genes.
