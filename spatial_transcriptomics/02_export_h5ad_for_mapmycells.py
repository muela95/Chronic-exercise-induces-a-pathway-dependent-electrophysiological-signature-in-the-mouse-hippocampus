"""
02 - Export h5ad files for MapMyCells (Allen Institute).

MapMyCells requires a cells x genes matrix. Raw counts are preferred, and the
portal accepts gene symbols (it performs its own identifier conversion). Symbols
are used here rather than Ensembl IDs because only 63.9% of the symbols in this
dataset map to Ensembl through org.Mm.eg.db, which would silently discard a
third of the genes.

Two outputs are written:
  - one combined file (all five animals, barcodes prefixed by sample) for a
    single upload
  - five per-sample files, as a fallback if the combined file is rejected

Bead filtering matches 01_build_objects.R: beads with zero counts are dropped,
so the annotations line up 1:1 with the Seurat objects.

Portal limit is 2 GB per file.
"""

import numpy as np
import pandas as pd
import anndata as ad
from pathlib import Path
from scipy.io import mmread
from scipy.sparse import csr_matrix

ROOT = Path(r"path/to/slide_seq_project")
OUT = ROOT / "out" / "mapmycells_input"
OUT.mkdir(parents=True, exist_ok=True)

GROUP = {"J11": "SED", "J12": "SED", "J21": "RUN", "J22": "RUN", "J23": "RUN"}

adatas = {}

for s, grp in GROUP.items():
    d = ROOT / s.lower()
    print(f"\n===== {s} ({grp}) =====", flush=True)

    # Seeker writes genes x beads; AnnData wants cells x genes.
    mat = mmread(d / f"{s}_MoleculesPerMatchedBead.mtx")
    genes = pd.read_csv(d / f"{s}_genes.tsv", header=None)[0].tolist()
    barcodes = pd.read_csv(d / f"{s}_barcodes.tsv", header=None)[0].tolist()
    X = csr_matrix(mat.T)
    print(f"matrix: {X.shape[0]} beads x {X.shape[1]} genes, {X.nnz} nonzero")

    obs = pd.DataFrame(
        {"sample": s, "group": grp, "barcode": barcodes},
        index=[f"{s}_{b}" for b in barcodes],
    )
    var = pd.DataFrame(index=pd.Index(genes, name="gene_symbol"))

    a = ad.AnnData(X=X, obs=obs, var=var)

    # Same filter as the Seurat build: drop zero-count beads.
    counts = np.asarray(a.X.sum(axis=1)).ravel()
    keep = counts > 0
    if (~keep).sum():
        print(f"dropping {(~keep).sum()} zero-count beads")
    a = a[keep].copy()

    a.write_h5ad(OUT / f"{s}_for_mapmycells.h5ad", compression="gzip")
    size_mb = (OUT / f"{s}_for_mapmycells.h5ad").stat().st_size / 1e6
    print(f"written {s}_for_mapmycells.h5ad ({size_mb:.0f} MB)")

    adatas[s] = a

print("\n===== combined =====", flush=True)
combined = ad.concat(list(adatas.values()), join="outer", fill_value=0)
combined.X = csr_matrix(combined.X)
print(f"combined: {combined.shape[0]} beads x {combined.shape[1]} genes, "
      f"{combined.X.nnz} nonzero")
print(combined.obs["sample"].value_counts().to_string())

combined_path = OUT / "ALL_for_mapmycells.h5ad"
combined.write_h5ad(combined_path, compression="gzip")
print(f"written {combined_path.name} "
      f"({combined_path.stat().st_size / 1e6:.0f} MB, limit 2000 MB)")
