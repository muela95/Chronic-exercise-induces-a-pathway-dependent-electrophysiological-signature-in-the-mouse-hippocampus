"""
09 - Assemble the supplementary enrichment table.

Collects every GSEA result written by script 06 into a single auditable table:
one row per term, carrying the category it came from, the ontology or database,
the enrichment statistics, and the leading-edge genes.

Two outputs:
  - Supplementary_Enrichment_FULL.csv       every term tested
  - Supplementary_Enrichment_SIGNIFICANT.csv  terms with p.adjust < 0.05
  - Supplementary_Enrichment.xlsx           the same, one sheet per source
"""

import pandas as pd
from pathlib import Path
import re

ROOT = Path(r"path/to/slide_seq_project")
GSEA = ROOT / "out" / "gsea"
OUT = ROOT / "out" / "supplementary_tables"
OUT.mkdir(parents=True, exist_ok=True)

rows = []
for f in sorted(GSEA.glob("GSEA_*.csv")):
    m = re.match(r"GSEA_(GOBP|GOMF|GOCC|KEGG)_(.+)\.csv$", f.name)
    if not m:
        continue
    source, category = m.group(1), m.group(2)

    df = pd.read_csv(f)
    df.insert(0, "source", {"GOBP": "GO:BP", "GOMF": "GO:MF",
                            "GOCC": "GO:CC", "KEGG": "KEGG"}[source])
    df.insert(0, "category", category)
    # Split the grouping variable back out of the category label.
    df.insert(0, "grouping", df.category.str.extract(
        r"^(FinalID|hippocampal_region)_")[0])
    df["category"] = df.category.str.replace(
        r"^(FinalID|hippocampal_region)_", "", regex=True)
    rows.append(df)
    print(f"{f.name}: {len(df)} terms")

full = pd.concat(rows, ignore_index=True)

# KEGG descriptions carry a species suffix that adds nothing to the table.
full["Description"] = full.Description.str.replace(
    r"\s*-\s*Mus musculus \(house mouse\)$", "", regex=True)

cols = ["grouping", "category", "source", "ID", "Description", "setSize",
        "enrichmentScore", "NES", "pvalue", "p.adjust", "qvalue",
        "rank", "leading_edge", "core_enrichment"]
cols = [c for c in cols if c in full.columns]
full = full[cols].sort_values(
    ["grouping", "category", "source", "p.adjust"])

full.to_csv(OUT / "Supplementary_Enrichment_FULL.csv", index=False)
sig = full[full["p.adjust"] < 0.05]
sig.to_csv(OUT / "Supplementary_Enrichment_SIGNIFICANT.csv", index=False)

print(f"\nFULL: {len(full)} rows across {full.category.nunique()} categories")
print(f"SIGNIFICANT (p.adjust < 0.05): {len(sig)} rows")
print("\nsignificant terms per category and source:")
print(sig.pivot_table(index="category", columns="source", values="ID",
                      aggfunc="count", fill_value=0).to_string())

try:
    with pd.ExcelWriter(OUT / "Supplementary_Enrichment.xlsx",
                        engine="openpyxl") as xl:
        sig.to_excel(xl, sheet_name="significant_all", index=False)
        for src in full.source.unique():
            sheet = src.replace(":", "_")
            full[full.source == src].to_excel(xl, sheet_name=sheet, index=False)
    print("\nxlsx written")
except Exception as e:
    print(f"\nxlsx skipped ({e}); csv files are complete")
