"""Create synthetic expression for installation checks, NOT a biological benchmark."""
import argparse
import json
from pathlib import Path
import anndata as ad
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--resources', required=True)
p.add_argument('--output', required=True)
a = p.parse_args()
out = Path(a.output)
if out.exists():
    raise FileExistsError(out)
resources = json.loads(Path(a.resources).read_text(encoding='utf-8'))
genes = sorted(resources['tokens'])[:256]
if len(genes) < 10:
    raise ValueError('At least 10 valid genes required')
counts = np.random.default_rng(42).poisson(3, (3, len(genes))).astype(np.int32)
obj = ad.AnnData(csr_matrix(counts),
    obs=pd.DataFrame(index=['synthetic_1', 'synthetic_2', 'synthetic_3']),
    var=pd.DataFrame(index=genes))
obj.uns['warning'] = 'Synthetic data: installation test only; no biological interpretation.'
out.parent.mkdir(parents=True, exist_ok=True)
obj.write_h5ad(out)
print(out)
