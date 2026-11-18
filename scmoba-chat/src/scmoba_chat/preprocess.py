"""Explicit input transforms, sparse/chunked ranking; no external annotation API."""
import hashlib
import json
from pathlib import Path
import re
import numpy as np
from scipy import sparse


def file_hash(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''): h.update(b)
    return h.hexdigest()


def prepare(input_path, output_dir, resources, *, input_scale, gene_column='index',
            gene_type='ensembl', layer='X', min_genes=10, chunk_size=128, max_cells=20000):
    import anndata as ad
    from datasets import Dataset
    if input_scale not in ('counts', 'log1p'): raise ValueError('Choose counts or log1p explicitly')
    if gene_type not in ('ensembl', 'symbol'): raise ValueError('gene_type must be ensembl or symbol')
    if min_genes < 1 or chunk_size < 1: raise ValueError('Invalid QC/chunk size')
    out = Path(output_dir)
    if out.exists(): raise FileExistsError(f'Refusing to overwrite {out}')
    ref = json.loads(Path(resources).read_text())
    a = ad.read_h5ad(input_path, backed='r')
    try:
        if not a.obs_names.is_unique: raise ValueError('Cell IDs must be unique')
        if a.n_obs == 0 or a.n_obs > max_cells: raise ValueError(f'Expected 1..{max_cells} cells; subset large files before upload')
        genes = a.var_names.astype(str).tolist() if gene_column == 'index' else a.var[gene_column].astype(str).tolist()
        ids = [re.sub(r'\.\d+$', '', g) if gene_type == 'ensembl' else ref['symbols'].get(g, '') for g in genes]
        valid = [(i, g) for i, g in enumerate(ids) if g in ref['tokens'] and g in ref['medians'] and ref['medians'][g] > 0]
        if not valid: raise ValueError('No genes mapped; check species, gene column and ID type (human only)')
        # Preserve first-occurrence gene order. Duplicate counts are summed before log1p.
        unique = list(dict.fromkeys(g for _, g in valid)); position = {g:i for i,g in enumerate(unique)}
        if input_scale == 'log1p' and len(unique) != len(valid):
            raise ValueError('Duplicate mapped genes in log1p input: supply counts so duplicates can be summed correctly')
        mapper = sparse.csr_matrix((np.ones(len(valid)), ([i for i,g in valid], [position[g] for i,g in valid])), shape=(a.n_vars,len(unique)))
        med = np.array([ref['medians'][g] for g in unique], dtype=np.float64)
        tokens = np.array([ref['tokens'][g] for g in unique], dtype=np.int64)
        matrix = a.X if layer == 'X' else a.layers[layer]
        if matrix is None: raise ValueError('Selected expression matrix is missing')
        kept=[]; rejected=[]
        for start in range(0,a.n_obs,chunk_size):
            x=sparse.csr_matrix(matrix[start:start+chunk_size],dtype=np.float64)
            x.sum_duplicates(); x.eliminate_zeros()
            if not np.isfinite(x.data).all() or (x.data < 0).any(): raise ValueError('Expression must be finite and nonnegative')
            if input_scale=='counts' and not np.allclose(x.data,np.rint(x.data),rtol=0,atol=1e-6):
                raise ValueError('counts input contains non-integers; choose correct matrix/scale, never silently re-normalize')
            totals=np.asarray(x.sum(axis=1)).ravel()
            mapped=(x @ mapper).tocsr()
            if input_scale=='counts':
                mapped=sparse.diags(np.divide(10000.,totals,out=np.zeros_like(totals),where=totals>0)) @ mapped
                mapped=mapped.tocsr(); mapped.data=np.log1p(mapped.data)
            # Historical scMOBA LangCell tokenizer: expression / gene median, nonzero descending.
            dense=mapped.toarray()/med[None,:]
            for j,values in enumerate(dense):
                nz=np.flatnonzero(values>0); cell=str(a.obs_names[start+j])
                if len(nz)<min_genes:
                    rejected.append({'cell_id':cell,'mapped_expressed_genes':len(nz),'reason':'too_few_mapped_genes'}); continue
                order=np.argsort(-values[nz])  # Match historical NumPy ordering, including ties.
                seq=tokens[nz][order][:2048].tolist()
                kept.append({'cell_id':cell,'input_ids':seq,'length':len(seq),'profile_type':'single_cell'})
        if not kept: raise ValueError('All cells failed mapped-gene QC')
        report={'schema_version':1,'input_sha256':file_hash(input_path),'resources_sha256':file_hash(resources),
                'input_scale':input_scale,'layer':layer,'gene_column':gene_column,'gene_type':gene_type,
                'n_cells':a.n_obs,'kept_cells':len(kept),'rejected_cells':len(rejected),'n_input_genes':a.n_vars,
                'mapped_columns':len(valid),'mapped_unique_genes':len(unique),'mapped_fraction':len(valid)/a.n_vars,
                'min_genes':min_genes,'max_length':2048,'normalization':'log1p(counts/total_all_genes*10000)' if input_scale=='counts' else 'caller-declared log1p normalized expression; no transform',
                'ranking':'expression / training gene median; descending nonzero; no CLS token',
                'warnings':['Human genes only. obs metadata is NOT supplied to model.','Low mapping or targeted/HVG-only matrices can shift model inputs.','Gene-set explanations are predictions, not measured activity.']}
        out.mkdir(parents=True)
        Dataset.from_list(kept).save_to_disk(str(out/'tokens'))
        (out/'qc.json').write_text(json.dumps(report,indent=2))
        (out/'rejected_cells.json').write_text(json.dumps(rejected,indent=2))
        return report
    finally:
        a.file.close()
