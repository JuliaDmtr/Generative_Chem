"""
chembl_graph_dataset.py

Wraps ChemblProcessor (existing SMILES loading/caching code) into a
PyTorch Geometric Dataset that converts SMILES -> graph, filtering
out invalid SMILES once up front so training loops stay simple.
"""

from torch_geometric.data import Dataset

from graph_vae.smiles_to_graph import smiles_to_graph
from smiles_vae.chembl_processor import ChemblProcessor  # your existing class


class ChemblGraphDataset(Dataset):
    def __init__(self, num_samples: int, max_len_of_sample: int = 100,
                 data_path_prefix: str = '../../Datasets/'):
        super().__init__()

        processor = ChemblProcessor(data_path_prefix=data_path_prefix)
        raw_smiles = processor.make_samples(num_samples, max_len_of_sample)

        # Pre-validate: convert once here to drop unparsable SMILES,
        # so __getitem__ stays simple and fast during training.
        self.graphs = []
        skipped = 0
        for smi in raw_smiles:
            g = smiles_to_graph(smi)
            if g is not None:
                self.graphs.append(g)
            else:
                skipped += 1

        print(f"ChemblGraphDataset: {len(self.graphs)} valid graphs, "
              f"{skipped} skipped (invalid SMILES)")

    def len(self):
        return len(self.graphs)

    def get(self, idx):
        return self.graphs[idx]


if __name__ == "__main__":
    # Quick manual test
    ds = ChemblGraphDataset(num_samples=1000, max_len_of_sample=100)
    print(f"Dataset size: {len(ds)}")
    print(f"First graph: {ds[0]}")