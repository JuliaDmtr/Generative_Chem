"""
smiles_tensor_dataset.py

Wraps ChemblProcessor's prepare_data_for_lstm() output (SMILES strings with
start/end chars added, but not yet padded/tokenized) into a PyTorch Dataset
that returns fixed-length padded integer tensors, ready for the Transformer VAE.
"""

import torch
from torch.utils.data import Dataset


class SmilesTensorDataset(Dataset):
    def __init__(self, sequences: list, char_to_int: dict, pad_idx: int, max_len: int):
        """
        sequences: list of strings, each already wrapped with start/end chars
                   (output of ChemblProcessor.prepare_data_for_lstm)
        char_to_int: mapping from character -> integer id
        pad_idx: integer id used for padding
        max_len: fixed sequence length to pad/truncate to (should be >= the
                 longest sequence + start/end chars; ChemblProcessor's
                 max_smile_length + 2 is a safe choice)
        """
        self.sequences = sequences
        self.char_to_int = char_to_int
        self.pad_idx = pad_idx
        self.max_len = max_len

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        ids = [self.char_to_int[c] for c in seq]

        if len(ids) > self.max_len:
            ids = ids[: self.max_len]
        else:
            ids = ids + [self.pad_idx] * (self.max_len - len(ids))

        return torch.tensor(ids, dtype=torch.long)