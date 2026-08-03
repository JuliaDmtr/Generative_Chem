
import gc
import pandas as pd
from sklearn.model_selection import train_test_split 
from sklearn.utils import shuffle
import random

class ChemblProcessor:
    def __init__(self, data_path_prefix='../../Datasets/'):
        self.data_path_prefix = data_path_prefix
        self.start_char = '$'
        self.end_char = 'E'
        self.padding_char = None
        self.pad_idx = None
        self.char_to_int = None
        self.int_to_char = None
        self.unique_chars = None
        self.max_smile_length = None
        self.column_names = None
        self.cache_file = f'{data_path_prefix}chembl_canonical_cache.txt'
    def make_samples(self, num_samples: int, max_len_of_sample: int) -> list:
        """
        Docstring for samples
        Sample number of molecules from dataset 
        input: number of samples, dataset
        output: list of sampled SMILES strings
        """
        with open(self.cache_file, 'r') as f:
            chembl_canonical = set(line.strip() for line in f)
            print(f"   ✅ Loaded {len(chembl_canonical):,} from cache")
        # shufle dataset        
        chembl_canonical = shuffle(list(chembl_canonical   ), random_state=0)
        # select num_samples from dataset
        samples = [s for s in chembl_canonical if len(s) <= max_len_of_sample][:num_samples]
        # determine the maximum length of the sampled SMILES strings
        self.max_smile_length = max(len(s) for s in samples)
        # determine the unique characters in the sampled SMILES strings
        self.unique_chars = set(''.join(samples))

        print(sorted(self.unique_chars))
        return samples
    
    def prepare_data_for_lstm(self, dataset_filtered: list) -> tuple[list, list]:
        if self.start_char in self.unique_chars or self.end_char in self.unique_chars:
            raise ValueError("Start and end characters must not be present in the dataset.")

        # Add special tokens
        self.unique_chars = set(self.unique_chars)
        self.unique_chars.add(self.start_char)
        self.unique_chars.add(self.end_char)

        # Pick a pad char (or you can just use a dedicated token like "<PAD>")
        padding_char = '<PAD>'
        while padding_char in self.unique_chars:
            padding_char = chr(ord(padding_char) + 1)
        self.padding_char = padding_char
        self.unique_chars.add(padding_char)
        ordered_chars = sorted(self.unique_chars)

        print("unique chars after adding special tokens: ", ordered_chars)
        print("number of unique chars after adding special tokens: ", len(ordered_chars))
        print("padding char: ", self.padding_char)
        

        self.char_to_int = {c: i for i, c in enumerate(ordered_chars)}
        self.int_to_char = {i: c for i, c in enumerate(ordered_chars)}
        self.pad_idx = self.char_to_int[self.padding_char]
        print("pad index: ", self.pad_idx)
        # IMPORTANT: do NOT pad here; just add start/end
        seqs = []
        for smile in dataset_filtered:
            if isinstance(smile, list):
                smile = ''.join(smile)
            seqs.append(self.start_char + smile + self.end_char)

        seqs = shuffle(seqs, random_state=0)
        training_data, testing_data = train_test_split(seqs, test_size=0.1, random_state=25)
        return training_data, testing_data