import os
import os
from random import random
import sys

from rdkit import Chem

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
import random
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from smiles_diffusion.conditional_smiles_denoiser import (
    ConditionalSmilesAutoregModel,
    build_generation_positions,
    build_smiles_vocab,
    corrupt_tokens,
    decode_logits_to_token_ids,
    decode_token_ids,
    decode_token_ids_with_specials,
    denoise_step,
    diffuse_tokens,
    diffuse_tokens_with_timestep,
    encode_smiles,
    filter_candidate_token_ids_by_syntax,
    is_smiles_syntax_plausible,
    sample_token_ids_from_logits,
    tokenize_smiles,
)
from smiles_vae.chembl_processor import ChemblProcessor

PARP_CSV_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "Datasets", "PARP_inhibitors_July26", "merged_parp_inhibitors.csv"))
CHEMBL_CACHE_PREFIX = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "Datasets")) + os.sep
MAX_LEN = 100
COND_DIM = 64
BATCH_SIZE = 16
EPOCHS = 100
MASK_IDX = 1
PAD_IDX = 0
EOS_IDX = 2


class SmilesDataset(Dataset):
    def __init__(self, positive_smiles, negative_smiles, vocab, positive_label=1.0, negative_label=0.0):
        self.vocab = vocab
        self.samples = []
        for smiles in positive_smiles:
            self.samples.append((smiles, positive_label))
        for smiles in negative_smiles:
            self.samples.append((smiles, negative_label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        smiles, label = self.samples[idx]
        tokens = encode_smiles(smiles, self.vocab, max_len=MAX_LEN, pad_token=self.vocab[0], mask_token=self.vocab[MASK_IDX], append_eos=True, eos_token=self.vocab[EOS_IDX])
        corrupted_tokens, _ = corrupt_tokens(tokens, mask_idx=MASK_IDX, vocab_size=len(self.vocab), corruption_prob=0.25, random_prob=0.25)
        condition = torch.zeros(COND_DIM, dtype=torch.float32)
        condition[0] = float(label)
        return torch.tensor(corrupted_tokens, dtype=torch.long), torch.tensor(tokens, dtype=torch.long), condition


def build_autoregressive_targets(token_ids, pad_idx):
    """Create next-token targets for an autoregressive model."""
    inputs = token_ids[:-1]
    targets = token_ids[1:]
    if len(inputs) < 1:
        return [pad_idx], [pad_idx]
    return inputs, targets


def build_denoising_autoregressive_targets(corrupted_token_ids, target_token_ids, pad_idx):
    """Create autoregressive targets from a corrupted prefix and the clean target sequence."""
    if len(corrupted_token_ids) != len(target_token_ids):
        raise ValueError("Corrupted and target token sequences must have the same length")
    inputs = corrupted_token_ids[:-1]
    targets = target_token_ids[1:]
    if len(inputs) < 1:
        return [pad_idx], [pad_idx]
    return inputs, targets


def pad_sequence_batch(sequences, pad_idx):
    """Pad a batch of variable-length token sequences to the same length."""
    max_len = max(len(seq) for seq in sequences)
    padded = torch.full((len(sequences), max_len), pad_idx, dtype=torch.long)
    padding_masks = torch.zeros((len(sequences), max_len), dtype=torch.bool)
    for row_idx, seq in enumerate(sequences):
        if len(seq) > 0:
            seq_tensor = torch.tensor(seq, dtype=torch.long)
            padded[row_idx, : len(seq)] = seq_tensor
            padding_masks[row_idx, : len(seq)] = False
            padding_masks[row_idx, len(seq) :] = True
    return padded, padding_masks


def sample_next_token(logits, current_tokens, vocab, temperature=0.9, top_k=20, top_p=0.95):
    next_token_logits = logits[0, -1]
    candidate_token_ids = torch.topk(next_token_logits, k=min(top_k, next_token_logits.size(-1))).indices.tolist()
    candidate_token_ids = filter_candidate_token_ids_by_syntax(current_tokens, candidate_token_ids, vocab)
    if not candidate_token_ids:
        return None

    candidate_logits = next_token_logits[candidate_token_ids].float()
    scaled_logits = candidate_logits / max(temperature, 1e-6)
    probs = torch.softmax(scaled_logits, dim=-1)

    if top_p is not None and 0 < top_p < 1.0:
        sorted_probs, sorted_indices = probs.sort(descending=True)
        cumulative_probs = sorted_probs.cumsum(dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 0] = False
        indices_to_remove = torch.zeros_like(probs, dtype=torch.bool)
        indices_to_remove[sorted_indices] = sorted_indices_to_remove
        probs = probs.masked_fill(indices_to_remove, 0.0)
        probs = probs / probs.sum().clamp_min(1e-12)

    sampled_idx = torch.multinomial(probs, num_samples=1).item()
    return candidate_token_ids[sampled_idx]


def split_train_test_smiles(positive_smiles, negative_smiles, test_fraction=0.2, seed=42):
    rng = random.Random(seed)
    pos_list = list(positive_smiles)
    neg_list = list(negative_smiles)
    rng.shuffle(pos_list)
    rng.shuffle(neg_list)

    test_pos_size = max(1, int(round(len(pos_list) * test_fraction)))
    test_neg_size = max(1, int(round(len(neg_list) * test_fraction)))

    test_pos = pos_list[:test_pos_size]
    test_neg = neg_list[:test_neg_size]
    train_pos = pos_list[test_pos_size:]
    train_neg = neg_list[test_neg_size:]
    return train_pos, test_pos, train_neg, test_neg


def train_one_epoch(model, loader, optimizer, pad_idx=PAD_IDX):
    model.train()
    total_loss = 0.0
    for corrupted_ids, target_ids, condition in loader:
        batch_inputs = []
        batch_targets = []
        for corrupted_seq, target_seq in zip(corrupted_ids.tolist(), target_ids.tolist()):
            inputs, targets = build_denoising_autoregressive_targets(corrupted_seq, target_seq, pad_idx=pad_idx)
            batch_inputs.append(inputs)
            batch_targets.append(targets)
        input_ids, input_padding_mask = pad_sequence_batch(batch_inputs, pad_idx=pad_idx)
        target_ids_tensor, _ = pad_sequence_batch(batch_targets, pad_idx=pad_idx)
        logits = model(input_ids, condition, padding_mask=input_padding_mask)
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            target_ids_tensor.view(-1),
            ignore_index=pad_idx,
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def evaluate_model(model, loader, pad_idx=PAD_IDX):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for corrupted_ids, target_ids, condition in loader:
            batch_inputs = []
            batch_targets = []
            for corrupted_seq, target_seq in zip(corrupted_ids.tolist(), target_ids.tolist()):
                inputs, targets = build_denoising_autoregressive_targets(corrupted_seq, target_seq, pad_idx=pad_idx)
                batch_inputs.append(inputs)
                batch_targets.append(targets)
            input_ids, input_padding_mask = pad_sequence_batch(batch_inputs, pad_idx=pad_idx)
            target_ids_tensor, _ = pad_sequence_batch(batch_targets, pad_idx=pad_idx)
            logits = model(input_ids, condition, padding_mask=input_padding_mask)
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                target_ids_tensor.view(-1),
                ignore_index=pad_idx,
            )
            total_loss += loss.item()
    return total_loss / len(loader)


def generate_samples(model, vocab, condition, num_samples=3, max_attempts=8):
    model.eval()
    generated = []
    valid_count = 0
    with torch.no_grad():
        for _ in range(num_samples):
            chosen = None
            for attempt_idx in range(max_attempts):
                cond_tensor = condition.unsqueeze(0)
                current_tokens = [MASK_IDX]
                temperature = 0.8 + 0.1 * attempt_idx
                for _ in range(MAX_LEN - 1):
                    prefix = torch.tensor([current_tokens], dtype=torch.long)
                    logits = model(prefix, cond_tensor)
                    next_token = sample_next_token(logits, current_tokens, vocab, temperature=temperature, top_k=20, top_p=0.95)
                    if next_token is None or next_token in {PAD_IDX, MASK_IDX, EOS_IDX}:
                        break
                    current_tokens.append(next_token)
                    if len(current_tokens) >= MAX_LEN:
                        break
                decoded = decode_token_ids(current_tokens, vocab)
                if not decoded:
                    continue
                if not is_smiles_syntax_plausible(decoded):
                    continue
                try:
                    mol = Chem.MolFromSmiles(decoded)
                except Exception:
                    mol = None
                if mol is not None:
                    chosen = (decoded, True)
                    break
                if decoded and len(decoded) > 8:
                    chosen = (decoded, False)
                    break
            if chosen is None:
                chosen = ("", False)
            generated.append(chosen)
            if chosen[1]:
                valid_count += 1
    return generated, valid_count


def main():
    processor = ChemblProcessor(data_path_prefix=CHEMBL_CACHE_PREFIX)
    parp_df = pd.read_csv(PARP_CSV_PATH)
    parp_smiles = [str(s) for s in parp_df["smiles"].tolist() if isinstance(s, str)]
    n_parp = len(parp_smiles)
    # Load the full cache once and sample a new negative subset at each epoch.
    with open(os.path.join(CHEMBL_CACHE_PREFIX, "chembl_canonical_cache.txt"), "r") as handle:
        all_chembl_smiles = [line.strip() for line in handle if line.strip()]

    print("Building vocabulary...")
    vocab = build_smiles_vocab(parp_smiles + all_chembl_smiles)
    print(vocab)

    model = ConditionalSmilesAutoregModel(vocab_size=len(vocab), max_len=MAX_LEN, cond_dim=COND_DIM)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    history = {"train_loss": [], "test_loss": []}

    for epoch in range(EPOCHS):
        negative_smiles = random.sample(all_chembl_smiles, k=min(n_parp, len(all_chembl_smiles)))
        negative_smiles = [s for s in negative_smiles if len(s) <= MAX_LEN]
        train_pos, test_pos, train_neg, test_neg = split_train_test_smiles(parp_smiles, negative_smiles, test_fraction=0.1, seed=epoch + 42)

        train_dataset = SmilesDataset(train_pos, train_neg, vocab=vocab)
        test_dataset = SmilesDataset(test_pos, test_neg, vocab=vocab)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

        train_loss = train_one_epoch(model, train_loader, optimizer)
        test_loss = evaluate_model(model, test_loader)
        history["train_loss"].append(train_loss)
        history["test_loss"].append(test_loss)
        print(f"epoch={epoch + 1} train_loss={train_loss:.4f} test_loss={test_loss:.4f}")

        if (epoch + 1) % 5 == 0:
            sample_batch = next(iter(test_loader))
            _, target_ids, condition = sample_batch
            target_text = decode_token_ids(target_ids[0].tolist(), vocab)
            with torch.no_grad():
                prefix = torch.tensor([[MASK_IDX]], dtype=torch.long)
                generated_tokens = [MASK_IDX]
                cond_tensor = condition[0:1]
                for _ in range(MAX_LEN - 1):
                    logits = model(prefix, cond_tensor)
                    next_token = sample_next_token(logits, generated_tokens, vocab, temperature=0.9, top_k=20, top_p=0.95)
                    if next_token is None or next_token in {PAD_IDX, MASK_IDX, EOS_IDX}:
                        break
                    generated_tokens.append(next_token)
                    prefix = torch.tensor([generated_tokens], dtype=torch.long)
            predicted_text = decode_token_ids(generated_tokens, vocab)
            print(f"sample target: {target_text}")
            print(f"sample pred  : {predicted_text}")

    generated, valid_count = generate_samples(model, vocab, torch.tensor([1.0] + [0.0] * (COND_DIM - 1), dtype=torch.float32), num_samples=3)
    print(f"generated samples: {generated}")
    print(f"valid smiles count: {valid_count}/{len(generated)}")

    plot_path = os.path.join(ROOT, "train_test_losses.png")
    plt.figure(figsize=(8, 4.5))
    epochs = list(range(1, len(history["train_loss"]) + 1))
    plt.plot(epochs, history["train_loss"], label="train loss", color="tab:blue")
    plt.plot(epochs, history["test_loss"], label="test loss", color="tab:orange")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-entropy loss")
    plt.title("Train/Test Loss Curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    print(f"Saved loss curve plot to {plot_path}")


if __name__ == "__main__":
    main()
