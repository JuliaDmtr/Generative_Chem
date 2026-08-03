import os
import sys

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from smiles_diffusion.conditional_smiles_denoiser import (
    ConditionalSmilesAutoregModel,
    build_smiles_vocab,
    corrupt_tokens,
    decode_token_ids,
    diffuse_tokens_with_timestep,
    encode_smiles,
    filter_candidate_token_ids_by_syntax,
    is_smiles_syntax_plausible,
    sample_token_ids_from_logits,
    tokenize_smiles,
)
from smiles_diffusion.train_conditional_denoiser import (
    build_denoising_autoregressive_targets,
    split_train_test_smiles,
)


def test_corrupt_tokens_masks_and_preserves_length():
    seq = [1, 2, 3, 4, 5]
    corrupted, mask = corrupt_tokens(seq, mask_idx=99, vocab_size=100, corruption_prob=1.0, random_prob=0.0)

    assert len(corrupted) == len(seq)
    assert len(mask) == len(seq)
    assert all(isinstance(x, int) for x in corrupted)
    assert sum(mask) >= 1


def test_corrupt_tokens_can_replace_with_any_vocab_token():
    seq = [1, 2, 3]
    corrupted, mask = corrupt_tokens(seq, mask_idx=0, vocab_size=5, corruption_prob=1.0, random_prob=1.0)

    assert mask == [1, 1, 1]
    assert all(token in range(5) for token in corrupted)


def test_smiles_vocab_round_trip():
    vocab = build_smiles_vocab(["CCO", "C=O"])
    token_ids = encode_smiles("CCO", vocab, max_len=4)
    decoded = decode_token_ids(token_ids, vocab)

    assert decoded == "CCO"


def test_diffuse_tokens_with_timestep_increases_noise_with_t():
    token_ids = [1, 2, 3, 4]
    corrupted_t0, _ = diffuse_tokens_with_timestep(token_ids, mask_idx=0, vocab_size=6, timestep=0)
    corrupted_t2, _ = diffuse_tokens_with_timestep(token_ids, mask_idx=0, vocab_size=6, timestep=2)

    assert corrupted_t0 == token_ids
    assert corrupted_t2 != token_ids


def test_tokenize_smiles_handles_aromatic_and_bracketed_tokens():
    tokens = tokenize_smiles("Cc1ccccc1[Si](C)C")

    assert tokens[:3] == ["C", "c", "1"]
    assert "[Si]" in tokens
    assert "(" in tokens and ")" in tokens
    assert tokens[-1] == "C"


def test_sample_token_ids_from_logits_respects_top_k_and_top_p():
    logits = torch.tensor([[[5.0, 4.0, 0.5, 0.1]]])

    sampled = sample_token_ids_from_logits(logits, temperature=0.7, top_k=2, top_p=0.9)

    assert sampled.shape == (1, 1)
    assert sampled.item() in {0, 1}


def test_syntax_plausibility_rejects_unbalanced_structure():
    assert is_smiles_syntax_plausible("CCO")
    assert is_smiles_syntax_plausible("C1CC1")
    assert not is_smiles_syntax_plausible("C1CC")
    assert not is_smiles_syntax_plausible("CC(O")
    assert not is_smiles_syntax_plausible("[NH")


def test_autoregressive_model_returns_logits_for_each_position():
    model = ConditionalSmilesAutoregModel(vocab_size=8, max_len=6, cond_dim=4, d_model=16)
    input_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    condition = torch.zeros(4, dtype=torch.float32)

    logits = model(input_ids, condition)

    assert logits.shape == (1, 4, 8)


def test_filter_candidate_token_ids_by_syntax_rejects_unbalanced_prefixes():
    vocab = ["<pad>", "<mask>", "C", "(", ")"]
    prefix_ids = [2, 3]
    candidate_ids = [2, 4]

    filtered = filter_candidate_token_ids_by_syntax(prefix_ids, candidate_ids, vocab)

    assert filtered == [4]


def test_split_train_test_smiles_respects_fraction_and_seed():
    positive = ["A", "B", "C", "D"]
    negative = ["E", "F", "G", "H"]

    train_pos, test_pos, train_neg, test_neg = split_train_test_smiles(
        positive,
        negative,
        test_fraction=0.25,
        seed=0,
    )

    assert len(train_pos) == 3
    assert len(test_pos) == 1
    assert len(train_neg) == 3
    assert len(test_neg) == 1


def test_build_denoising_autoregressive_targets_uses_corrupted_prefix():
    corrupted = [1, 2, 3, 4]
    target = [5, 6, 7, 8]

    inputs, targets = build_denoising_autoregressive_targets(corrupted, target, pad_idx=0)

    assert inputs == [1, 2, 3]
    assert targets == [6, 7, 8]
