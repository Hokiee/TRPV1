"""
Scaffold-based dataset splitting utilities.

This module provides functions for:
- Bemis-Murcko scaffold extraction
- Scaffold-based train/test splitting (DeepChem-style)
- Balanced scaffold splitting to prevent data leakage
"""

import random
import logging
from collections import defaultdict
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from .config import SPLIT_SIZES, RANDOM_SEED


# ============================================================================
# Scaffold Extraction
# ============================================================================

def get_bemis_murcko_scaffold(smiles, include_chirality=False):
    """
    Extract Bemis-Murcko scaffold from SMILES.

    The Bemis-Murcko scaffold is the core ring system + linker atoms,
    with all side chains removed.

    Args:
        smiles: Input SMILES string
        include_chirality: Whether to preserve chirality in scaffold (default: False)

    Returns:
        Scaffold SMILES string, or None if extraction fails
    """
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return None

    try:
        scaffold_smiles = MurckoScaffold.MurckoScaffoldSmiles(
            mol=mol,
            includeChirality=include_chirality
        )
        return scaffold_smiles
    except Exception:
        return None


def build_scaffold_to_indices(smiles_list):
    """
    Build mapping from scaffold to molecule indices.

    Args:
        smiles_list: List of SMILES strings

    Returns:
        Dictionary mapping scaffold SMILES to set of indices
    """
    scaffold_map = defaultdict(set)

    for idx, smiles in enumerate(smiles_list):
        scaffold = get_bemis_murcko_scaffold(smiles, include_chirality=False)
        if scaffold is not None:
            scaffold_map[scaffold].add(idx)

    return scaffold_map


# ============================================================================
# Scaffold Splitting
# ============================================================================

def scaffold_split(smiles_list, sizes=SPLIT_SIZES, seed=RANDOM_SEED, balanced=True):
    """
    Perform scaffold-based train/test split.

    This ensures molecules with the same scaffold stay in the same split,
    preventing data leakage. Uses DeepChem-style balanced splitting.

    Algorithm:
    1. Group molecules by scaffold
    2. Sort scaffolds by size (balanced mode: separate big/small scaffolds)
    3. Greedily assign scaffolds to train/test to match target ratio
    4. Ensure no scaffold overlap between splits

    Args:
        smiles_list: List of SMILES strings
        sizes: Tuple of (train_ratio, test_ratio), e.g., (0.8, 0.2)
        seed: Random seed for reproducibility
        balanced: If True, use balanced splitting (shuffle big/small separately)

    Returns:
        Tuple of (train_indices, test_indices)
    """
    # Validate split sizes
    assert abs(sum(sizes) - 1.0) < 1e-6, f"Sizes must sum to 1.0, got {sum(sizes)}"

    n_total = len(smiles_list)
    n_train_target = int(round(sizes[0] * n_total))

    # Build scaffold mapping
    scaffold_map = build_scaffold_to_indices(smiles_list)

    # Get scaffold groups as list of index sets
    scaffold_sets = list(scaffold_map.values())

    # --- Ordering logic (DeepChem-style) ---
    if balanced:
        # Separate "big" and "small" scaffolds
        # Big = more than half of test set size
        threshold = n_total * sizes[1] / 2
        big_scaffolds = [s for s in scaffold_sets if len(s) > threshold]
        small_scaffolds = [s for s in scaffold_sets if len(s) <= threshold]

        # Shuffle each group independently
        rng = random.Random(seed)
        rng.shuffle(big_scaffolds)
        rng.shuffle(small_scaffolds)

        # Process big scaffolds first, then small
        scaffold_sets = big_scaffolds + small_scaffolds
    else:
        # Simple sorting by size (largest first)
        scaffold_sets = sorted(scaffold_sets, key=len, reverse=True)

    # --- Greedy assignment ---
    train_indices = []
    test_indices = []

    for scaffold_set in scaffold_sets:
        # Add to train if it doesn't exceed target, otherwise add to test
        if len(train_indices) + len(scaffold_set) <= n_train_target:
            train_indices.extend(scaffold_set)
        else:
            test_indices.extend(scaffold_set)

    # --- Validation checks ---
    # Ensure no overlap
    assert len(set(train_indices) & set(test_indices)) == 0, "Train/test overlap detected!"

    # Check split size deviation
    actual_train_ratio = len(train_indices) / n_total
    deviation = abs(actual_train_ratio - sizes[0])
    if deviation > 0.05:
        logging.warning(
            f"Train size deviates by {deviation*100:.1f}% from target "
            f"({actual_train_ratio:.2%} vs {sizes[0]:.2%})"
        )

    return train_indices, test_indices


# ============================================================================
# Validation Utilities
# ============================================================================
def validate_split(train_df, test_df, label_col):
    """
    Validate train/test split.

    Checks:
    1. Both splits contain at least 2 classes
    2. No molecule overlap between splits (by InChIKey)
    3. Logs class distributions

    Args:
        train_df: Training DataFrame
        test_df: Test DataFrame
        label_col: Column name containing class labels (e.g., 'CLASS')

    Raises:
        ValueError: If validation fails
    """
    from collections import Counter

    # Check class presence in each split
    for df, name in [(train_df, "train"), (test_df, "test")]:
        class_counts = Counter(df[label_col])
        logging.info(f"{name.capitalize()} class distribution: {dict(class_counts)}")

        if len(class_counts) < 2:
            raise ValueError(
                f"{name.capitalize()} set has only one class: {class_counts}. "
                f"Try a different random seed."
            )

    # Check for molecule overlap (data leakage check) using InChIKey
    if "InChIKey" not in train_df.columns or "InChIKey" not in test_df.columns:
        raise ValueError("InChIKey column missing. Run Step 1 standardization first.")

    train_ids = set(train_df["InChIKey"].dropna().astype(str))
    test_ids = set(test_df["InChIKey"].dropna().astype(str))
    overlap = train_ids & test_ids

    if overlap:
        raise ValueError(f"Train/test molecule overlap detected: {len(overlap)} shared InChIKeys")

    logging.info("Split validation passed: both splits have all classes, no molecule overlap by InChIKey")

