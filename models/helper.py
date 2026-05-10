import torch


def find_pruneable_heads_and_indices(
    heads: list[int],
    n_heads: int,
    head_size: int,
    already_pruned_heads: set[int],
) -> tuple[set[int], torch.LongTensor]:
    """
    Finds the heads and the flattened indices to keep, taking already-pruned heads
    into account.

    Parameters
    ----------
    heads : list[int]
        Head indices requested for pruning.
    n_heads : int
        Total number of attention heads before this pruning step.
    head_size : int
        Size of each attention head.
    already_pruned_heads : set[int]
        Heads that were pruned in earlier steps.

    Returns
    -------
    tuple[set[int], torch.LongTensor]
        (new_heads_to_prune, flattened_indices_to_keep)
    """
    mask = torch.ones(n_heads, head_size, dtype=torch.bool)

    heads = set(heads) - already_pruned_heads

    for head in heads:
        # Shift the head index left by however many smaller heads
        # were already removed earlier.
        shifted_head = head - sum(1 for h in already_pruned_heads if h < head)
        mask[shifted_head] = False

    index = torch.arange(n_heads * head_size)[mask.view(-1)].long()
    return heads, index