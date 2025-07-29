from typing import Optional

import torch


def get_downsampled_inter_chain_contacts(
    contacts: torch.Tensor,  # (N, N)
    chain_idx: torch.Tensor,
    min_contacts: int = 1,
    max_contacts: int = 20,
    min_dist: float = 4.0,
    max_dist: float = 7.0,  # prefer true backbone contacts
) -> torch.Tensor:
    """
    In multimers, down-sample interchain contacts to just a few true contacts.

    Assumes contacts have already been masked to valid pairs (e.g. using `res_mask`)
    """
    # Escape if not multimeric
    inter_chain_mask = chain_idx.unsqueeze(-1) != chain_idx.unsqueeze(-2)  # (N, N)
    if not inter_chain_mask.any():
        return contacts

    # Find all valid inter-chain contacts, focusing on actual backbone contacts (default < 7.0) angstroms)
    valid_inter_chain = (contacts > min_dist) & (contacts < max_dist) & inter_chain_mask
    if not valid_inter_chain.any():
        return contacts

    # Take only upper triangle, to de-dupe pairs
    valid_inter_chain = torch.triu(valid_inter_chain)

    # Get coordinates of valid contacts
    contact_coords = valid_inter_chain.nonzero()  # (num_contacts, 2)

    # Target number of contacts to keep (random between 1 and 20)
    num_contacts = torch.randint(min_contacts, max_contacts + 1, (1,)).item()
    num_contacts = min(num_contacts, contact_coords.shape[0])

    # Randomly select a subset of contacts
    if num_contacts < contact_coords.shape[0]:
        selected_indices = torch.randperm(contact_coords.shape[0])[:num_contacts]
        selected_coords = contact_coords[selected_indices]
    else:
        selected_coords = contact_coords

    # Create new contact matrix with only selected contacts
    new_contact_matrix = torch.zeros_like(contacts)

    for coord in selected_coords:
        i, j = coord[0].item(), coord[1].item()
        # Keep both ij and ji for symmetry
        new_contact_matrix[i, j] = contacts[i, j]
        new_contact_matrix[j, i] = contacts[j, i]

    return new_contact_matrix


def get_contact_conditioning_matrix(
    trans: torch.Tensor,  # (N, 3) or (B, N, 3)
    res_mask: torch.Tensor,  # (N,) or (B, N)
    chain_idx: torch.Tensor,  # (N,) or (B, N)
    motif_mask: Optional[torch.Tensor],  # (N,) or (B, N)
    include_inter_chain: bool = False,
    downsample_inter_chain: bool = False,
    downsample_inter_chain_min_contacts: int = 1,
    downsample_inter_chain_max_contacts: int = 20,
    motif_only: bool = False,
    min_res_gap: int = 7,
    min_dist: float = 4.0,
    max_dist: float = 20.0,
    dist_noise_ang: float = 0.2,
) -> torch.Tensor:
    """
    Create a contact conditioning matrix from a set of translations.
    Supports both single batch (N, 3) and batched (B, N, 3) inputs.

    Requires residues are `min_res_gap` apart; the diagonal is removed.
    Adds `dist_noise_ang` * noise to the actual distances.

    In multimers, `include_inter_chain` can be used to include inter-chain contacts.
    In inpainting, constraints can be limited to `motif_only`.

    As an example, if you were trying to design a nanobody against a target,
    you could specify the target as a motif (structure will be defined using guidance),
    and the nanobody with a 2D constraint, so avoid constraining its placement in space.
    To guide its placement, define some inter-chain contacts (or just define hotspots on target).
    """
    assert trans.ndim in [2, 3] and trans.shape[-1] == 3

    # handle batch vs single item
    if trans.ndim == 2:
        single_batch = True

        assert res_mask.ndim == 1
        assert chain_idx.ndim == 1
        assert motif_mask is None or motif_mask.ndim == 1

        # Single batch: (N, 3) -> (1, N, 3)
        trans = trans.unsqueeze(0)
        res_mask = res_mask.unsqueeze(0)
        chain_idx = chain_idx.unsqueeze(0)
        if motif_mask is not None:
            motif_mask = motif_mask.unsqueeze(0)
    elif trans.ndim == 3:
        single_batch = False

        assert res_mask.ndim == 2
        assert chain_idx.ndim == 2
        assert motif_mask is None or motif_mask.ndim == 2

    else:
        raise ValueError(f"Unsupported trans shape: {trans.shape}")

    B, N, _ = trans.shape

    # Compute pairwise distances for each batch
    contact_conditioning = torch.cdist(trans, trans)  # (B, N, N)

    # TODO(conditioning) - allow masking high b factor / low plddt residues
    edge_mask = res_mask.unsqueeze(-1) & res_mask.unsqueeze(-2)  # (B, N, N)

    # identify inter-chain and intra-chain contacts
    inter_chain_mask = chain_idx.unsqueeze(-1) != chain_idx.unsqueeze(-2)  # (B, N, N)
    intra_chain_mask = ~inter_chain_mask.bool()

    # determine actual contacts, limited to min / max dists
    contact_mask = edge_mask & (
        (contact_conditioning < max_dist) & (contact_conditioning > min_dist)
    )
    contact_conditioning = contact_conditioning * contact_mask.float()

    # optionally, add small noise to 2d constraint
    if dist_noise_ang > 0:
        noise = torch.rand(B, N, N, device=trans.device) * dist_noise_ang
        contact_conditioning += noise
        contact_conditioning = (
            contact_conditioning.clamp(min=min_dist, max=max_dist)
            * contact_mask.float()
        )

    # mask closely adjacent residue pairs: |i − j| < min_res_gap
    if min_res_gap > 1:
        res_adjacency_mask = torch.tril(
            torch.ones_like(contact_conditioning[0], dtype=torch.bool),
            diagonal=min_res_gap - 1,
        ) & torch.triu(
            torch.ones_like(contact_conditioning[0], dtype=torch.bool),
            diagonal=-(min_res_gap - 1),
        )  # (N, N)
        # limit res adjacency to intra-chain contacts
        res_adjacency_mask = (
            res_adjacency_mask.unsqueeze(0) & intra_chain_mask
        )  # (B, N, N)
        contact_conditioning = contact_conditioning * (~res_adjacency_mask).float()

    # optionally, limit to motif mask
    if motif_only and motif_mask is not None and motif_mask.any():
        motif_edge_mask = motif_mask.unsqueeze(-1) & motif_mask.unsqueeze(-2)
        contact_conditioning = contact_conditioning * motif_edge_mask

    # optionally, remove inter-chain contacts
    if not include_inter_chain:
        contact_conditioning = contact_conditioning * intra_chain_mask.float()
    # otherwise, optionally downsample inter-chain contacts
    elif downsample_inter_chain:
        # apply downsampling to each batch item independently
        for b in range(B):
            contacts_b = contact_conditioning[b]
            contact_conditioning[b] = get_downsampled_inter_chain_contacts(
                contacts=contacts_b,
                chain_idx=chain_idx[b],
                min_contacts=downsample_inter_chain_min_contacts,
                max_contacts=downsample_inter_chain_max_contacts,
                # Use default min / max dists, to prioritize true backbone contacts
            )

    # Return to original shape if single batch
    if single_batch:
        contact_conditioning = contact_conditioning.squeeze(0)

    return contact_conditioning
