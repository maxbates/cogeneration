from cogeneration.type.str_enum import StrEnum


class MetricName(StrEnum):
    """
    Enumeration of metrics we calculate for protein structures, sequences, and trajectories
    """

    # Functions that parse structures / sequences / trajectories

    # ca-ca metrics
    ca_ca_deviation = "ca_ca_deviation"
    ca_ca_valid_percent = "ca_ca_valid_percent"
    num_ca_ca_clashes = "num_ca_ca_clashes"

    # aatype metrics
    aatype_histogram_dist = "aatype_histogram_dist"

    # mdtraj metrics
    non_coil_percent = "non_coil_percent"
    coil_percent = "coil_percent"
    helix_percent = "helix_percent"
    strand_percent = "strand_percent"
    radius_of_gyration = "radius_of_gyration"

    # Assessment pipeline

    # `parse_pdb_feats()` and `parse_chain_feats()` are called when parsing structures
    # see DatasetProteinColumns

    # metadata
    sample_id = "sample_id"  # unique identifier for sample
    sample_length = "sample_length"  # length of sample

    # sequence + structure information
    header = "header"  # sequence name, header in fasta file
    sequence = "sequence"  # sequence of amino acids
    sample_pdb_path = "sample_pdb_path"  # filepath to generated pdb file
    folded_pdb_path = "folded_path"  # filepath to folded pdb file

    # assess folded structures
    plddt_mean = "plddt_mean"  # mean pLDDT score

    # structure comparison
    # RMSD generated sample -> folded structure
    bb_rmsd_folded = "bb_rmsd"
    # designability of structure, i.e. RMSD < 2.0
    is_designable = "is_designable"
    # RMSD generated sample -> ground truth (if true_bb_positions provided)
    bb_rmsd_gt = "bb_rmsd_gt"
    # RMSD folded structure -> ground truth structure (if true_bb_positions provided)
    bb_rmsd_folded_gt = "bb_rmsd_folded_gt"
    # (inpainting) RMSD of fixed motifs, generated sample -> folded structure
    motif_bb_rmsd_folded = "motif_bb_rmsd_folded"
    # (inpainting) RMSD of fixed motifs, generated sample -> GT
    motif_bb_rmsd_gt = "motif_bb_rmsd_gt"  # RMSD of fixed motifs to ground truth
    # (inpainting) RMSD of fixed motifs, folded structure -> GT
    motif_bb_rmsd_folded_gt = (
        "motif_bb_rmsd_folded_gt"  # RMSD of fixed motifs after folding to ground truth
    )

    # sequence recovery
    # gt => ground truth true_aa provided
    # inverse folded sequence -> predicted sequence recovery (only for designability df)
    inverse_folding_sequence_recovery_pred = "inverse_folding_sequence_recovery_pred"
    # either predicted or inverse folded sequence recovery with ground truth sequence
    inverse_folding_sequence_recovery_gt = "inverse_folding_sequence_recovery_gt"
    # (inpainting) sequence recovery of fixed motifs
    motif_sequence_recovery = "motif_sequence_recovery"
    # (inpainting) sequence recovery of fixed motifs after inverse folding
    motif_inverse_folding_sequence_recovery = "motif_inverse_folding_sequence_recovery"

    # summary metrics for inverse folding + folding
    # counts
    num_inverse_folded = "num_inverse_folded"
    num_designable = "num_designable"
    # codesign
    inverse_folding_sequence_recovery_mean = "inverse_folding_sequence_recovery_mean"
    inverse_folding_sequence_recovery_max = "inverse_folding_sequence_recovery_max"
    inverse_folding_bb_rmsd_single_seq = "inverse_folding_bb_rmsd_single_seq"
    inverse_folding_bb_rmsd_min = "inverse_folding_bb_rmsd_min"
    inverse_folding_bb_rmsd_mean = "inverse_folding_bb_rmsd_mean"
    # inpainting
    inverse_folding_motif_sequence_recovery_mean = (
        "inverse_folding_motif_sequence_recovery_mean"
    )
    inverse_folding_motif_bb_rmsd_mean = "inverse_folding_motif_bb_rmsd_mean"


class OutputFileName(StrEnum):
    # input PDB
    true_structure_pdb = "true.pdb"
    true_sequence_fa = "true.fasta"

    # trajectory
    sample_pdb = "sample.pdb"
    sample_pdb_backbone = "sample_bb.pdb"
    sample_traj_pdb = "sample_traj.pdb"
    model_pred_traj_pdb = "model_pred_traj.pdb"
    aa_traj_fa = "aa_traj.fasta"
    logits_traj_gif = "logits_traj.gif"
    logits_traj_mp4 = "logits_traj.mp4"
    traj_panel_gif = "traj_panel.gif"
    traj_panel_mp4 = "traj_panel.mp4"

    # folding validation
    sample_sequence_fa = "sample.fasta"
    # inverse_folded_fa determined by MPNN, matches sample name
    # inverse_folded_fa = "inverse_fold.fasta"
    # folded_pdb_path determined by alphafold + model we use
    # folded_pdb_path = "folded.pdb"

    # sample summaries / top samples
    top_sample_json = "top_samples.json"
    codesign_df = "codesign.csv"
    designability_df = "designability.csv"

    # all samples + metrics
    all_top_samples_df = "all_top_samples.csv"
    designable_metrics_df = "designable_metrics.csv"
    forward_fold_metrics_df = "forward_fold_metrics.csv"
    inverse_fold_metrics_df = "inverse_fold_metrics.csv"
