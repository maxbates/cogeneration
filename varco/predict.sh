# checkpoints: 
# variable motifs split mass
# ckpt/varco/inpainting_20260110_222619/20260110_222619/last.ckpt

# Other options to consider:
# inference.use_vhh_dataset=True  # use VHH inpainting dataset

python varco/predict.py \
 shared.seed=2236 \
 shared.id=vhh_stoch04_sharp19_haz25_fixmult \
 inference.use_vhh_dataset=True \
 dataset.tree_plan.min_scaffold_nuclei=3 \
 dataset.tree_plan.max_scaffold_nuclei=5 \
 inference.ckpt_path=ckpt/varco/inpainting_20260110_222619/20260110_222619/last.ckpt \
 inference.folding_validation.enabled=False \
 interpolant.trans_coupler.noise_scale=0.4 \
 interpolant.trans_coupler.noise_end_t=0.9 \
 interpolant.rotation_coupler.noise_scale=0.25 \
 interpolant.rotation_coupler.noise_end_t=0.9 \
 interpolant.motif_guidance.enabled=True \
 interpolant.motif_guidance.var_scale_type=ot \
 interpolant.motif_guidance.guidance_decay=True \
 interpolant.motif_guidance.guidance_start_t=0.02 \
 interpolant.motif_guidance.guidance_end_t=0.98 \
 interpolant.motif_guidance.obs_noise_rot_rad=0.5 \
 interpolant.sampling.indel_sharpness=1.9 \
 interpolant.sampling.split_hazard.power=2.5 \
 dataset.inpainting.strategy=variable_motifs \
 dataset.inpainting.scaffold_length_scale=1.0 \
 dataset.filter.max_non_residue_entities=0 \
 dataset.filter.min_plddt=0.9 \
 dataset.filter.max_coil_percent=0.3 \
 dataset.filter.max_percent_residues_unknown=0.05 \
 dataset.filter.num_chains=[2] \
 dataset.filter.min_num_res=120 \
 dataset.filter.max_num_res=384
