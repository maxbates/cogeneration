# Other options to consider:
# inference.use_vhh_dataset=True  # use VHH inpainting dataset

python varco/predict.py \
 shared.seed=27352 \
 inference.use_vhh_dataset=True \
 shared.id=0116_vhh_stoch07_t095_sharp125_haz25_nuc35_guid1decay_metrics \
 dataset.tree_plan.min_scaffold_nuclei=3 \
 dataset.tree_plan.max_scaffold_nuclei=5 \
 inference.ckpt_path=ckpt/varco/inpainting_20260116_062338/20260116_062338/last.ckpt \
 interpolant.trans_coupler.noise_scale=0.7 \
 interpolant.trans_coupler.noise_end_t=0.90 \
 interpolant.rotation_coupler.noise_scale=0.35 \
 interpolant.rotation_coupler.noise_end_t=0.90 \
 interpolant.motif_guidance.enabled=True \
 interpolant.motif_guidance.var_scale_type=ot \
 interpolant.motif_guidance.guidance_decay=True \
 interpolant.motif_guidance.guidance_start_t=0.02 \
 interpolant.motif_guidance.guidance_end_t=1.00 \
 interpolant.motif_guidance.obs_noise_rot_rad=0.3 \
 interpolant.sampling.indel_sharpness=1.25 \
 interpolant.sampling.split_hazard.power=2.5 \
 dataset.num_eval_lengths=48 \
 dataset.samples_per_eval_length=4 \
 dataset.inpainting.strategy=variable_motifs \
 dataset.inpainting.scaffold_length_scale=1.0 \
 dataset.filter.max_non_residue_entities=0 \
 dataset.filter.min_plddt=0.9 \
 dataset.filter.max_coil_percent=0.3 \
 dataset.filter.max_percent_residues_unknown=0.05 \
 dataset.filter.num_unique_seqs=[1] \
 dataset.filter.num_chains=[1] \
 dataset.filter.min_num_res=96 \
 dataset.filter.max_num_res=480
