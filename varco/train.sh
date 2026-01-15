# variable-motif variable-length scaffolding

# fine-tune varco model, e.g.:
# experiment.warm_start_ckpt=ckpt/varco/inpainting_20260115_065333/20260115_065333/last.ckpt
# fine-tune cogeneration model, e.g.:
# experiment.cogen_ckpt_path=ckpt/varco/inpainting_20260108_234526/20260108_234526/last.ckpt
# on A100 40GB:
# data.sampler.max_num_res_squared=260000

python varco/train.py \
 shared.seed=18532 \
 experiment.warm_start_ckpt=ckpt/varco/inpainting_20260115_065333/20260115_065333/last.ckpt  \
 dataset.filter.num_unique_seqs=[1,2,3] \
 dataset.filter.min_num_res=50 \
 dataset.filter.max_num_res=512 \
 dataset.filter.max_coil_percent=0.4 \
 dataset.filter.max_percent_residues_unknown=0.1 \
 dataset.filter.min_plddt=0.9 \
 dataset.enable_cogeneration_afdb=False \
 dataset.inpainting.strategy=variable_motifs \
 dataset.inpainting.min_motif_len=5 \
 dataset.inpainting.max_motif_len=60 \
 data.sampler.max_num_res_squared=260000 \
 experiment.trainer.check_val_every_n_epoch=5
 