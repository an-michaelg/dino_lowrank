# June 26 2024 - training DINO without LoRA as baseline
CUDA_VISIBLE_DEVICES=2,3 python -m torch.distributed.launch --nproc_per_node=2 main_dino.py \
	--arch vit_small \
	--patch_size 8 \
	--lora_rank 0 \
	--out_dim 2048 \
	--norm_last_layer false \
	--use_fp16 false \
	--lr 0.002 \
	--batch_size_per_gpu 8 \
	--epochs 10 \
	--warmup_epochs 0 \
	--saveckp_freq 3 \
	--local_crops_number 4 \
	--output_dir ../logs/training_base

# # June 22 2024 - first training run of DINO+LoRA
# CUDA_VISIBLE_DEVICES=1,3 python -m torch.distributed.launch --nproc_per_node=2 main_dino.py \
	# --arch vit_small \
	# --patch_size 8 \
	# --lora_rank 4 \
	# --out_dim 2048 \
	# --norm_last_layer false \
	# --use_fp16 false \
	# --lr 0.002 \
	# --batch_size_per_gpu 8 \
	# --epochs 10 \
	# --warmup_epochs 0 \
	# --saveckp_freq 3 \
	# --local_crops_number 4 \
	# --output_dir ../logs/training_1

# # --saveckp_freq
# # June 21 2024 - trial runs of DINO+LoRA
# # set --nproc_per_node to be equal to the number of visible GPUs
# # largest batch is somewhere between 16 and 8
# # probably set learning rate to be slightly lower in upcoming runs
# CUDA_VISIBLE_DEVICES=1,3 python -m torch.distributed.launch --nproc_per_node=2 main_dino.py \
	# --arch vit_small \
	# --patch_size 8 \
	# --lora_rank 4 \
	# --out_dim 2048 \
	# --norm_last_layer false \
	# --use_fp16 false \
	# --lr 0.005 \
	# --batch_size_per_gpu 8 \
	# --epochs 10 \
	# --warmup_epochs 0 \
	# --local_crops_number 4 \
	# --output_dir ../logs/test_run