num_epochs=6000
batch_size=32
num_episodes=200
dataset_dir=dataset/processed/

task=test

save_dir=path_to_save
python policy/act/train.py \
--dataset_dir $dataset_dir \
--ckpt_dir $save_dir \
--num_episodes $num_episodes \
--batch_size $batch_size \
--num_epochs $num_epochs \
--task_name $task \
--policy_class ACT \
--chunk_size 32 \
--use_tactile_image true \
--pretrained_tactile_backbone true \
--tactile_backbone_path /path/to/your/pretrain/checkpoint
