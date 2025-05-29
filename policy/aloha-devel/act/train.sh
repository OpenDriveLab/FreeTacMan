
# num_epochs=10000
# batch_size=64
# num_episodes=35
# chunk_size=32
# task=fold_towel_new
# dataset_dir=/home/agilex/data
# policy_class=Diffusion
# save_dir=/home/agilex/cobot_magic/aloha-devel/checkpoint/${task}_${policy_class}_sample_cs_${chunk_size}

# python /home/agilex/cobot_magic/aloha-devel/act/train.py \
# --dataset_dir $dataset_dir \
# --ckpt_dir $save_dir \
# --num_episodes $num_episodes \
# --batch_size $batch_size \
# --num_epochs $num_epochs \
# --task_name $task \
# --policy_class $policy_class \
# --chunk_size $chunk_size

num_epochs=8000
batch_size=32
num_episodes=50
chunk_size=16
task=wly0221
dataset_dir=/home/agilex/data
policy_class=Diffusion
save_dir=/home/agilex/cobot_magic/aloha-devel/checkpoint/${task}_${policy_class}_sample_cs_${chunk_size}

python /home/agilex/cobot_magic/aloha-devel/act/train.py \
--dataset_dir $dataset_dir \
--ckpt_dir $save_dir \
--num_episodes $num_episodes \
--batch_size $batch_size \
--num_epochs $num_epochs \
--task_name $task \
--policy_class $policy_class \
--chunk_size $chunk_size

