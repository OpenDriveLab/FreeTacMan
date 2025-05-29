num_epochs=6000
batch_size=32
num_episodes=200
dataset_dir=/cpfs01/shared/opendrivelab/yuchecheng/data

task=hdf5_writetacfive
save_dir=/nas/shared/opendrivelab/yuchecheng/checkpoint/${task}_act_resnet_64
python /cpfs01/user/yuchecheng/FastUMI_Data/cobot_magic/aloha-devel/act/train_tac.py \
--dataset_dir $dataset_dir \
--ckpt_dir $save_dir \
--num_episodes $num_episodes \
--batch_size $batch_size \
--num_epochs $num_epochs \
--task_name $task \
--policy_class ACT \
--chunk_size 64 \
--use_tactile_image true \
--pretrained_tactile_backbone true \
--tactile_backbone_path /cpfs01/user/yuchecheng/FastUMI_Data/cobot_magic/aloha-devel/checkpoint/clip_resnet/4/epoch_1999_tactile_encoder.pth

save_dir=/nas/shared/opendrivelab/yuchecheng/checkpoint/${task}_act_resnet_48
python /cpfs01/user/yuchecheng/FastUMI_Data/cobot_magic/aloha-devel/act/train_tac.py \
--dataset_dir $dataset_dir \
--ckpt_dir $save_dir \
--num_episodes $num_episodes \
--batch_size $batch_size \
--num_epochs $num_epochs \
--task_name $task \
--policy_class ACT \
--chunk_size 48 \
--use_tactile_image true \
--pretrained_tactile_backbone true \
--tactile_backbone_path /cpfs01/user/yuchecheng/FastUMI_Data/cobot_magic/aloha-devel/checkpoint/clip_resnet/4/epoch_1999_tactile_encoder.pth

save_dir=/nas/shared/opendrivelab/yuchecheng/checkpoint/${task}_act_resnet_32
python /cpfs01/user/yuchecheng/FastUMI_Data/cobot_magic/aloha-devel/act/train_tac.py \
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
--tactile_backbone_path /cpfs01/user/yuchecheng/FastUMI_Data/cobot_magic/aloha-devel/checkpoint/clip_resnet/4/epoch_1999_tactile_encoder.pth


# wait

# save_dir=/cpfs01/user/yuchecheng/FastUMI_Data/cobot_magic/aloha-devel/checkpoint/${task}_dp_32
# python /cpfs01/user/yuchecheng/FastUMI_Data/cobot_magic/aloha-devel/act/train.py \
# --dataset_dir $dataset_dir \
# --ckpt_dir $save_dir \
# --num_episodes $num_episodes \
# --batch_size $batch_size \
# --num_epochs $num_epochs \
# --task_name $task \
# --policy_class Diffusion \
# --chunk_size 32 &

# # No tac
# save_dir=/cpfs01/user/yuchecheng/FastUMI_Data/cobot_magic/aloha-devel/checkpoint/${task}_no_tac_dp_64
# python /cpfs01/user/yuchecheng/FastUMI_Data/cobot_magic/aloha-devel/act/train_no_tac.py \
# --dataset_dir $dataset_dir \
# --ckpt_dir $save_dir \
# --num_episodes $num_episodes \
# --batch_size $batch_size \
# --num_epochs $num_epochs \
# --task_name $task \
# --policy_class Diffusion \
# --chunk_size 64 &

# wait

# save_dir=/cpfs01/user/yuchecheng/FastUMI_Data/cobot_magic/aloha-devel/checkpoint/${task}_no_tac_dp_48
# python /cpfs01/user/yuchecheng/FastUMI_Data/cobot_magic/aloha-devel/act/train_no_tac.py \
# --dataset_dir $dataset_dir \
# --ckpt_dir $save_dir \
# --num_episodes $num_episodes \
# --batch_size $batch_size \
# --num_epochs $num_epochs \
# --task_name $task \
# --policy_class Diffusion \
# --chunk_size 48 &

# save_dir=/cpfs01/user/yuchecheng/FastUMI_Data/cobot_magic/aloha-devel/checkpoint/${task}_no_tac_dp_32
# python /cpfs01/user/yuchecheng/FastUMI_Data/cobot_magic/aloha-devel/act/train_no_tac.py \
# --dataset_dir $dataset_dir \
# --ckpt_dir $save_dir \
# --num_episodes $num_episodes \
# --batch_size $batch_size \
# --num_epochs $num_epochs \
# --task_name $task \
# --policy_class Diffusion \
# --chunk_size 32 &

# wait