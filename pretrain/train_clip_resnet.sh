export CUDA_VISIBLE_DEVICES=0
python ./clip_pretraining.py \
--dataset_dir /cpfs01/shared/opendrivelab/yuchecheng/whole_dataset_freetacman_1 \
--save_dir /cpfs01/user/yuchecheng/FastUMI_Data/cobot_magic/aloha-devel/checkpoint/clip_resnet \
--num_episodes 1000 \
--batch_size 45 \
--n_clip_images 5 \
--min_distance 20 \
--save_freq 100 \
--plot_freq 50 \
--n_epochs 5000 \
--resnet_lr 1e-5 \
--projection_lr 1e-4 

