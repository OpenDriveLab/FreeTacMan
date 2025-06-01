export CUDA_VISIBLE_DEVICES=0
python ./clip_pretraining_resnet.py \
--dataset_dir /path/to/your/full/dataset \
--save_dir /path/to/your/save/directory \
--num_episodes 1000 \
--batch_size 45 \
--n_clip_images 5 \
--min_distance 20 \
--save_freq 100 \
--plot_freq 50 \
--n_epochs 5000 \
--resnet_lr 1e-5 \
--projection_lr 1e-4 

