export CUDA_VISIBLE_DEVICES=0,1
# CUDA_LAUNCH_BLOCKING=1
python train_gleason.py \
--n_class 6 \
--file_index_train "/media/newhd/ysong/dataset/Gleason2019/file_index/train_list_with_mask.csv" \
--file_index_val "/media/newhd/ysong/dataset/Gleason2019/file_index/val_list_with_mask.csv" \
--model_path "/media/newhd/ysong/project/GLNet/saved_models/" \
--log_path "/media/newhd/ysong/project/GLNet/log" \
--task_name "fpn_gleason_global" \
--mode 1 \
--batch_size 8 \
--sub_batch_size 16 \
--size_g 320 \
--size_p 320
