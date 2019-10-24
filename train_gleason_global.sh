export CUDA_VISIBLE_DEVICES=0
python train_gleason.py \
--n_class 5 \
--file_index_train "/home/neo/dataset/dataset/Gleason2019/file_index/train_list_with_mask.csv" \
--file_index_val "/home/neo/dataset/dataset/Gleason2019/file_index/val_list_with_mask.csv" \
--model_path "/home/neo/project/GLNet/saved_models/" \
--log_path "/home/neo/project/GLNet/logs" \
--task_name "fpn_gleason_global" \
--mode 1 \
--batch_size 8 \
--sub_batch_size 8 \
--size_g 320 \
--size_p 320