export CUDA_VISIBLE_DEVICES=0
python train_gleason.py \
--n_class 6 \
--file_index_train "/media/newhd/ysong/dataset/Gleason2019/file_index/train_list_with_mask.csv" \
--file_index_val "/media/newhd/ysong/dataset/Gleason2019/file_index/val_list_with_mask.csv" \
--model_path "/media/newhd/ysong/project/GLNet/saved_models/" \
--log_path "/media/newhd/ysong/project/GLNet/logs/" \
--task_name "fpn_gleason_local2global" \
--mode 3 \
--batch_size 4 \
--sub_batch_size 4 \
--size_g 320 \
--size_p 320 \
--path_g "fpn_gleason_global2local.pth" \
--path_g2l "fpn_gleason_global2local.pth" \
