export CUDA_VISIBLE_DEVICES=0
python train_deep_globe.py \
--n_class 5 \
--data_path "/ssd1/chenwy/deep_globe/data/" \
--model_path "/home/chenwy/deep_globe/saved_models/" \
--log_path "/home/chenwy/deep_globe/runs/" \
--task_name "fpn_deepglobe_global" \
--mode 1 \
--batch_size 4 \
--sub_batch_size 8 \
--size_g 320 \
--size_p 320 \