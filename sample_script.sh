
source path_to_env/bin/activate

python3 run.py --experiment_name exp_name \
--log_dir path_to_dir \
--chunk_file_location path_to_json_file \
--patch_pattern pattern_of_patches \
--subtypes p53abn=0 p53wt=1 \
--batch_size 50 \
--eval_batch_size 100 \
--num_classes 2 \
--num_patch_workers 4 \
--backbone resnet34 \
train-attention \
--lr 0.00005 \
--wd 0.00001 \
--epochs 30 \
--optimizer Adam \
--patience 10 \
--lr_patience 5 \
--use_schedular \
VarMIL \
