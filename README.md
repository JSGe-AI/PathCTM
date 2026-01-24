

train:

python CONCH_4-scale_train.py \
    --train_h5_dir /path/to/train.txt \
    --val_h5_dir   /path/to/val.txt \
    --log_dir logs/exp1 \
    --num_class   \
    --device 0



Inference:

python CONCH_4-scale_inference.py \
    --batch_size_test 1 \
    --device 0 \
    --num_class 7 \
    --threshold 0.9 \
    --fold n \
    --test_h5_dir "/path/to/test_list.txt" \
    --checkpoint_path "/path/to/checkpoint.pth" \
    --log_dir "./logs/"
