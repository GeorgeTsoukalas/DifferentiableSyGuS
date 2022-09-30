train_inv_specific(){
    CUDA_VISIBLE_DEVICES=$1 python3 train_nas.py \
    --algorithm nas \
    --exp_name inv \
    --trial 1 \
    --train_data data/inv_processed/train_inv_data.npy \
    --valid_data data/inv_processed/valid_inv_data.npy \
    --test_data data/inv_processed/test_inv_data.npy \
    --train_labels data/inv_processed/train_inv_labels.npy \
    --valid_labels data/inv_processed/valid_inv_labels.npy \
    --test_labels data/inv_processed/test_inv_labels.npy \
    --input_type "list" \
    --output_type "atom" \
    --input_size 2 \
    --output_size 1 \
    --num_labels 2 \
    --lossfxn "crossentropy" \
    --max_depth 4 \
    --learning_rate 0.01 \
    --search_learning_rate 0.01 \
    --train_valid_split 0.6 \
    --symbolic_epochs 6 \
    --neural_epochs 4 \
    --batch_size 200 \
    --random_seed 0 \
    --penalty 0.01 \
    --finetune_epoch 15 \
    --finetune_lr 0.01 \
    --node_share \
    --graph_unfold
}


train_inv_specific 0