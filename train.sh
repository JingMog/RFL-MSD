export CUDA_VISIBLE_DEVICES=0,1,2,3
TRAIN_GPU_PER_NODE=2

OMP_NUM_THREADS=2 torchrun --nproc_per_node $TRAIN_GPU_PER_NODE \
        --master_port=12503 ce_trainer.py




