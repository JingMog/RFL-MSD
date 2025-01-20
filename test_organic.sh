conda activate torch


TEST_LRC=ssml_valid.lrc
TEST_NAME=EDU-CHEMC-v3
EPOCH=45 # 45, 32
USED_GPU_ID=0_1_2
PROCESS_PER_GPU=3

echo TEST_LRC=${TEST_LRC}
echo TEST_NAME=${TEST_NAME}
echo EPOCH=${EPOCH}
echo PROCESS_PER_GPU=${PROCESS_PER_GPU}
echo USED_GPU_ID=${USED_GPU_ID}

python3 test_lrc_top1top3_log.py \
        --process_per_gpu ${PROCESS_PER_GPU} \
        --used_gpu_id ${USED_GPU_ID} \
        --test_lrc=${TEST_LRC} \
        --name=${TEST_NAME} \
        --load_epoch=${EPOCH} \
        --is_show=False


