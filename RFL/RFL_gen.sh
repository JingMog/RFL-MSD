conda activate torch

WORKERS=40
# EDU-CHEMC
# Test
python ./RFL_main.py \
    -input ./test_ssml_sd.txt \
    -output ./result/test_cs_string.txt \
    -error_output  ./result/test_error.txt\
    -num_workers ${WORKERS}

# Train
# python ./complex2simple_main.py \
#     -input ./train_ssml_sd.txt \
#     -output ./result/train_cs_string.txt \
#     -error_output  ./result/train_error.txt\
#     -num_workers ${WORKERS}





