torchrun --nproc_per_node=4 \
 --nnodes=1 \
 --node_rank=0 \
 --master_add="192.168.0.246" \
 --master_port=2655 \
 LD_PRELOAD="./src/cuda_capture/libinttemp.so" runtime.py ./config/config.json 1 1 1