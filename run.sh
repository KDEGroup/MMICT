export TOKENIZERS_PARALLELISM=true
python3 -um torch.distributed.launch --nnodes=1 --nproc_per_node=4 \
--node_rank=$INDEX --master_port=1111 --master_addr=$CHIEF_IP \
train.py \
--cfg-path lavis/projects/blip2/train/mmict.yaml
