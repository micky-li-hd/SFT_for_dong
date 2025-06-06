set -ex
if [ ! -z ${AZ_BATCHAI_GPU_COUNT+x} ]; then
    GPU_PER_NODE_COUNT=${AZ_BATCHAI_GPU_COUNT}
fi
export AZUREML_NODE_COUNT=${AZUREML_NODE_COUNT:=1}
export GPU_PER_NODE_COUNT=${GPU_PER_NODE_COUNT:=8}
export NODE_RANK=${NODE_RANK:=0}
export MASTER_ADDR=${MASTER_ADDR:=localhost}
export MASTER_PORT=${MASTER_PORT:=1828}
# exam environment variables
echo "world_size:"$WORLD_SIZE
echo "local_rank:"$LOCAL_RANK
echo "unstable_global_rank:"$RANK
echo "node_rank:"$NODE_RANK 
echo "master_addr:"$MASTER_ADDR
echo "master_port:"$MASTER_PORT  
echo "mpinode_rank:"$OMPI_COMM_WORLD_RANK 
echo "mpimaster_portaddr:"$AZ_BATCH_MASTER_NODE 
arrIN=(${AZ_BATCH_MASTER_NODE//:/ })
echo ${arrIN[0]} 
echo ${arrIN[1]}
echo "world_size:"$WORLD_SIZE
echo "local_rank:"$LOCAL_RANK
echo "unstable_global_rank:"$RANK
echo "node_rank:"$NODE_RANK 
echo "master_addr:"$MASTER_ADDR
echo "master_port:"$MASTER_PORT
arrIN=(${AZ_BATCH_MASTER_NODE//:/ })
echo ${arrIN[0]} 
echo ${arrIN[1]}
 
NUM_PROCESS=$((GPU_PER_NODE_COUNT * AZUREML_NODE_COUNT))
echo "gpu count:"$GPU_PER_NODE_COUNT
echo "node count:"$AZUREML_NODE_COUNT
echo "num process:"$NUM_PROCESS
 
export NCCL_DEBUG='INFO'
 
NODE_RANK_PADDED=$(printf "%02d" ${NODE_RANK})
accelerate launch --num_machines=${AZUREML_NODE_COUNT} --num_processes=${NUM_PROCESS} --machine_rank=${NODE_RANK} --main_process_port=${MASTER_PORT} --main_process_ip=${MASTER_ADDR} --config_file accelerate_config/2_8_GPU_config.yaml \
    training/train.py config=config/sft.yaml
