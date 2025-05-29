Start through "run.sh"

1. 设置batchsize：
打开config/sft.yaml-->training:batch_size: 4

2. 设置input_ids max_length(截断长度)
打开training/data_new.py-->class DataCollatorForSupervisedDataset中max_len = 1024

3. 设置多卡
打开accelerate_config/4_GPU_config-->num_machines: 1/num_processes: 4   

4. data
打开config/sft.yaml-->dataset: params: path: /mnt/v-haodongli/cot_output_test_train


启动：run.yaml