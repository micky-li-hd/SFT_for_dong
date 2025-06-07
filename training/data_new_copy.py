import os
from dataclasses import dataclass
from typing import Dict, Sequence
import torch
from torch.utils.data import Dataset
from datasets import load_dataset, concatenate_datasets
from transformers import AutoProcessor
import glob
import transformers
from torch.nn.utils.rnn import pad_sequence
from dataclasses import dataclass
from typing import Sequence, Dict, Any
import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import PreTrainedTokenizer

@dataclass
class DataCollatorForSupervisedDataset:
    """Collate examples for supervised fine-tuning."""
    tokenizer: PreTrainedTokenizer
    processor: Any  # 替换为你的具体 processor 类型（如 VLMProcessor）
    max_length: int = 1024
    IGNORE_INDEX: int = -100

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids_list = []
        labels_list = []
        text_ids_mask_list = []
        image_ids_mask_list = []
        label_text_ids_mask_list = []
        label_image_ids_mask_list = []

        for instance in instances:
            # 提取 caption 和 img_index
            try:
                json_data = instance['json']
                caption = json_data['caption']
                cot = json_data['cot']  # 注意这里新增了 cot 字段
                img_index = json_data['img_index']  # list of int 或者 tensor
            except KeyError as e:
                raise ValueError(f"Missing key in instance: {e}")

            # 构造 conversation
            conversation = [
                {"role": "<|User|>", "content": caption},
                {"role": "<|Assistant|>", "content": f"{cot}<begin_of_image><end_of_image>"},
            ]
            system_prompt = "You are an assistant that creates images from descriptions. First, describe the image in detail, then generate it."

            # 使用 self.processor 来生成 prompt
            prompt = self.processor.apply_sft_template_for_multi_turn_prompts(
                conversations=conversation,
                sft_format=self.processor.sft_format,
                system_prompt=system_prompt,
            )

            # Tokenize prompt
            text_ids = self.tokenizer.encode(prompt)

            # 插入图像 token ID
            all_ids = text_ids[:-2] + img_index + text_ids[-2:]
            all_ids = torch.LongTensor(all_ids)

            # 构建图像 token 的 mask
            all_image_ids_mask = torch.zeros(len(all_ids), dtype=torch.bool)
            all_image_ids_mask[-len(img_index)-2:-2] = True

            # 找到 Assistant 回答开始的位置
            try:
                assistant_start_token_id = self.tokenizer.encode("<|Assistant|>")[0]
                assistant_start_index = (all_ids == assistant_start_token_id).nonzero(as_tuple=True)[0][0].item()
            except Exception:
                assistant_start_index = 0

            # 构造各类 mask
            assistant_mask = torch.zeros(len(all_ids), dtype=torch.bool)
            assistant_mask[assistant_start_index:] = True

            # 构造 input 和 label
            input_ids = all_ids[:-1]
            label_ids = all_ids[1:]

            text_mask = (all_image_ids_mask[:-1] == False)
            image_mask = all_image_ids_mask[:-1]

            label_text_mask = assistant_mask[1:] & (all_image_ids_mask[1:] == False)
            label_image_mask = assistant_mask[1:] & all_image_ids_mask[1:]

            # 添加进列表
            input_ids_list.append(input_ids)
            labels_list.append(label_ids)
            text_ids_mask_list.append(text_mask)
            image_ids_mask_list.append(image_mask)
            label_text_ids_mask_list.append(label_text_mask)
            label_image_ids_mask_list.append(label_image_mask)

        # Padding 处理
        input_ids = pad_sequence(input_ids_list, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        labels = pad_sequence(labels_list, batch_first=True, padding_value=self.IGNORE_INDEX)
        text_ids_mask = pad_sequence(text_ids_mask_list, batch_first=True, padding_value=False)
        image_ids_mask = pad_sequence(image_ids_mask_list, batch_first=True, padding_value=False)
        label_text_ids_mask = pad_sequence(label_text_ids_mask_list, batch_first=True, padding_value=False)
        label_image_ids_mask = pad_sequence(label_image_ids_mask_list, batch_first=True, padding_value=False)

        # 截断处理
        if input_ids.size(1) > self.max_length:
            input_ids = input_ids[:, :self.max_length]
            labels = labels[:, :self.max_length]
            text_ids_mask = text_ids_mask[:, :self.max_length]
            image_ids_mask = image_ids_mask[:, :self.max_length]
            label_text_ids_mask = label_text_ids_mask[:, :self.max_length]
            label_image_ids_mask = label_image_ids_mask[:, :self.max_length]

        return dict(
            input_ids=input_ids,
            label_ids=labels,
            attention_mask=(input_ids != self.tokenizer.pad_token_id),
            text_id_mask=text_ids_mask,
            image_id_mask=image_ids_mask,
            label_text_id_mask=label_text_ids_mask,
            label_image_id_mask=label_image_ids_mask,
        )

# ======== 主程序入口 ========
if __name__ == "__main__":
    from janus.models.processing_vlm import VLChatProcessor
    
    # 初始化处理器和 tokenizer
    processor: VLChatProcessor = VLChatProcessor.from_pretrained("deepseek-ai/Janus-Pro-7B")
    tokenizer = processor.tokenizer
    padding_id = tokenizer.pad_token_id
    data_files = glob.glob(os.path.join("/home/v-haodongli/mnt/v-haodongli-container/cot_output_test_train", "*.tar"))
    # train_dataset = load_dataset("webdataset", data_files=data_files, split="train", streaming=True ,num_proc=8)
    train_dataset = load_dataset("webdataset", data_files=data_files, split="train", num_proc=8)

    # 创建 collator
    data_collator = DataCollatorForSupervisedDataset(tokenizer)

    # 测试调用
    batch = data_collator([train_dataset[1], train_dataset[2], train_dataset[3]])
    
    # 打印输出示例
    print("Batch keys:", batch.keys())
    print("Input IDs shape:", batch["input_ids"].shape)
    print("Label IDs shape:", batch["label_ids"].shape)
    print("Attention mask shape:", batch["attention_mask"].shape)