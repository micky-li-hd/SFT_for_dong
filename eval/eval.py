import os
import argparse
import torch
import numpy as np
import PIL.Image
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer
from janus.models import MultiModalityCausalLM, VLChatProcessor
import wandb

def parse_args():
    parser = argparse.ArgumentParser(description="Generate images from prompts using Janus model.")
    parser.add_argument("--prompt_file", type=str, required=True,
                        help="Path to the file containing prompts.")
    parser.add_argument("--checkpoint_path", type=str, required=True,
                        help="Path to the model checkpoint.")
    parser.add_argument("--save_root", type=str, default="~/mnt/v-haodongli-container_doch/haodongli/eval/geneval",
                        help="Directory to save generated images.")
    return parser.parse_args()


@torch.inference_mode()
def generate_text_then_image_with_cfg(
    mmgpt: MultiModalityCausalLM,
    vl_chat_processor: VLChatProcessor,
    prompt: str,
    max_new_tokens: int = 200,
    image_token_num_per_image: int = 576,
    temperature: float = 0.5,
    cfg_weight: float = 5.0,
    img_size: int = 384,
    patch_size: int = 16,
    parallel_size: int = 16,
):
    tokenizer = vl_chat_processor.tokenizer
    device = mmgpt.device
    vocab_size = tokenizer.vocab_size
    begin_of_image_id = tokenizer.convert_tokens_to_ids("<begin_of_image>")

    # Step 1: 文本生成阶段（不使用 CFG）
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    inputs_embeds = mmgpt.language_model.get_input_embeddings()(input_ids)

    generated_text_tokens = []
    past_key_values = None
    is_generating_image = False
    image_token_count = 0

    print("🔍 Starting text generation...")
    for step in range(max_new_tokens):
        with torch.no_grad():
            outputs = mmgpt.language_model.model(
                inputs_embeds=inputs_embeds,
                use_cache=True,
                past_key_values=past_key_values
            )
            hidden_states = outputs.last_hidden_state
            past_key_values = outputs.past_key_values

        logits = mmgpt.language_model.lm_head(hidden_states[:, -1, :])
        probs = torch.softmax(logits / temperature, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1).squeeze(0)

        next_token_id = next_token.item()
        generated_text_tokens.append(next_token_id)

        if next_token_id == begin_of_image_id:
            print(f"🖼️ Detected <begin_of_image>, switching to image generation.")
            is_generating_image = True
            break

        inputs_embeds = mmgpt.language_model.get_input_embeddings()(next_token.unsqueeze(0))

    assert is_generating_image, "Model did not generate <begin_of_image>."
    generated_text = tokenizer.decode(generated_text_tokens, skip_special_tokens=True)
    print(f"📝 Generated text: {generated_text}")

    # Step 2: 构造 condition/uncondition 输入用于图像生成
    cond_tokens = torch.cat([
        input_ids[0],
        torch.tensor(generated_text_tokens, dtype=torch.long, device=device),
        torch.tensor([begin_of_image_id], dtype=torch.long, device=device)
    ])

    uncond_tokens = torch.cat([
        input_ids[0][:1],
        torch.tensor([vl_chat_processor.pad_id] * (len(cond_tokens) - 2), dtype=torch.long, device=device),
        torch.tensor([begin_of_image_id], dtype=torch.long, device=device)
    ])

    cond_tokens = cond_tokens.unsqueeze(0).repeat(parallel_size, 1)
    uncond_tokens = uncond_tokens.unsqueeze(0).repeat(parallel_size, 1)
    tokens = torch.cat([cond_tokens, uncond_tokens], dim=0)
    inputs_embeds = mmgpt.language_model.get_input_embeddings()(tokens)
    past_key_values = None

    generated_image_tokens = torch.zeros((parallel_size, image_token_num_per_image), dtype=torch.int, device=device)

    print("🖼️ Starting image token generation with CFG...")

    for i in range(image_token_num_per_image):
        with torch.no_grad():
            outputs = mmgpt.language_model.model(
                inputs_embeds=inputs_embeds,
                use_cache=True,
                past_key_values=past_key_values
            )
            hidden_states = outputs.last_hidden_state
            past_key_values = outputs.past_key_values

        logits = mmgpt.gen_head(hidden_states[:, -1, :])
        logit_cond = logits[0::2, :]
        logit_uncond = logits[1::2, :]

        logits = logit_uncond + cfg_weight * (logit_cond - logit_uncond)
        probs = torch.softmax(logits / temperature, dim=-1)

        next_token = torch.multinomial(probs, num_samples=1)
        generated_image_tokens[:, i] = next_token.squeeze(dim=-1)

        next_token = torch.cat([next_token.unsqueeze(dim=1), next_token.unsqueeze(dim=1)], dim=1).view(-1)
        img_embeds = mmgpt.prepare_gen_img_embeds(next_token)
        inputs_embeds = img_embeds.unsqueeze(dim=1)

    # Step 3: 解码图像 token 成图像
    dec = mmgpt.gen_vision_model.decode_code(
        generated_image_tokens.to(dtype=torch.int),
        shape=[parallel_size, 8, img_size // patch_size, img_size // patch_size]
    )

    dec = dec.to(torch.float32).cpu().numpy().transpose(0, 2, 3, 1)
    dec = np.clip((dec + 1) / 2 * 255, 0, 255).astype(np.uint8)

    visual_img = np.zeros((parallel_size, img_size, img_size, 3), dtype=np.uint8)
    visual_img[:, :, :] = dec

    return generated_text, visual_img

@torch.inference_mode()
def generate(
    mmgpt: MultiModalityCausalLM,
    vl_chat_processor: VLChatProcessor,
    prompt: str,
    temperature: float = 1,
    parallel_size: int = 16,
    cfg_weight: float = 5,
    image_token_num_per_image: int = 576,
    img_size: int = 384,
    patch_size: int = 16,
):
    input_ids = vl_chat_processor.tokenizer.encode(prompt)
    input_ids = torch.LongTensor(input_ids)

    tokens = torch.zeros((parallel_size*2, len(input_ids)), dtype=torch.int).cuda()
    for i in range(parallel_size*2):
        tokens[i, :] = input_ids
        if i % 2 != 0:
            tokens[i, 1:-1] = vl_chat_processor.pad_id

    inputs_embeds = mmgpt.language_model.get_input_embeddings()(tokens)

    generated_tokens = torch.zeros((parallel_size, image_token_num_per_image), dtype=torch.int).cuda()

    for i in range(image_token_num_per_image):
        outputs = mmgpt.language_model.model(inputs_embeds=inputs_embeds, use_cache=True, past_key_values=outputs.past_key_values if i != 0 else None)
        hidden_states = outputs.last_hidden_state
        
        logits = mmgpt.gen_head(hidden_states[:, -1, :])
        logit_cond = logits[0::2, :]
        logit_uncond = logits[1::2, :]
        
        logits = logit_uncond + cfg_weight * (logit_cond-logit_uncond)
        probs = torch.softmax(logits / temperature, dim=-1)

        next_token = torch.multinomial(probs, num_samples=1)
        generated_tokens[:, i] = next_token.squeeze(dim=-1)

        next_token = torch.cat([next_token.unsqueeze(dim=1), next_token.unsqueeze(dim=1)], dim=1).view(-1)
        img_embeds = mmgpt.prepare_gen_img_embeds(next_token)
        inputs_embeds = img_embeds.unsqueeze(dim=1)


    dec = mmgpt.gen_vision_model.decode_code(generated_tokens.to(dtype=torch.int), shape=[parallel_size, 8, img_size//patch_size, img_size//patch_size])
    dec = dec.to(torch.float32).cpu().numpy().transpose(0, 2, 3, 1)

    dec = np.clip((dec + 1) / 2 * 255, 0, 255)

    visual_img = np.zeros((parallel_size, img_size, img_size, 3), dtype=np.uint8)
    visual_img[:, :, :] = dec

    return  visual_img

def main():
    args = parse_args()

    # 初始化 wandb
    wandb.init(
        project="image-generation-evaluation",  # 项目名可自定义
        config={
            "prompt_file": args.prompt_file,
            "checkpoint_path": args.checkpoint_path,
            "parallel_size": 16,
            "max_new_tokens": 200,
            "image_token_num_per_image": 576,
            "temperature": 0.5,
            "cfg_weight": 5.0,
        }
    )

    # 加载模型和 processor
    model_path = "deepseek-ai/Janus-Pro-7B"
    vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(model_path)
    tokenizer = vl_chat_processor.tokenizer

    vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True
    ).to("cuda").eval()

    vl_gpt_tune: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
        args.checkpoint_path,
        trust_remote_code=True
    ).to("cuda").eval()

    # 读取所有 prompt
    with open(args.prompt_file) as fp:
        prompts = [line.strip() for line in fp if line.strip()]

    print(f"总共 {len(prompts)} 条 prompt")

    # 批量处理每条 prompt
    for idx, prompt_text in enumerate(prompts):
        print(f"\n🚀 正在处理第 {idx + 1}/{len(prompts)} 条 prompt:")
        print(f"Prompt: {prompt_text}")

        # 构建对话格式
        conversation = [
            {
                "role": "<|User|>",
                "content": prompt_text,
            },
            {"role": "<|Assistant|>", "content": ""},
        ]

        # 应用 SFT 模板格式
        sft_format = vl_chat_processor.apply_sft_template_for_multi_turn_prompts(
            conversations=conversation,
            sft_format=vl_chat_processor.sft_format,
            system_prompt="You are a helpful assistant that generates images based on text prompts.",
        )
        prompt = sft_format  # 不加 <begin_of_image>

        # 生成图像
        generated_text, visual_img_tune = generate_text_then_image_with_cfg(
            mmgpt=vl_gpt_tune,
            vl_chat_processor=vl_chat_processor,
            prompt=prompt,
            max_new_tokens=200,
            image_token_num_per_image=576,
            temperature=1,
            cfg_weight=5.0,
            parallel_size=4,
        )

        visual_img = generate(
            mmgpt=vl_gpt,
            vl_chat_processor=vl_chat_processor,
            prompt=prompt + vl_chat_processor.image_start_tag,
            image_token_num_per_image=576,
            temperature=1,
            cfg_weight=5.0,
            parallel_size=4,
        )

        # 保存图像路径或转换为 wandb.Image
        image_logs_tune = []
        for i in range(visual_img_tune.shape[0]):
            pil_img = PIL.Image.fromarray(visual_img_tune[i])
            image_logs_tune.append(wandb.Image(pil_img, caption=f"Image {i}"))

        image_logs = []
        for i in range(visual_img.shape[0]):
            pil_img = PIL.Image.fromarray(visual_img[i])
            image_logs.append(wandb.Image(pil_img, caption=f"Image {i}"))
        
        compare_images = []

        for img_tune_np, img_orig_np in zip(visual_img_tune, visual_img):
            # 检查是否是 [0, 255] 范围的 uint8 类型
            if img_tune_np.dtype != np.uint8:
                img_tune_np = (img_tune_np * 255).astype(np.uint8)
            if img_orig_np.dtype != np.uint8:
                img_orig_np = (img_orig_np * 255).astype(np.uint8)

            # 转换为 PIL 图像
            img_tune = Image.fromarray(img_tune_np)
            img_orig = Image.fromarray(img_orig_np)

            # 拼接图像
            w = img_tune.width + img_orig.width
            h = max(img_tune.height, img_orig.height)
            compared_img = Image.new('RGB', (w, h))
            compared_img.paste(img_tune, (0, 0))
            compared_img.paste(img_orig, (img_tune.width, 0))

            # 添加标题
            caption = f"Prompt: {prompt_text}\nFine-tuned Output: {generated_text}"
            compare_images.append(wandb.Image(compared_img, caption=caption))

        # Log 到 WandB
        wandb.log({"comparison_fine_tuned_vs_original": compare_images})
if __name__ == "__main__":
    main()