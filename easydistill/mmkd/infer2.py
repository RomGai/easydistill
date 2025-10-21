import json
import os
import math
import torch
import logging
import argparse
import jsonlines
from tqdm import tqdm
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def read_json_field(filename, limit=50):
    try:
        with open(filename, 'r', encoding='utf-8') as file:
            data = json.load(file)
            if limit is not None:
                data = data[:limit]
                logging.info(f"⚙️ Only processing first {limit} samples")
            return data
    except Exception as e:
        logging.error(f"Error reading JSON: {e}")
        return []



def write_data_to_json_file(data, file_path):
    try:
        with open(file_path, 'w', encoding='utf-8') as file:
            json.dump(data, file, ensure_ascii=False, indent=4)
        logging.info(f"✅ Data written to {file_path}")
    except Exception as e:
        logging.error(f"Error writing JSON: {e}")


def load_qwen_model(config):
    model_path = config["models"]["teacher"]
    logging.info(f"Loading Qwen2.5-VL model from {model_path}")

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True,
    )
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    tokenizer = processor.tokenizer

    id_map = {str(i): tokenizer.convert_tokens_to_ids(str(i)) for i in range(1, 6)}

    @torch.no_grad()
    def compute_logits(inputs):
        logits = model(**inputs).logits[:, -1, :]
        batch_scores = torch.stack([logits[:, id_map[str(i)]] for i in range(1, 6)], dim=1)
        probs = torch.nn.functional.softmax(batch_scores, dim=1).cpu().tolist()
        return probs

    return processor, compute_logits, id_map


def generate_teacher_logits_batch(processor, compute_logits, id_map, data_list, config):
    outcomes=[]
    logits_records = []

    total = len(data_list)
    logging.info(f"Processing {total} samples...")

    for idx, item in enumerate(tqdm(data_list, desc="Generating teacher logits")):
        # === 提取 frame_path 与 query ===
        try:
            user_content = next(
                (block["content"] for block in item if block.get("role") == "user"), None
            )
            if user_content is None:
                raise ValueError("Missing user content")

            frame_path = next(
                (c["image"] for c in user_content if c.get("type") == "image"), None
            )
            query = next(
                (c["text"] for c in user_content if c.get("type") == "text"), None
            )

            if not frame_path or not query:
                raise ValueError(f"Incomplete sample: {item}")
        except Exception as e:
            logging.warning(f"⚠️ Failed to parse sample #{idx}: {e}")
            continue

        if not os.path.exists(frame_path):
            logging.warning(f"⚠️ Missing frame: {frame_path}")
            continue

        if not os.path.exists(frame_path):
            logging.warning(f"⚠️ Missing frame: {frame_path}")
            continue

        # === 构造 prompt ===
        prompt_text = f"""Given the image, which is a frame from a video, rate how relevant this frame is for answering the question: '{query}'.
Output only one number from 1 to 5, where:
1 = completely irrelevant — the frame provides no visual or contextual information related to the question or its answer.
2 = slightly relevant — the frame shows general background or context, but it is unlikely to contribute to answering.
3 = moderately relevant — the frame includes partial clues or indirect context that might help infer the answer, but the key evidence is missing.
4 = mostly relevant — the frame provides substantial visual or contextual information that can be used to answer the question, though not fully decisive.
5 = highly relevant — the frame clearly contains the decisive evidence or strong contextual cues that directly or indirectly support the correct answer."""

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": frame_path},
                    {"type": "text", "text": prompt_text},
                ],
            }
        ]

        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to("cuda")

        probs = compute_logits(inputs)[0]
        #weighted_score = sum((i + 1) * p for i, p in enumerate(probs))
        score = int(torch.tensor(probs).argmax().item() + 1)

        out={
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": str(score),
                }
            ],
        }

        item.append(out)

        outcomes.extend(item)

        # === 使用 token_id 作为 key ===
        probs_with_key = [{str(token_id): float(p) for token_id, p in zip(id_map.values(), probs)}]
        logits_records.append(probs_with_key)


        if (idx + 1) % 50 == 0:
            logging.info(f"[{idx+1}/{total}] processed...")


    # === 保存 logits ===
    with jsonlines.open(config["dataset"]["logits_path"], mode='w') as writer:
        for row in logits_records:
            writer.write(row)

    write_data_to_json_file(outcomes, config["dataset"]["labeled_path"])
    logging.info("✅ Teacher inference completed.")


def infer_with_teacher_model(config):
    logging.info('Generating distillation data (white-box)...')
    data_list = read_json_field(config["dataset"]["instruction_path"])

    processor, compute_logits, id_map = load_qwen_model(config)
    generate_teacher_logits_batch(processor, compute_logits, id_map, data_list, config)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to JSON config')
    args = parser.parse_args()

    config = json.load(open(args.config, 'r', encoding='utf-8'))
    infer_with_teacher_model(config)


if __name__ == "__main__":
    main()
