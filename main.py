import re
import warnings

import pandas as pd
import torch
import transformers
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)

# disable some warnings
transformers.logging.set_verbosity_error()
transformers.logging.disable_progress_bar()
warnings.filterwarnings("ignore")


def get_concat_h(im1, im2):
    # concat images horizontally
    dst = Image.new("RGB", (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst


def generate(model, tokenizer, text: str, image=None, max_new_tokens=8):
    if "bunny" in model.config._name_or_path.lower():
        if image is None:
            input_ids = tokenizer(text, return_tensors="pt").input_ids.to(model.device)
            # generate
            output = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                use_cache=True,
                do_sample=False,
            )[0]
        else:
            assert (
                "<image>" in text
            ), "Prompt should contain '<image>' to insert the image."
            text_chunks = [
                tokenizer(chunk).input_ids for chunk in text.split("<image>")
            ]
            input_ids = (
                torch.tensor(
                    text_chunks[0] + [-200] + text_chunks[1][1:], dtype=torch.long
                )
                .unsqueeze(0)
                .to(model.device)
            )
            image_tensor = model.process_images([image], model.config).to(
                dtype=model.dtype, device=model.device
            )
            # generate
            output = model.generate(
                input_ids,
                images=image_tensor,
                max_new_tokens=max_new_tokens,
                use_cache=True,
                do_sample=False,
            )[0]
        return tokenizer.decode(output, skip_special_tokens=True)
    elif "llava" in model.config._name_or_path.lower():
        inputs = tokenizer(text=text, images=image, return_tensors="pt").to(
            model.device
        )
        inputs["input_ids"][inputs["input_ids"] == 64003] = 64000
        output = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False
        )
        return tokenizer.decode(output[0], skip_special_tokens=True)


def similarity_generator(
    model, tokenizer, example_1, example_2, mode="text"
) -> list[str]:
    text_only_instructions = """Given two scenes, find up to five similarities between each scene. Output each similarity in a list."""
    image_only_instructions = """Given the two side-by-side images, find up to five similarities between each image.  Output each similarity in a list."""
    both_instructions = """Given two scenes and their corresponding images, find up to five similarities between each scene.  Output each similarity in a list."""

    scene_1 = example_1["description"]
    scene_2 = example_2["description"]
    image_1 = example_1["image"].resize([512, 512])
    image_2 = example_2["image"].resize([512, 512])
    if mode == "text":
        image = None
        prompt = f"{text_only_instructions}\n\nScene 1:\n\n{scene_1}\n\nScene 2:\n\n{scene_2}\n\nSimilarities:\n\n"
    elif mode == "image":
        image = get_concat_h(image_1, image_2)
        prompt = f"{image_only_instructions}<image>\n\nSimilarities:\n\n"
    elif mode == "both":
        image = get_concat_h(image_1, image_2)
        prompt = f"{both_instructions}<image>\n\nScene 1:\n\n{scene_1}\n\nScene 2:\n\n{scene_2}\n\nSimilarities:\n\n"
    else:
        raise ValueError("mode should be 'text' or 'image'")
    pred = generate(model, tokenizer, prompt, max_new_tokens=128, image=image)
    statements = pred.split("Similarities:\n\n")[-1].split("\n")
    statements = [re.sub(r"^\d+\.\s*", "", s) for s in statements]
    return statements


def similarity_statement_check(
    model,
    tokenizer,
    example_1: str,
    example_2: str,
    statement: str,
    mode="text",
) -> str:
    check_text_only_instructions = """Given two scenes, does the following statement apply to one of the images or to both images? Answer with 'one' or 'both'."""
    check_image_only_instructions = """Given the two side-by-side images, does the following statement apply to one of the images or to both images? Answer with 'one' or 'both'."""
    check_both_instructions = """Given two scenes and their corresponding images, does the following statement apply to one of the images or to both images? Answer with 'one' or 'both'."""

    scene_1 = example_1["description"]
    scene_2 = example_2["description"]
    image_1 = example_1["image"].resize([512, 512])
    image_2 = example_2["image"].resize([512, 512])
    if mode == "text":
        image = None
        prompt = f"{check_text_only_instructions}\n\nScene 1:\n\n{scene_1}\n\nScene 2:\n\n{scene_2}\n\nStatement:{statement}\n\nAnswer:"
    elif mode == "image":
        image = get_concat_h(image_1, image_2)
        prompt = f"{check_image_only_instructions}<image>\n\nStatement:{statement}\n\nAnswer:"
    elif mode == "both":
        image = get_concat_h(image_1, image_2)
        prompt = f"{check_both_instructions}<image>\n\nScene 1:\n\n{scene_1}\n\nScene 2:\n\n{scene_2}\n\nStatement:{statement}\n\nAnswer:"
    else:
        raise ValueError("mode should be 'text' or 'image'")
    pred = generate(model, tokenizer, prompt, max_new_tokens=16, image=image)
    pred = pred.split("Answer:")[-1].lower().strip(".").strip()
    if "both" in pred:
        return "both"
    elif "one" in pred:
        return "one"
    else:
        return "unknown"


if __name__ == "__main__":
    print("Loading dataset")
    dataset = load_dataset("google/docci")
    df = pd.read_csv("dataset.csv")
    print("Loading Model")
    model = AutoModelForCausalLM.from_pretrained(
        "BAAI/Bunny-v1_1-Llama-3-8B-V",
        torch_dtype=torch.float16,  # float32 for cpu
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        "BAAI/Bunny-v1_1-Llama-3-8B-V", trust_remote_code=True
    )

    statements = []
    for i, row in tqdm(df.iterrows(), total=len(df)):
        idx_1 = int(row.original_idx)
        idx_2 = int(row.distractor_idx)
        example_1 = dataset["train"][idx_1]
        example_2 = dataset["train"][idx_2]
        for modality in ["text", "image", "both"]:
            generated_similarity_statements = similarity_generator(
                model, tokenizer, example_1, example_2, mode=modality
            )
            for generated_similarity_statement in generated_similarity_statements:
                text_check = similarity_statement_check(
                    model,
                    tokenizer,
                    example_1,
                    example_2,
                    statement=generated_similarity_statement,
                    mode="text",
                )
                image_check = similarity_statement_check(
                    model,
                    tokenizer,
                    example_1,
                    example_2,
                    statement=generated_similarity_statement,
                    mode="image",
                )
                both_check = similarity_statement_check(
                    model,
                    tokenizer,
                    example_1,
                    example_2,
                    statement=generated_similarity_statement,
                    mode="both",
                )
                statements.append(
                    {
                        "dataset_idx": row.dataset_idx,
                        "statement": generated_similarity_statement,
                        "generated_with": modality,
                        "eval_text": text_check,
                        "eval_image": image_check,
                        "eval_both": both_check,
                    }
                )
    statements_df = pd.DataFrame(statements)
    # save to csv
    statements_df.to_csv("evaluated_statements.csv", index=False)
