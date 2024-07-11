import warnings
import random
import torch
import json
import transformers
import pandas as pd
from tqdm import tqdm
import spacy
transformers.logging.set_verbosity_error()
transformers.logging.disable_progress_bar()
warnings.filterwarnings('ignore')
# set device
device = 'cuda'  # or cpu
torch.set_default_device(device)

from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image

from datasets import load_dataset

def get_example_ids(min_thresh, max_thresh):
    data = pd.read_csv("/home/hackathon/hackathon/HallusionBench/dataset.csv")
    sub_data = data[(data.clip_similarity >min_thresh)&(data.clip_similarity<max_thresh)]
    original_idxs = sub_data.original_idx.values.tolist()
    distractor_idxs = sub_data.distractor_idx.values.tolist()
    example_idxs = set(original_idxs + distractor_idxs)
    return list(example_idxs)


if __name__ == "__main__":
    dataset = load_dataset("google/docci")
    nlp = spacy.load("en_core_web_sm")
    # model_name = '/nfs/public/hf/models/meta-llama/Meta-Llama-3-70B-Instruct/'
    model_name = 'microsoft/Phi-3-medium-4k-instruct'

    # create model
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16, # float32 for cpu
            device_map='auto',
            trust_remote_code=True,
            attn_implementation="flash_attention_2"
        )

    prompt = """Can you generate 3 general and special short question-answer pairs for the about objects, colors, 
count of colors or objects and actions in the sentence  '{caption}'.
Consider the answer to be one or two words only and avoid very-short questions. 
Return the result in json format containing list of dictionaries as given in below example:

```json
[
    {'question': 'What type of plants are depicted in the image?', 'answer': 'Cattail'},
    {'question': 'What is the nature of the water body in the image?', 'answer': 'Brackish'}
]
```
"""
    questions = {}
    example_idxs = get_example_ids(min_thresh=0.9, max_thresh=0.91)
    print(len(example_idxs))
    
    pbar = tqdm(enumerate(example_idxs), total=len(example_idxs))
    # random_idx = random.sample(range(0, len(dataset["train"])), 50)
    # pbar = tqdm(random_idx, total=len(random_idx))
    invalid_examples = set()
    for _, example_idx in pbar:
        example_A = dataset["train"][example_idx]
        doc_A = nlp(example_A["description"])
        example_id = example_A["example_id"]

        sentences_A = list(doc_A.sents)
        # batch_size = 8 if len(sentences_A)> 7 else len(sentences_A)
        batch_size = len(sentences_A)
        questions[example_id] = []
        for idx in range(0,len(sentences_A), batch_size):
            messages = []
            orig_sentences = []
            for batch in range(batch_size):
                messages.append(prompt.replace("{caption}",str(sentences_A[batch])))
                orig_sentences.append(str(sentences_A[batch]))
            
            # print(idx, end="\r", flush=True)
            # messages = [{"role": "user", "content": prompt.format(caption=sentence)}]
            # inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
            inputs = tokenizer(messages, return_tensors="pt", padding=True).input_ids
            
            outputs = model.generate(
                inputs, 
                max_new_tokens=256,
                use_cache=True,
                do_sample=False,
                repetition_penalty=1.0,
                # return_full_text=False, 
                temperature=0.0, 
            )
            text_responses = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            for ridx, text in enumerate(text_responses):
                response = text[len(messages[ridx]):]
                # print(response)
                # import pdb
                # pdb.set_trace()
                try:
                    start_idx = response.index("```json")
                    response = response[start_idx:]
                    json_data = eval(response.replace("```json", "").replace("```", ""))
                    questions[example_id].append({
                        "sentence_idx":ridx, 
                        "sentence":orig_sentences[ridx], 
                        "pairs": json_data
                    })
                except Exception as e:
                    # print(f"{example_id} || {e}")
                    invalid_examples.add(example_id)
                    continue
        #     text = tokenizer.decode(outputs[0, len(inputs[0]):], skip_special_tokens=True)
        #     start_idx = text.index("```json")
        #     text = text[start_idx:]
        #     try:
        #         json_data = eval(text.replace("```json", "").replace("```", ""))
        #         questions[example_id].append({"sentence_idx":idx, "sentence":str(sentence), "pairs": json_data})
        #     except:
        #         continue
        
        with open(f"/scratch/datasets/hackthon/docci_phi3_qa_pair_{example_id}.json" ,"w") as f:
            json.dump(questions[example_id], f, indent=4)

    with open("docci_phi3_qa_pairs.json" ,"w") as f:
        json.dump(questions, f, indent=4)