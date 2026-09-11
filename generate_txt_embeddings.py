import json
import numpy as np
import torch
from transformers import BertTokenizer
from Model.model import MODEL_RobustText, LanguageModel
import tqdm
import sys
device = "cuda"
def generate_embeddings(cache_filename="text_cache.json", output_filename="BUFFEDtext_embeddings.npy"):
    print("Generating embeddings from cache:", cache_filename.split("/")[-1])
    with open(cache_filename) as f:
        text_cache = json.load(f)

    tokenizer = BertTokenizer.from_pretrained(
        "emilyalsentzer/Bio_ClinicalBERT"
    )

    text_encoder = LanguageModel(freeze=False).to(device)
    text_encoder.eval()

    embeddings = {}
    ids = []

    with torch.no_grad():
        for ecg_id, text in tqdm.tqdm(text_cache.items()):
            tokens = tokenizer(
                text,
                padding="max_length",
                truncation=True,
                max_length=100,
                return_tensors="pt"
            )

            input_ids = tokens["input_ids"].to(device)
            attn_mask = tokens["attention_mask"].to(device)

            emb = text_encoder(input_ids, attn_mask)
            emb = emb.squeeze(0).cpu().numpy()

            embeddings[ecg_id] = emb
            ids.append(ecg_id)

    # salva tudo junto
    np.save(output_filename, embeddings)
    
def generate_embeddings2(cache_filename="text_cache.json", output_filename="ATTRtext_embeddings.npy"):
    print("Generating embeddings from cache:", cache_filename.split("/")[-1])
    with open(cache_filename) as f:
        text_cache = json.load(f)


    embeddings = {}
    ids = []


    for ecg_id, text in tqdm.tqdm(text_cache.items()):

        text = list(map(float, text.split(",")))

        embeddings[ecg_id] = text
        ids.append(ecg_id)

    print("Sample embedding:", embeddings[ids[0]])
    # salva tudo junto
    np.save(output_filename, embeddings)

def main():

    # ja gerados anteriormente:
    # generate_embeddings(cache_filename=".../llm_text_cache_{split}.json", output_filename=".../llm_text_embeddings_{split}.npy")
    # generate_embeddings(cache_filename=".../robust_text_cache_{split}.json", output_filename=".../robust_text_embeddings_{split}.npy")
    # generate_embeddings2(cache_filename=".../ATTRcache_{split}.json", output_filename=".../ATTRtext_embeddings_{split}.npy")

    #Generate Qwen (qwen3.5:9b) text embeddings for train, val, test
    generate_embeddings(cache_filename="/home/giovanidl/doutorado/prelim/cache/json_cache/qwen_text_cache_train.json",
                        output_filename="/home/giovanidl/doutorado/prelim/cache/npy_cache/qwen_text_embeddings_train.npy")
    generate_embeddings(cache_filename="/home/giovanidl/doutorado/prelim/cache/json_cache/qwen_text_cache_val.json",
                        output_filename="/home/giovanidl/doutorado/prelim/cache/npy_cache/qwen_text_embeddings_val.npy")
    generate_embeddings(cache_filename="/home/giovanidl/doutorado/prelim/cache/json_cache/qwen_text_cache_test.json",
                        output_filename="/home/giovanidl/doutorado/prelim/cache/npy_cache/qwen_text_embeddings_test.npy")

    #Generate Gemma (gemma4:12b) text embeddings for train, val, test
    generate_embeddings(cache_filename="/home/giovanidl/doutorado/prelim/cache/json_cache/gemma_text_cache_train.json",
                        output_filename="/home/giovanidl/doutorado/prelim/cache/npy_cache/gemma_text_embeddings_train.npy")
    generate_embeddings(cache_filename="/home/giovanidl/doutorado/prelim/cache/json_cache/gemma_text_cache_val.json",
                        output_filename="/home/giovanidl/doutorado/prelim/cache/npy_cache/gemma_text_embeddings_val.npy")
    generate_embeddings(cache_filename="/home/giovanidl/doutorado/prelim/cache/json_cache/gemma_text_cache_test.json",
                        output_filename="/home/giovanidl/doutorado/prelim/cache/npy_cache/gemma_text_embeddings_test.npy")



if __name__ == "__main__":
    main()