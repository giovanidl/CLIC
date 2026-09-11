"""Pré-computa o sinal de ECG já filtrado/normalizado (ecgfounder_preprocess)
pra cada split do PTB-XL, e salva em .npy — pra ser consumido por
CachedSignalSource. Elimina o custo de refazer o filtro passa-banda (caro,
CPU, síncrono) em toda amostra de toda época nos scripts de finetuning que
rodam o backbone ECGFounder sobre sinal bruto
(main_experiments_text_finetuning.py, main_experiments_llm_finetuning_lastlayer.py).
"""
import os

import numpy as np
import pandas as pd
import torch
import wfdb
from tqdm import tqdm

from utils.ecg_sources import ecgfounder_preprocess

DATA_DIR = "/home/giovanidl/Datasets/PTBXL"
OUTPUT_DIR = "/home/giovanidl/doutorado/prelim/cache/npy_cache"


def build_split_metadata(split):
    metadata = pd.read_csv(os.path.join(DATA_DIR, "ptbxl_database.csv"))
    if split == "train":
        metadata = metadata[metadata["strat_fold"] < 9]
    elif split == "val":
        metadata = metadata[metadata["strat_fold"] == 9]
    elif split == "test":
        metadata = metadata[metadata["strat_fold"] == 10]
    return metadata.reset_index(drop=True)


def generate_cache(split):
    metadata = build_split_metadata(split)
    cache = {}

    print(f"Gerando cache de sinal filtrado pro split={split} ({len(metadata)} registros)")
    for _, record in tqdm(metadata.iterrows(), total=len(metadata)):
        ecg_fn = record["filename_hr"]
        file_path = os.path.join(DATA_DIR, ecg_fn)
        signal, _ = wfdb.rdsamp(file_path)
        signal_t = torch.tensor(signal.T, dtype=torch.float32)
        cache[ecg_fn] = ecgfounder_preprocess(signal_t).numpy()

    output_path = os.path.join(OUTPUT_DIR, f"ecgfounder_filtered_signal_{split}.npy")
    np.save(output_path, cache)
    print(f"Salvo {len(cache)} registros em {output_path}")


def main():
    for split in ("train", "val", "test"):
        generate_cache(split)


if __name__ == "__main__":
    main()
