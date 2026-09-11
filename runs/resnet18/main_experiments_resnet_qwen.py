"""ResNet18 treinada do zero (sinal bruto) + texto clínico gerado pelo Qwen
(qwen3.5:9b) via ClinicalBERT CONGELADO (embeddings pré-computados),
concatenação simples. Mesma arquitetura de main_experiments_resnet_llm.py
(que usa o Llama 3.1:8b); só muda a fonte de texto — serve pra comparar se
um LLM mais forte no lugar do Llama traz algum ganho, agora com o encoder
ResNet18 (em vez do ECGFounder de main_experiments_qwen.py).
"""
import os
import sys

# Permite rodar este script de qualquer lugar (ex: python3 runs/resnet18/arquivo.py)
# sem quebrar os imports de Model/utils/engine/experiments, que sao relativos
# a raiz do projeto, nao a pasta deste arquivo.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

import torch
import torch.nn as nn

from Model.model import ECGTextFusion
from engine.forward_adapters import precomputed_forward
from experiments.config import ExperimentConfig
from experiments.runner import run_experiment
from utils.ecg_sources import RawSignalSource
from utils.text_sources import NPYTextSource

DATA_DIR = "/home/giovanidl/Datasets/PTBXL"
CACHE_DIR = "/home/giovanidl/doutorado/prelim/cache/npy_cache"

if __name__ == "__main__":
    cfg = ExperimentConfig(
        name="ResNet18_CLICQwen",
        data_dir=DATA_DIR,
        sampling_rate=500,
        ecg_source_fn=lambda split: RawSignalSource(DATA_DIR),
        text_source_fn=lambda split: NPYTextSource(f"{CACHE_DIR}/qwen_text_embeddings_{split}.npy"),
        model_factory=lambda: ECGTextFusion(embedding_dim=512, mlp_hidden=512),
        optimizer_factory=lambda model: torch.optim.Adam(model.parameters(), lr=1e-3),
        loss_fn_factory=nn.BCEWithLogitsLoss,
        forward_fn=precomputed_forward,
        save_dir="/home/giovanidl/doutorado/prelim/checkpoints/ECGQwen/",
        ckpt_name="ResNet18_CLICQwen.pt",
        results_dir=os.path.join(PROJECT_ROOT, "results/ECGQwen/"),
    )

    run_experiment(cfg)
