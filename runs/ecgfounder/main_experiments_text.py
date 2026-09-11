"""ECGFounder (embeddings pré-computados) + texto clínico DtT (Data-to-Text),
embeddings de texto pré-computados via ClinicalBERT congelado.
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

from Model.model import MODEL_RobustText_FM
from engine.forward_adapters import precomputed_forward
from experiments.config import ExperimentConfig
from experiments.runner import run_experiment
from utils.ecg_sources import PrecomputedECGSource
from utils.text_sources import NPYTextSource

DATA_DIR = "/home/giovanidl/Datasets/PTBXL"
CACHE_DIR = "/home/giovanidl/doutorado/prelim/cache/npy_cache"

if __name__ == "__main__":
    cfg = ExperimentConfig(
        name="CLICDtT_ECGFounder",
        data_dir=DATA_DIR,
        sampling_rate=500,
        ecg_source_fn=lambda split: PrecomputedECGSource(f"{CACHE_DIR}/ECGFounder_embeddings_{split}.npy"),
        text_source_fn=lambda split: NPYTextSource(f"{CACHE_DIR}/robust_text_embeddings_{split}.npy"),
        model_factory=lambda: MODEL_RobustText_FM(embedding_dim=1024, mlp_hidden=512),
        optimizer_factory=lambda model: torch.optim.Adam(model.parameters(), lr=1e-3),
        loss_fn_factory=nn.BCEWithLogitsLoss,
        forward_fn=precomputed_forward,
        save_dir="/home/giovanidl/doutorado/prelim/checkpoints/ECGText/",
        ckpt_name="CLICDtT_ECGFounder.pt",
        results_dir=os.path.join(PROJECT_ROOT, "results/ECGText/"),
    )

    run_experiment(cfg)
