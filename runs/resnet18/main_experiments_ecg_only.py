"""ResNet18 treinada do zero, ponta a ponta, só com o sinal de ECG (sem texto).
Baseline: o que um encoder dedicado aprende só com os dados rotulados do PTB-XL.
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

from Model.model import MODEL
from engine.forward_adapters import precomputed_forward
from experiments.config import ExperimentConfig
from experiments.runner import run_experiment
from utils.ecg_sources import RawSignalSource
from utils.text_sources import NullTextSource

DATA_DIR = "/home/giovanidl/Datasets/PTBXL"

if __name__ == "__main__":
    cfg = ExperimentConfig(
        name="ResNet18_ECGOnly",
        data_dir=DATA_DIR,
        sampling_rate=500,
        ecg_source_fn=lambda split: RawSignalSource(DATA_DIR),
        text_source_fn=lambda split: NullTextSource(),
        model_factory=lambda: MODEL(embedding_dim=512, mlp_hidden=512),
        optimizer_factory=lambda model: torch.optim.Adam(model.parameters(), lr=1e-3),
        loss_fn_factory=nn.BCEWithLogitsLoss,
        forward_fn=precomputed_forward,
        save_dir="/home/giovanidl/doutorado/prelim/checkpoints/ECG/",
        ckpt_name="ResNet18_ECGOnly.pt",
        results_dir=os.path.join(PROJECT_ROOT, "results/ECG/"),
    )

    run_experiment(cfg)
