"""Finetuning A1: ResNet18 100% CONGELADA (a partir do melhor checkpoint
entre as 5 runs de main_experiments_resnet_dtt.py, só em modo inferência) +
ClinicalBERT (últimas 2 camadas + pooler destravados) — só o texto finetuna.
"""
import os
import sys

# Permite rodar este script de qualquer lugar (ex: python3 runs/resnet18/arquivo.py)
# sem quebrar os imports de Model/utils/engine/experiments, que sao relativos
# a raiz do projeto, nao a pasta deste arquivo.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

import torch.nn as nn
import torch.optim as optim

from Model.model import MODEL_RobustText_Finetuning
from engine.forward_adapters import FinetuningTextForward
from experiments.config import ExperimentConfig, WarmStart
from experiments.runner import run_experiment
from utils.ecg_sources import RawSignalSource
from utils.text_sources import JSONTextSource

DATA_DIR = "/home/giovanidl/Datasets/PTBXL"
JSON_CACHE_DIR = "/home/giovanidl/doutorado/prelim/cache/json_cache"

PRETRAIN_SAVE_DIR = "/home/giovanidl/doutorado/prelim/checkpoints/ECGText/"
PRETRAIN_CKPT_NAME = "ResNet18_CLICDtT.pt"
PRETRAIN_N_RUNS = 5


def build_optimizer(model):
    # ecg_encoder inteiro está congelado (unfreeze_ecg_layers=0) — nenhum
    # grupo de parâmetros dele entra no otimizador.
    return optim.AdamW([
        {"params": model.classifier.parameters(), "lr": 1e-3},
        {"params": model.text_encoder.language_model.encoder.layer[-2:].parameters(), "lr": 2e-5},
        {"params": model.text_encoder.language_model.pooler.parameters(), "lr": 2e-5},
    ])


if __name__ == "__main__":
    cfg = ExperimentConfig(
        name="ResNet18_CLICDtT_BertOnly_Finetuning",
        data_dir=DATA_DIR,
        sampling_rate=500,
        ecg_source_fn=lambda split: RawSignalSource(DATA_DIR),
        text_source_fn=lambda split: JSONTextSource(f"{JSON_CACHE_DIR}/robust_text_cache_{split}.json"),
        model_factory=lambda: MODEL_RobustText_Finetuning(embedding_dim=512, mlp_hidden=512, unfreeze_ecg_layers=0),
        optimizer_factory=build_optimizer,
        loss_fn_factory=nn.BCEWithLogitsLoss,
        forward_fn=FinetuningTextForward(),
        warm_start=WarmStart.best_of(
            save_dir=PRETRAIN_SAVE_DIR,
            ckpt_name=PRETRAIN_CKPT_NAME,
            n_runs=PRETRAIN_N_RUNS,
            submodules=["ecg_encoder"],
        ),
        save_dir="/home/giovanidl/doutorado/prelim/checkpoints/ECGText/",
        ckpt_name="ResNet18_CLICDtT_BertOnly_Finetuning.pt",
        results_dir=os.path.join(PROJECT_ROOT, "results/ECGText/"),
    )

    run_experiment(cfg)
