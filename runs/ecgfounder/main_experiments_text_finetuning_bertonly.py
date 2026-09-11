"""Finetuning A2: embeddings ECGFounder pré-computados (100% congelados) +
ClinicalBERT (últimas 2 camadas + pooler destravados) sobre texto DtT —
mesma arquitetura de main_experiments_llm_finetuning.py (MODEL_FM_LLM_Finetuning
não tem nenhum parâmetro de ECG treinável, então serve igual pra DtT ou LLM;
só muda a fonte de texto).
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

from Model.model import MODEL_FM_LLM_Finetuning
from engine.forward_adapters import FinetuningTextForward
from experiments.config import ExperimentConfig
from experiments.runner import run_experiment
from utils.ecg_sources import PrecomputedECGSource
from utils.text_sources import JSONTextSource

DATA_DIR = "/home/giovanidl/Datasets/PTBXL"
NPY_CACHE_DIR = "/home/giovanidl/doutorado/prelim/cache/npy_cache"
JSON_CACHE_DIR = "/home/giovanidl/doutorado/prelim/cache/json_cache"


def build_optimizer(model):
    return optim.AdamW([
        {"params": model.classifier.parameters(), "lr": 1e-3},
        {"params": model.text_encoder.language_model.encoder.layer[-2:].parameters(), "lr": 2e-5},
        {"params": model.text_encoder.language_model.pooler.parameters(), "lr": 2e-5},
    ])


if __name__ == "__main__":
    cfg = ExperimentConfig(
        name="CLICDtT_BertOnly_Finetuning_ECGFounder",
        data_dir=DATA_DIR,
        sampling_rate=500,
        ecg_source_fn=lambda split: PrecomputedECGSource(f"{NPY_CACHE_DIR}/ECGFounder_embeddings_{split}.npy"),
        text_source_fn=lambda split: JSONTextSource(f"{JSON_CACHE_DIR}/robust_text_cache_{split}.json"),
        model_factory=lambda: MODEL_FM_LLM_Finetuning(embedding_dim=1024, mlp_hidden=512),
        optimizer_factory=build_optimizer,
        loss_fn_factory=nn.BCEWithLogitsLoss,
        forward_fn=FinetuningTextForward(),
        save_dir="/home/giovanidl/doutorado/prelim/checkpoints/ECGText/",
        ckpt_name="CLICDtT_BertOnly_Finetuning_ECGFounder.pt",
        results_dir=os.path.join(PROJECT_ROOT, "results/ECGText/"),
    )

    run_experiment(cfg)
