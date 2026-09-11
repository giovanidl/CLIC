# -*- coding = utf-8 -*-
# @File : ecg_sources.py
"""Estratégias de obtenção da representação de ECG para PTBXLDataset.

Cada source expõe __call__(record) -> torch.Tensor, isolando a diferença
entre ler o sinal bruto (para treinar/finetunar um encoder) e carregar um
embedding pré-computado (ex: ECGFounder congelado).
"""
import os

import numpy as np
import torch
import wfdb

from ECGFounder.util import filter_bandpass


def ecgfounder_preprocess(signal):
    """Replica o pré-processamento usado em precompute_ecg_embeddings.py pra
    gerar os embeddings ECGFounder (.npy): filtro passa-banda + normalização
    z-score. Necessário como `transform` do RawSignalSource sempre que o
    sinal bruto alimenta um encoder ECGFounder (mesmo parcialmente
    congelado) — sem isso a entrada fica fora da distribuição que os pesos
    pré-treinados do ECGFounder esperam.
    """
    arr = signal.numpy()
    arr = np.nan_to_num(arr, nan=0)
    arr = filter_bandpass(arr, 500)
    arr = (arr - np.mean(arr)) / (np.std(arr) + 1e-8)
    return torch.tensor(arr, dtype=torch.float32)


class RawSignalSource:
    """Lê o sinal de ECG (12 derivações) direto do WFDB."""

    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform

    def __call__(self, record):
        file_path = os.path.join(self.data_dir, record["file_path"])
        signal, _ = wfdb.rdsamp(file_path)
        signal = torch.tensor(signal.T, dtype=torch.float32)  # (12, L)
        if self.transform:
            signal = self.transform(signal)
        return signal


class PrecomputedECGSource:
    """Carrega embeddings de ECG pré-computados (.npy, dict {ecg_id: {"embedding": ...}})."""

    def __init__(self, npy_path):
        self.embeddings = np.load(npy_path, allow_pickle=True).item()

    def __call__(self, record):
        ecg_id = record["filename_hr"]
        return torch.tensor(self.embeddings[ecg_id]["embedding"], dtype=torch.float32)


class CachedSignalSource:
    """Carrega sinal de ECG já filtrado/normalizado (ecgfounder_preprocess)
    e pré-computado (.npy, dict {ecg_id: array}), gerado por
    precompute_ecgfounder_signal_cache.py. Evita refazer o filtro
    passa-banda (caro, CPU, síncrono) em toda amostra de toda época — o
    pré-processamento não muda entre runs, então só precisa ser feito uma
    vez.
    """

    def __init__(self, npy_path):
        self.signals = np.load(npy_path, allow_pickle=True).item()

    def __call__(self, record):
        ecg_id = record["filename_hr"]
        return torch.tensor(self.signals[ecg_id], dtype=torch.float32)
