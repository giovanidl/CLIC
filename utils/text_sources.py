# -*- coding = utf-8 -*-
# @File : text_sources.py
"""Estratégias de obtenção da representação de texto para PTBXLDataset.

Cada source expõe __call__(record) -> objeto consumido pelo forward_adapter
correspondente em engine/forward_adapters.py: texto cru (para tokenizar ao
vivo durante finetuning) ou um embedding já pronto (frozen encoder).
"""
import json

import numpy as np
import torch


class RawTextSource:
    """Retorna o texto clínico bruto do registro (tokenizado no forward)."""

    def __init__(self, text_fn):
        """text_fn(record) -> str, ex: gerador de texto demográfico/DtT."""
        self.text_fn = text_fn

    def __call__(self, record):
        return self.text_fn(record)


class JSONTextSource:
    """Carrega um cache de texto bruto (.json, {ecg_id: str}) pré-gerado."""

    def __init__(self, json_path):
        with open(json_path) as f:
            self.text_cache = json.load(f)

    def __call__(self, record):
        return self.text_cache[record["filename_hr"]]


class NPYTextSource:
    """Carrega embeddings de texto pré-computados (.npy, {ecg_id: vetor})."""

    def __init__(self, npy_path):
        self.embeddings = np.load(npy_path, allow_pickle=True).item()

    def __call__(self, record):
        return torch.tensor(
            self.embeddings[record["filename_hr"]], dtype=torch.float32
        )


class NullTextSource:
    """Placeholder pra modelos ECG-only: PTBXLDataset sempre retorna um
    (ecg_repr, text_repr, label), então mesmo modelos que ignoram o texto
    (ex: MODEL) precisam de algo aqui só pra manter o formato do batch.
    """

    def __call__(self, record):
        return torch.zeros(1, dtype=torch.float32)
