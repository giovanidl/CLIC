# -*- coding = utf-8 -*-
# @File : labels.py
"""Fonte única de verdade para o rótulo multilabel do PTB-XL.

Segue o padrão do benchmark oficial do PTB-XL (Strodthoff et al. / repo
ptb-xl-benchmarking): para cada registro, todo scp_code anotado com
diagnostic==1 contribui com um 1 na sua diagnostic_class; registros sem
nenhum código diagnóstico não têm rótulo válido e devem ser descartados.
"""
import ast
import os

import numpy as np
import pandas as pd

CATEGORIES = ["NORM", "MI", "STTC", "CD", "HYP"]


def load_diagnostic_label_map(data_dir):
    """scp_statements.csv filtrado às linhas diagnósticas (diagnostic == 1)."""
    label_map_geral = pd.read_csv(
        os.path.join(data_dir, "scp_statements.csv"), index_col=0
    )
    return label_map_geral, label_map_geral[label_map_geral.diagnostic == 1]


def parse_scp_codes(scp_codes_str):
    return list(ast.literal_eval(scp_codes_str).keys())


def extract_multihot(scp_codes_str, label_map, categories=CATEGORIES):
    """Retorna o vetor multi-hot das superclasses diagnósticas do registro.

    Retorna None se nenhum dos códigos anotados é diagnóstico (o registro
    deve ser descartado do dataset, não empurrado para uma classe default).
    """
    codes = parse_scp_codes(scp_codes_str)
    diagnostic_codes = [c for c in codes if c in label_map.index]
    if not diagnostic_codes:
        return None

    multihot = np.zeros(len(categories), dtype=np.float32)
    for code in diagnostic_codes:
        superclass = label_map.loc[code].diagnostic_class
        multihot[categories.index(superclass)] = 1.0
    return multihot


def build_split_records(data_dir, split, sampling_rate, categories=CATEGORIES):
    """Metadata do PTB-XL filtrada por split, com coluna 'label' (multi-hot)
    e 'file_path' resolvidos, descartando registros sem código diagnóstico.
    """
    metadata = pd.read_csv(os.path.join(data_dir, "ptbxl_database.csv"))

    if split == "train":
        metadata = metadata[metadata["strat_fold"] < 9]
    elif split == "val":
        metadata = metadata[metadata["strat_fold"] == 9]
    elif split == "test":
        metadata = metadata[metadata["strat_fold"] == 10]
    else:
        raise ValueError(f"split desconhecido: {split!r}")

    _, label_map = load_diagnostic_label_map(data_dir)

    labels = metadata["scp_codes"].apply(
        lambda s: extract_multihot(s, label_map, categories)
    )
    metadata = metadata[labels.notna()].copy()
    metadata["label"] = labels[labels.notna()]

    metadata["file_path"] = (
        metadata["filename_lr"] if sampling_rate == 100 else metadata["filename_hr"]
    )

    return metadata.reset_index(drop=True)
