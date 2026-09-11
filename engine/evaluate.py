# -*- coding = utf-8 -*-
# @File : evaluate.py
"""evaluate_model único (multilabel), substitui teste.py e teste_finetuning.py.

Sem argmax: aplica sigmoid + threshold por classe e usa classification_report
do sklearn diretamente sobre as matrizes indicadoras multilabel. Reporta
precision/recall/F1 por classe (como antes) e adiciona uma coluna 'auroc'
por classe + macro avg, pra manter o mesmo formato de DataFrame que os
main_experiments_*.py já usam pra tirar média/desvio-padrão entre as runs
(pd.concat(df_list).groupby(level=0).mean()/.std()).
"""
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, roc_auc_score
from tqdm import tqdm


def evaluate_model(model, dataloader, device, forward_fn, categories, threshold=0.5):
    print("=" * 30)
    print("INICIANDO AVALIAÇÃO NO TEST SET")
    print("=" * 30)

    model.eval()

    all_probs = []
    all_targets = []

    loop = tqdm(dataloader, file=sys.stdout, desc="Testing")
    with torch.no_grad():
        for batch in loop:
            logits, targets = forward_fn(model, batch, device)
            probs = torch.sigmoid(logits)

            all_probs.append(probs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    probs = np.concatenate(all_probs, axis=0)
    targets = np.concatenate(all_targets, axis=0).astype(int)
    preds = (probs >= threshold).astype(int)

    print("\n" + "-" * 20)
    print(f"RELATÓRIO DETALHADO POR CLASSE (multilabel, threshold={threshold}):")
    print("-" * 20)
    report_dict = classification_report(
        targets, preds, target_names=categories, digits=4, zero_division=0, output_dict=True
    )
    print(classification_report(targets, preds, target_names=categories, digits=4, zero_division=0))

    auroc_per_class = roc_auc_score(targets, probs, average=None)
    macro_auroc = float(np.mean(auroc_per_class))
    micro_auroc = roc_auc_score(targets, probs, average="micro")
    print(f"AUROC por classe: {dict(zip(categories, auroc_per_class))}")
    print(f"Macro AUROC: {macro_auroc:.4f} | Micro AUROC: {micro_auroc:.4f}")

    df = pd.DataFrame(report_dict).transpose()
    df["auroc"] = np.nan
    for cat, auroc in zip(categories, auroc_per_class):
        df.loc[cat, "auroc"] = auroc
    df.loc["macro avg", "auroc"] = macro_auroc

    return df, probs, preds, targets
