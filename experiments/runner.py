# -*- coding = utf-8 -*-
# @File : runner.py
"""run_experiment(cfg): o loop de N_RUNS que hoje está duplicado (com
pequenas variações) em cada main_experiments_*.py. Monta datasets/model a
partir da ExperimentConfig, aplica warm-start se houver, treina, avalia no
test set e agrega os relatórios (mean/std) entre as runs — exatamente a
mesma lógica de pd.concat(df_list).groupby(level=0).mean()/.std() que os
scripts antigos já usavam, agora com AUROC por classe incluído no relatório.
"""
import os
import random

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from engine.evaluate import evaluate_model
from engine.train import train_loop
from utils.dataset import PTBXLDataset


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def apply_warm_start(model, warm_start, device):
    checkpoint = torch.load(warm_start.checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint["model_state_dict"]

    for prefix in warm_start.submodules:
        sub_state = {
            k[len(prefix) + 1:]: v
            for k, v in state_dict.items()
            if k.startswith(prefix + ".")
        }
        submodule = getattr(model, prefix)
        submodule.load_state_dict(sub_state, strict=warm_start.strict)
        print(f"[warm_start] Carregado '{prefix}' de {warm_start.checkpoint_path}")


def build_datasets(cfg):
    datasets = {}
    for split in ("train", "val", "test"):
        datasets[split] = PTBXLDataset(
            data_dir=cfg.data_dir,
            ecg_source=cfg.ecg_source_fn(split),
            text_source=cfg.text_source_fn(split),
            split=split,
            sampling_rate=cfg.sampling_rate,
            categories=cfg.categories,
        )
    return datasets


def run_experiment(cfg):
    device = cfg.device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    datasets = build_datasets(cfg)
    print("Train size:", len(datasets["train"]))
    print("Val size:", len(datasets["val"]))
    print("Test size:", len(datasets["test"]))

    df_list = []
    for run in range(cfg.n_runs):
        seed = cfg.base_seed + run
        print(f"\n===== {cfg.name} | RUN {run + 1}/{cfg.n_runs} | seed={seed} =====")
        set_seed(seed)

        train_loader = DataLoader(datasets["train"], batch_size=cfg.batch_size, shuffle=True)
        val_loader = DataLoader(datasets["val"], batch_size=cfg.batch_size, shuffle=False)
        test_loader = DataLoader(datasets["test"], batch_size=cfg.batch_size, shuffle=False)

        model = cfg.model_factory().to(device)
        if cfg.warm_start is not None:
            apply_warm_start(model, cfg.warm_start, device)

        optimizer = cfg.optimizer_factory(model)
        loss_fn = cfg.loss_fn_factory()

        print("\nParâmetros treináveis:")
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(name)

        run_ckpt_name = f"run{run + 1}_{cfg.ckpt_name}"

        metrics_df = train_loop(
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            forward_fn=cfg.forward_fn,
            num_labels=len(cfg.categories),
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            epochs=cfg.epochs,
            save_path=cfg.save_dir,
            ckpt_name=run_ckpt_name,
            patience=cfg.patience,
            monitor=cfg.monitor,
            mode=cfg.mode,
            device=device,
        )
        os.makedirs(cfg.save_dir, exist_ok=True)
        metrics_df.to_csv(
            os.path.join(cfg.save_dir, f"{cfg.name}_run{run + 1}_history.csv"), index=False
        )

        # train_loop deixa "model" no estado da ÚLTIMA época treinada, não
        # necessariamente a de melhor val_auroc (o early stopping só para o
        # treino, não reverte os pesos). Recarrega o melhor checkpoint salvo
        # em disco antes de avaliar no test set.
        best_ckpt_path = os.path.join(cfg.save_dir, run_ckpt_name)
        best_ckpt = torch.load(best_ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(best_ckpt["model_state_dict"])

        report_df, probs, preds, targets = evaluate_model(
            model, test_loader, device, cfg.forward_fn, cfg.categories, threshold=cfg.threshold
        )
        df_list.append(report_df)

    mean_report = pd.concat(df_list).groupby(level=0).mean(numeric_only=True)
    std_report = pd.concat(df_list).groupby(level=0).std(numeric_only=True)

    os.makedirs(cfg.results_dir, exist_ok=True)
    mean_report.to_csv(os.path.join(cfg.results_dir, f"{cfg.name}_mean.csv"))
    std_report.to_csv(os.path.join(cfg.results_dir, f"{cfg.name}_std.csv"))

    print("\n===== MÉDIA (todas as runs) =====")
    print(mean_report)
    print("\n===== DESVIO PADRÃO (todas as runs) =====")
    print(std_report)

    return mean_report, std_report
