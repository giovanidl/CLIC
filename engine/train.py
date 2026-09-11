# -*- coding = utf-8 -*-
# @File : train.py (engine)
"""train_loop único (multilabel), substitui train.py e train_finetuning.py."""
import os

import pandas as pd
import torch

from engine.steps import EpochRunner, StepRunner


def train_loop(
    model,
    optimizer,
    loss_fn,
    forward_fn,
    num_labels,
    train_dataloader,
    val_dataloader=None,
    epochs=10,
    save_path="./checkpoint/",
    ckpt_name="checkpoint.pt",
    patience=5,
    monitor="val_auroc",
    mode="max",
    device=None,
):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_history = {}

    print("=" * 25 + "Start Training" + "=" * 25)

    best_score = float("-inf") if mode == "max" else float("inf")
    epochs_no_improve = 0

    for epoch in range(1, epochs + 1):
        print(f"\nEpoch {epoch}/{epochs}")

        train_step_runner = StepRunner(
            model=model, loss_fn=loss_fn, forward_fn=forward_fn, device=device,
            stage="train", optimizer=optimizer,
        )
        train_epoch_runner = EpochRunner(train_step_runner, num_labels)
        train_metrics = train_epoch_runner(train_dataloader)

        for name, metric in train_metrics.items():
            train_history[name] = train_history.get(name, []) + [metric]

        if val_dataloader:
            val_step_runner = StepRunner(
                model=model, loss_fn=loss_fn, forward_fn=forward_fn, device=device, stage="val"
            )
            val_epoch_runner = EpochRunner(val_step_runner, num_labels)
            with torch.no_grad():
                val_metrics = val_epoch_runner(val_dataloader)

            for name, metric in val_metrics.items():
                train_history[name] = train_history.get(name, []) + [metric]

            print(
                f"TRAIN -> Loss: {train_metrics['train_loss']:.4f} | "
                f"F1: {train_metrics['train_f1']:.4f} | AUROC: {train_metrics['train_auroc']:.4f}"
            )
            print(
                f"VAL   -> Loss: {val_metrics['val_loss']:.4f} | "
                f"F1: {val_metrics['val_f1']:.4f} | AUROC: {val_metrics['val_auroc']:.4f}"
            )

            current_score = val_metrics[monitor]
            is_better = (
                current_score > best_score if mode == "max" else current_score < best_score
            )

            if is_better:
                best_score = current_score
                epochs_no_improve = 0

                os.makedirs(save_path, exist_ok=True)
                ckpt_path = os.path.join(save_path, ckpt_name)

                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "monitor_val": current_score,
                    },
                    ckpt_path,
                )
                print(f">>> Melhor {monitor} alcançado! Checkpoint salvo: {ckpt_name}")
            else:
                epochs_no_improve += 1
                print(f">>> Sem melhoria em {monitor} por {epochs_no_improve} épocas.")

            if epochs_no_improve >= patience:
                print(">>> Early Stopping ativado.")
                break

    return pd.DataFrame(train_history)
