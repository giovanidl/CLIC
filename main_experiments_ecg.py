import functools
import logging
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import logging as tflogging
import random
from Model.model import MODEL, MODEL_RobustText, ECGTextFusion
from train import train_loop
from teste import test_model
from utils.dataset import  PTBXLDataset, PTBXLDataset_with_prompt, PTBXLDataset_with_generated_prompt,PTBXLDatasetWithTextEmbeddingNPY
from utils.utils import get_smallest_loss_model_path, init_log
import torch.nn as nn
# from zero_shot_classification import zero_shot_classification
from sklearn.metrics import classification_report, accuracy_score, f1_score


def compute_global_metrics(y_true, y_pred):
    report = classification_report(
        y_true,
        y_pred,
        output_dict=True,
        zero_division=0
    )
    
    return {
        "accuracy": report["accuracy"],
        "macro_f1": report["macro avg"]["f1-score"],
        "weighted_f1": report["weighted avg"]["f1-score"]
    }

# -------------------------
# Utils
# -------------------------
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



# -------------------------
# Main
# -------------------------
if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    N_RUNS = 1
    BASE_SEED = 42

    results = []

    # -------------------------
    # Datasets
    # -------------------------
    train_dataset = PTBXLDatasetWithTextEmbeddingNPY(
        data_dir="/home/giovanidl/Datasets/PTBXL",
        text_embedding_path="/home/giovanidl/doutorado/prelim/cache/npy_cache/robust_text_embeddings_train.npy",
        split="train",
        sampling_rate=500
    )
    val_dataset = PTBXLDatasetWithTextEmbeddingNPY(
        data_dir="/home/giovanidl/Datasets/PTBXL",
        text_embedding_path="/home/giovanidl/doutorado/prelim/cache/npy_cache/robust_text_embeddings_val.npy",
        split="val",
        sampling_rate=500
    )

    test_dataset = PTBXLDatasetWithTextEmbeddingNPY(
        data_dir="/home/giovanidl/Datasets/PTBXL",
        text_embedding_path="/home/giovanidl/doutorado/prelim/cache/npy_cache/robust_text_embeddings_test.npy",
        split="test",
        sampling_rate=500
    )

    print("Train size:", len(train_dataset))
    print("Val size:", len(val_dataset))
    print("Test size:", len(test_dataset))

    # -------------------------
    # Experiments loop
    # -------------------------
    all_results = []
    df_list = []
    for run in range(N_RUNS):
        seed = BASE_SEED + run
        print(f"\n===== RUN {run+1}/{N_RUNS} | seed={seed} =====")

        set_seed(seed)

        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

        model = MODEL(embedding_dim=512, mlp_hidden=256)
        model = model.to(device)

        loss_fn = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        metrics_df = train_loop(
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            metrics_dict=None,
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            epochs=1000,
            save_path='/home/giovanidl/doutorado/prelim/checkpoints/ecg/',
            patience=50,
            monitor="val_loss",
            mode="min"
        )
        print(metrics_df)
        metrics_df.to_csv(f"/home/giovanidl/doutorado/prelim/checkpoints/ecg/metrics_run_{run+1}_MLP256_64.csv", index=False)
        # ----- Teste (usa SUA função) -----
        preds, targets = test_model(
            model,
            test_loader,
            device,
            class_names=['NORM', 'MI', 'STTC', 'CD', 'HYP']
        )
        # df = pd.DataFrame(classification_report(targets, preds, target_names=['NORM', 'MI', 'STTC', 'CD', 'HYP'], output_dict=True)).transpose()
        # df_list.append(df)
        

    # mean_report = pd.concat(df_list).groupby(level=0).mean()
    # std_report = pd.concat(df_list).groupby(level=0).std()

    # mean_report.to_csv("results/ECG/ECG_64mlp_5_1000_mean.csv")
    # std_report.to_csv("results/ECG/ECG_64mlp_5_1000_std.csv")
    # print("\n=====MEDIA =====")
    # print(mean_report)
    # print("\n=====STD =====")
    # print(std_report)
    
    
    # # -------------------------
    # # Aggregate results
    # # -------------------------
    # results_df = pd.DataFrame(results)
    # results_df.to_csv("summary_results_all_runs.csv", index=False)

    # mean_results = results_df.mean()
    # std_results = results_df.std()

    # summary_df = pd.DataFrame({
    #     "mean": mean_results,
    #     "std": std_results
    # })

    # summary_df.to_csv("summary_results_mean_std.csv")

    # print("\n===== FINAL RESULTS (mean ± std) =====")
    # print(summary_df)

    
  
