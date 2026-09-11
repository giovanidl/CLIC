import os
import wfdb
import torch
import numpy as np
import pandas as pd
import tqdm
import sys
from ECGFounder.util import filter_bandpass              # do repo oficial ECGFounder
from Model.model import MODEL_FM_RobustText_Finetuning    # de onde vem _load_ecgfounder_backbone

device = "cuda"



def preprocess_signal(file_path):
    signal, meta = wfdb.rdsamp(file_path)
    
    signal = np.transpose(signal, (1, 0))
    signal = np.nan_to_num(signal, nan=0)

    
    signal = filter_bandpass(signal, 500)
    signal = (signal - np.mean(signal)) / (np.std(signal) + 1e-8)
    return signal.astype(np.float32)


def generate_ecg_embeddings(
    data_dir,
    metadata_df,
    ckpt_path="./checkpoint/12_lead_ECGFounder.pth",
    output_filename="ecg_embeddings.npy"
):
    print("Generating ECG embeddings for split with", len(metadata_df), "records")

    ecg_encoder, embedding_dim = MODEL_FM_RobustText_Finetuning._load_ecgfounder_backbone(
        device=device, pth=ckpt_path, freeze=True
    )
    ecg_encoder.eval()
    #print("Embedding dim:", embedding_dim)

    embeddings = {}
    ids = []

    with torch.no_grad():
        for idx, record in tqdm.tqdm(metadata_df.iterrows(), total=len(metadata_df)):
            ecg_fn = record["filename_hr"]  # mesmo ID usado no cache de texto -> garante alinhamento
            file_path = os.path.join(data_dir, ecg_fn)
            
            signal = preprocess_signal(file_path)
            print(signal.shape)

            signal_t = torch.tensor(signal, dtype=torch.float32).unsqueeze(0).to(device)  # (1,12,5000)
            #emb = ecg_encoder(signal_t)
            #emb = emb.squeeze(0).cpu().numpy()
            sys.exit(0)
            embeddings[ecg_fn] = {
                #"patient_id": record["patient_id"],
                "file_path": signal,
                "embedding": emb
            }
            ids.append(ecg_fn)
            #print("Sample embedding for first record:", embeddings[ids[0]])

    # print("Sample embedding shape:", embeddings[ids[0]]["embedding"].shape)
    # print(embeddings)
    # print("Sample embedding for first record:", embeddings[ids[0]])  # print first 5 values
    # salva tudo junto

    np.save(output_filename, embeddings)
    #print(f"Salvo {len(embeddings)} registros em {output_filename}")


def main():
    DATA_DIR = "/home/giovanidl/Datasets/PTBXL"
    CKPT_PATH = "/home/giovanidl/doutorado/prelim/ECGFounder/12_lead_ECGFounder.pth"

    metadata = pd.read_csv(os.path.join(DATA_DIR, "ptbxl_database.csv"))

    train_df = metadata[metadata["strat_fold"] < 9].reset_index(drop=True)
    val_df = metadata[metadata["strat_fold"] == 9].reset_index(drop=True)
    test_df = metadata[metadata["strat_fold"] == 10].reset_index(drop=True)

    # Generate ECGFounder embeddings for train, val, test
    generate_ecg_embeddings(
        data_dir=DATA_DIR,
        metadata_df=train_df,
        ckpt_path=CKPT_PATH,
        output_filename="/home/giovanidl/doutorado/prelim/cache/npy_cache/ECGFounder_preprocess_train.npy"
    )
    generate_ecg_embeddings(
        data_dir=DATA_DIR,
        metadata_df=val_df,
        ckpt_path=CKPT_PATH,
        output_filename="/home/giovanidl/doutorado/prelim/cache/npy_cache/ECGFounder_preprocess_val.npy"
    )
    generate_ecg_embeddings(
        data_dir=DATA_DIR,
        metadata_df=test_df,
        ckpt_path=CKPT_PATH,
        output_filename="/home/giovanidl/doutorado/prelim/cache/npy_cache/ECGFounder_preprocess_test.npy"
    )


if __name__ == "__main__":
    main()