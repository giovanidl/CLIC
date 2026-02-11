import json
from tqdm import tqdm
from utils.dataset import PTBXLDataset_with_generated_prompt, PTBXLDatasetWithTextEmbeddingNPY
import numpy as np
import sys
import os
import wfdb
import torch

from torch.utils.data import DataLoader
from Model.ECG_encoder.resnet1d import resnet18_1d

def generate_ecg_embbeding(dataloader, output_filename="ecg_embeddings_GIOVANI.npy"):
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    all_embeddings = []
    
    

    with torch.no_grad():
        for batch in tqdm(dataloader):
            # Ajuste se o batch tiver mais elementos
            ecg, embeddings, targets = batch  
            ecg_encoder = resnet18_1d(in_channels=12, projection_size=512)
            ecg_features = ecg_encoder(ecg)
            ecg_features = ecg_features.to(device)

            all_embeddings.append(ecg_features.cpu().numpy())
    
    # for idx in tqdm(range(len(dataset))):
        
    #     record = dataset.records.iloc[idx]
         
    #     file_path = os.path.join(data_dir, record["file_path"])
    #     ecg_id = record["filename_hr"]  # use um ID único
    #     # Lê o sinal de ECG com o wfdb
    #     signal, _ = wfdb.rdsamp(file_path)
        
    #     signal = torch.tensor(signal.T, dtype=torch.float32)  # shape (12, length)
        
    #     labels = dataset._extract_labels(record["scp_codes"])
    #     ecg_encoder = resnet18_1d(in_channels=12, projection_size=512)
    #     ecg_features = ecg_encoder(signal)
    #     print(ecg_features.shape)
    #     print(labels)
    #     # embeddings[ecg_id] = ecg_features
    #     # print(signal.shape)
    
    
    np.save(output_filename, embeddings)
        





def main():
    
    train_dataset_robust_text = PTBXLDatasetWithTextEmbeddingNPY(
        data_dir="/home/giovanidl/Datasets/PTBXL",
        text_embedding_path="/home/giovanidl/doutorado/prelim/cache/npy_cache/robust_text_embeddings_train.npy",
        split="train",
        sampling_rate=500
    )
    val_dataset_robust_text = PTBXLDatasetWithTextEmbeddingNPY(
        data_dir="/home/giovanidl/Datasets/PTBXL",
        text_embedding_path="/home/giovanidl/doutorado/prelim/cache/npy_cache/robust_text_embeddings_val.npy",
        split="val",
        sampling_rate=500
    )

    test_dataset_robust_text = PTBXLDatasetWithTextEmbeddingNPY(
        data_dir="/home/giovanidl/Datasets/PTBXL",
        text_embedding_path="/home/giovanidl/doutorado/prelim/cache/npy_cache/robust_text_embeddings_test.npy",
        split="test",
        sampling_rate=500
        )

    train_loader_robust_text = DataLoader(train_dataset_robust_text, batch_size=16, shuffle=False)
    val_loader_robust_text = DataLoader(val_dataset_robust_text, batch_size=16, shuffle=False)
    test_loader_robust_text = DataLoader(test_dataset_robust_text, batch_size=16, shuffle=False)


    generate_ecg_embbeding(train_loader_robust_text, output_filename="/home/giovanidl/doutorado/prelim/cache/npy_cache/ecg_embeddings_train.npy")
    # generate_ecg_embbeding(split='val', output_filename="/home/giovanidl/doutorado/prelim/cache/npy_cache/ecg_embeddings_val.npy")
    # generate_ecg_embbeding(split='test', output_filename="/home/giovanidl/doutorado/prelim/cache/npy_cache/ecg_embeddings_test.npy")

if __name__ == "__main__":
    main()
