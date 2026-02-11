import functools
import logging
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import logging as tflogging

from Model.model import MODEL, MODEL_RobustText
from train import train_loop
from teste import test_model
from utils.dataset import  PTBXLDataset, PTBXLDataset_with_prompt
from utils.utils import get_smallest_loss_model_path, init_log
import torch.nn as nn
# from zero_shot_classification import zero_shot_classification

# close BERT pretrain file loading warnings
tflogging.set_verbosity_error()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    #init_log()

    train_dataset = PTBXLDataset_with_prompt(data_dir="/home/giovanidl/Datasets/PTBXL", split="train", sampling_rate=250)
    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    for inputs, robust_text_input_ids, attention_mask, targets in train_dataloader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        robust_text_input_ids, attention_mask = robust_text_input_ids.to(device), attention_mask.to(device)

        
    
    print("Train dataset size:", len(train_dataset))
    
    # val_dataset = SSLECGTextDataset(num_samples, ecg_length)
    val_dataset = PTBXLDataset_with_prompt(data_dir="/home/giovanidl/Datasets/PTBXL", split="val", sampling_rate=250)
    val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=True)
    for inputs, robust_text_input_ids, attention_mask, targets in val_dataloader:
        
        inputs = inputs.to(device)
        targets = targets.to(device)
        robust_text_input_ids, attention_mask = robust_text_input_ids.to(device), attention_mask.to(device)

    print("Validation dataset size:", len(val_dataset))
    # test_dataset = ZeroShotTestECGTextDataset(num_samples, ecg_length)
    test_dataset = PTBXLDataset_with_prompt(data_dir="/home/giovanidl/Datasets/PTBXL", split="test", sampling_rate=250)
    test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=True)
    for inputs, robust_text_input_ids, attention_mask, targets in test_dataloader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        robust_text_input_ids, attention_mask = robust_text_input_ids.to(device), attention_mask.to(device)


    print("Test dataset size:", len(test_dataset))
    # Instancia o modelo
    # embedding_dim deve bater com a saída da resnet18_1d antes da projeção ou a dimensão definida
    model = MODEL_RobustText(embedding_dim=512, mlp_hidden=512, num_classes=5) 
    model = model.to(device)
        
    # Use BCEWithLogitsLoss pois seu target já é um vetor (one-hot)
    loss_fn = nn.BCEWithLogitsLoss()
    
    # 2. MUDANÇA CRÍTICA: Otimizador
    # Deve otimizar model.parameters() para incluir o classificador (a MLP nova)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Opcional: Adicionar métricas de acurácia
    # metrics_dict = {"acc": Accuracy(task="multiclass", num_classes=5).to(device)}

    # Chama o loop de treino (renomeei ssl_train para train_loop para fazer sentido)
    train_dfhistory = train_loop(model, # Pode renomear a função ssl_train depois
                                optimizer,
                                loss_fn,
                                metrics_dict=None, 
                                train_dataloader=train_dataloader,
                                val_dataloader=val_dataloader,
                                epochs=1,
                                patience=50,
                                monitor="val_loss",
                                mode="min")
    
    #print("\n" + train_dfhistory.to_string())
    
    
    preds, targets = test_model(model, test_dataloader, device)
    # Opcional: Salvar predições em CSV para análise posterior
    df = pd.DataFrame({'target': targets, 'pred': preds})
    df.to_csv('test_results.csv', index=False)
    print("Resultados salvos em test_results.csv")
  
