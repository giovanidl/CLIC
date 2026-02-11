import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import sys
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report
import torchmetrics
from torchmetrics.classification import MulticlassConfusionMatrix

def test_model(model, test_loader, device, class_names=None):
    print("="*30)
    print("INICIANDO AVALIAÇÃO NO TEST SET")
    print("="*30)
    
    model.eval() # Importante: Desliga Dropout e Batch Norm changes
    
    # Listas para armazenar todas as predições e targets
    all_preds = []
    all_targets = []
    
    loop = tqdm(test_loader, file=sys.stdout, desc="Testing")
    
    with torch.no_grad(): # Economiza memória e cálculo
        for batch in loop:
            features, text_emb, targets = batch
            
            features = features.to(device)
            targets = targets.to(device)
            text_emb = text_emb.to(device)
            
            # Forward
            logits = model(features, text_emb)  # [batch_size, num_classes]
            
            # Conversão: Logits -> Probabilidades -> Classe Predita (Índice)
            # Se quiser as probabilidades, use torch.sigmoid(logits)
            preds_indices = torch.argmax(logits, dim=1)
            
            # Conversão: Target One-Hot -> Índice Real
            targets_indices = torch.argmax(targets, dim=1)
            
            # Armazena na CPU para cálculo final com Scikit-Learn
            all_preds.extend(preds_indices.cpu().numpy())
            all_targets.extend(targets_indices.cpu().numpy())

    # ==========================================
    # 3. GERAÇÃO DE RELATÓRIOS
    # ==========================================
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    
    # 1. Matriz de Confusão (Visualização textual)
    cm_metric = MulticlassConfusionMatrix(num_classes=5).to(device)
    # Precisamos converter de volta pra tensor para usar torchmetrics ou usar sklearn direto
    cm_tensor = cm_metric(torch.tensor(all_preds).to(device), torch.tensor(all_targets).to(device))
    
    print("\n" + "-"*20)
    print("MATRIZ DE CONFUSÃO FINAL:")
    print("-"*20)
    print(cm_tensor.cpu().numpy().astype(int))
    
    # 2. Relatório de Classificação (Precision, Recall, F1 por classe)
    print("\n" + "-"*20)
    print("RELATÓRIO DETALHADO POR CLASSE:")
    print("-"*20)
    
    # Nomes das classes padrão do PTB-XL (ajuste se a ordem for diferente)
    if class_names is None:
        class_names = ['NORM', 'MI', 'STTC', 'CD', 'HYP'] # Exemplo, verifique sua ordem correta!
        
    report = classification_report(all_targets, all_preds, target_names=class_names, digits=4)
    print(report)
    
    return all_preds, all_targets