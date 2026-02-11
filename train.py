import logging
import os.path
import sys
from copy import deepcopy
import torchmetrics
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import datetime
from torchmetrics.classification import MulticlassConfusionMatrix

from utils.utils import keep_top_files, epoch_log


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
date = datetime.datetime.now().strftime(format="%Y%m%d_%H%M%S")

class StepRunner:
    def __init__(self, model, loss_fn, stage="train", metrics_dict=None, optimizer=None):
        self.net = model
        self.loss_fn = loss_fn
        self.metrics_dict = metrics_dict
        self.stage = stage
        self.optimizer = optimizer
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Garanta que o device está definido

    def step(self, inputs, text_embedding,targets):
        inputs = inputs.to(self.device)
        targets = targets.to(self.device)
        text_embedding = text_embedding.to(self.device)
        logits = self.net(inputs, text_embedding) # [batch_size, num_classes]
        # Para BCEWithLogitsLoss, targets devem ser Float
        loss = self.loss_fn(logits, targets.float())

        # Backward Pass
        if self.optimizer is not None and self.stage == "train":
            self.optimizer.zero_grad() # Zera gradientes ANTES do backward
            loss.backward()
            self.optimizer.step()

        # Retorno da Loss
        return loss.item(), logits, targets
    
    def train_step(self, ecg_data, text_embedding, targets):
        self.net.train()
        return self.step(ecg_data, text_embedding, targets)

    @torch.no_grad()
    def eval_step(self, ecg_data, text_embedding, targets):
        self.net.eval()
        return self.step(ecg_data, text_embedding, targets)

    def __call__(self, ecg_data, text_embedding, targets):
        if self.stage == "train":
            return self.train_step(ecg_data, text_embedding, targets)
        else:
            return self.eval_step(ecg_data, text_embedding, targets)

 
class EpochRunner:
    def __init__(self, steprunner):
        self.steprunner = steprunner
        self.stage = steprunner.stage
        self.device = steprunner.device
        
        # Configuração das métricas (Multiclasse com 5 classes)
        # 'average=macro' calcula a métrica para cada classe e tira a média (bom para classes desbalanceadas)
        metrics_cfg = {'task': 'multiclass', 'num_classes': 5, 'average': 'macro'}
        
        self.acc_metric = torchmetrics.Accuracy(**metrics_cfg).to(self.device)
        self.prec_metric = torchmetrics.Precision(**metrics_cfg).to(self.device)
        self.rec_metric = torchmetrics.Recall(**metrics_cfg).to(self.device)
        self.f1_metric = torchmetrics.F1Score(**metrics_cfg).to(self.device)

    def __call__(self, dataloader):      
            
        
        total_loss, step = 0, 0
        #conf_mat = MulticlassConfusionMatrix(num_classes=5).to(device)

        # Reseta as métricas no início da época
        self.acc_metric.reset()
        self.prec_metric.reset()
        self.rec_metric.reset()
        self.f1_metric.reset()
        # Iterar diretamente no dataloader, não criar lista sampled_batches
        loop = tqdm(dataloader, file=sys.stdout) 
        
        for batch in loop:
            # O batch já vem como (inputs, targets) do DataLoader
            # Desempacota dentro do steprunner ou aqui:
            
            
            features, text_embedding, targets = batch
            
            loss, logits, targets = self.steprunner(features, text_embedding,targets)
            target_indices = torch.argmax(targets.to(self.device), dim=1)
            
            #conf_mat.update(logits, target_indices)
            
            
            # Atualiza métricas acumuladas
            self.acc_metric.update(logits, target_indices)
            self.prec_metric.update(logits, target_indices)
            self.rec_metric.update(logits, target_indices)
            self.f1_metric.update(logits, target_indices)
            
            step_log = dict({self.stage + "_loss": loss})
            total_loss += loss
            step += 1
            
            loop.set_postfix(**step_log)

        
        # print(conf_mat.compute())
        epoch_loss = total_loss / step
        epoch_acc = self.acc_metric.compute().item()
        epoch_prec = self.prec_metric.compute().item()
        epoch_rec = self.rec_metric.compute().item()
        epoch_f1 = self.f1_metric.compute().item()
        
        epoch_log = {
            f"{self.stage}_loss": epoch_loss,
            f"{self.stage}_acc": epoch_acc,
            f"{self.stage}_prec": epoch_prec,
            f"{self.stage}_rec": epoch_rec,
            f"{self.stage}_f1": epoch_f1
        }
        
        return epoch_log


def train_loop(model, optimizer, loss_fn, metrics_dict,
              train_dataloader, val_dataloader=None,
              epochs=10, save_path='./checkpoint/',
              patience=5, monitor="val_loss", mode="min"):
    
    train_history = {}
    
    
    print("=" * 25 + "Start Training" + "=" * 25)
    
    best_score = float('inf') if mode == "min" else float('-inf')
    epochs_no_improve = 0
    
        
    for epoch in range(1, epochs + 1):

        epoch_log("Epoch {0} / {1}".format(epoch, epochs))
        
        

        # 1 train -------------------------------------------------
        train_step_runner = StepRunner(model=model, stage="train",
                                       loss_fn=loss_fn, metrics_dict=deepcopy(metrics_dict),
                                       optimizer=optimizer)
        train_epoch_runner = EpochRunner(train_step_runner)
        train_metrics = train_epoch_runner(train_dataloader)

        for name, metric in train_metrics.items():
            train_history[name] = train_history.get(name, []) + [metric]

        # 2 validate -------------------------------------------------
        if val_dataloader:
            val_step_runner = StepRunner(model=model, stage="val", loss_fn=loss_fn)
            val_epoch_runner = EpochRunner(val_step_runner)
            with torch.no_grad():
                val_metrics = val_epoch_runner(val_dataloader)
            
            # Salva histórico de validação
            for name, metric in val_metrics.items():
                train_history[name] = train_history.get(name, []) + [metric]
            
            print(train_history)
            # PRINT BONITO DOS RESULTADOS
            print(f"TRAIN -> Loss: {train_metrics['train_loss']:.4f} | Acc: {train_metrics['train_acc']:.4f} | F1: {train_metrics['train_f1']:.4f}")
            print(f"VAL   -> Loss: {val_metrics['val_loss']:.4f}   | Acc: {val_metrics['val_acc']:.4f}   | F1: {val_metrics['val_f1']:.4f}")

            # 3. CHECKPOINT E EARLY STOPPING
            current_score = val_metrics[monitor]
            
            # Lógica para min (loss) ou max (f1/acc)
            is_better = (current_score < best_score) if mode == "min" else (current_score > best_score)
            
            if is_better:
                best_score = current_score
                epochs_no_improve = 0
                
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                
                date = datetime.datetime.now().strftime(format="%Y%m%d_%H%M%S")
                
                ckpt_name = f"MLP512_256_Epoch_{epoch}_{date}.pt"
                ckpt_path = os.path.join(save_path, ckpt_name)
                
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'monitor_val': current_score,
                }, ckpt_path)
                
                print(f">>> Melhor {monitor} alcançado! Checkpoint salvo: {ckpt_name}")
            else:
                epochs_no_improve += 1
                print(f">>> Sem melhoria em {monitor} por {epochs_no_improve} épocas.")
                
            if epochs_no_improve >= patience:
                print(">>> Early Stopping ativado.")
                break


    return pd.DataFrame(train_history)
