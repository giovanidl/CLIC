import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertTokenizer

from Model.ECG_encoder.resnet1d import resnet18_1d

bert_pretrain_path = "Model/BERT_pretrain/"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

class FrozenLanguageModel(nn.Module):
    def __init__(self):
        super(FrozenLanguageModel, self).__init__()
        self.language_model = BertModel.from_pretrained('emilyalsentzer/Bio_ClinicalBERT', cache_dir=bert_pretrain_path)
        for param in self.language_model.parameters():
            param.requires_grad = False

    def forward(self, input_ids, attention_mask):
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        outputs = self.language_model(input_ids=input_ids, attention_mask=attention_mask)
        sentence_representation = outputs.last_hidden_state[:, 0, :]
        return sentence_representation

class MODEL(nn.Module):
    def __init__(self, embedding_dim=None, mlp_hidden=256, num_classes=5):
        super(MODEL, self).__init__()
        
        self.embedding_dim = embedding_dim
        
        # Encoder ECG (mantém igual)
        self.ecg_encoder = resnet18_1d(in_channels=12, projection_size=self.embedding_dim)
        
        # MLP Classifier: embedding_dim -> hidden -> num_classes
        # Nota: A última camada deve ter dimensão de saída igual ao número de classes
        # self.classifier = nn.Sequential(
        #     nn.Linear(self.embedding_dim, mlp_hidden),
        #     nn.ReLU(),
        #     nn.Dropout(0.5),
        #     nn.Linear(mlp_hidden, num_classes) 
        # )
        
        
        mlp_hidden2 = mlp_hidden // 4  
        #256 -> 64 -> 5
        self.classifier = nn.Sequential(
            nn.Linear(self.embedding_dim, mlp_hidden),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(mlp_hidden, mlp_hidden2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(mlp_hidden2, 5)     # 5 classes
        )
        
        
        # 256 -> 5
        # self.classifier = nn.Sequential(    
        #     nn.Linear(self.embedding_dim, mlp_hidden),
        #     nn.ReLU(),
        #     nn.Dropout(0.5),
        #     nn.Linear(mlp_hidden, 5)     # 5 classes
        # )
        
        # 64 -> 5
        # self.classifier = nn.Sequential(
        #     nn.Linear(self.embedding_dim, mlp_hidden2),
        #     nn.ReLU(),
        #     nn.Dropout(0.5),
        #     nn.Linear(mlp_hidden2, 5)     # 5 classes
        # )

    def forward(self, ecg_data, text_emb):
        # 1. Extração de características
        # ext_emb = torch.tensor(
        #     self.text_embeddings[ecg_fn],
        #     dtype=torch.float32
        # )
        features = self.ecg_encoder(ecg_data)
        
        # 2. Classificação
        logits = self.classifier(features)
        
        return logits

class MODEL_RobustText(nn.Module):
    def __init__(self, embedding_dim=None, mlp_hidden=256, num_classes=5, stage="train"):
        
        super(MODEL_RobustText, self).__init__()

        self.stage = stage
        self.ecg_embedding_dim = embedding_dim

        # ----- ECG ENCODER -----
        self.ecg_encoder = resnet18_1d(
            in_channels=12, 
            projection_size=self.ecg_embedding_dim
        )

        # ----- TEXT ENCODER (FROZEN) -----
        self.text_encoder = FrozenLanguageModel()
        self.text_embedding_dim = self.text_encoder.language_model.config.hidden_size  # geralmente 768

        # ----- FUSÃO (CONCAT) -----
        fusion_dim = self.ecg_embedding_dim + self.text_embedding_dim
        mlp_hidden2 = mlp_hidden // 2  # segunda camada oculta menor
        mlp_hidden3 = mlp_hidden2 // 4
        
        
        #----- CLASSIFICADOR (MLP) -----
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, mlp_hidden),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(mlp_hidden, mlp_hidden2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(mlp_hidden2, 5)     # 5 classes
        )
        
        # self.classifier = nn.Sequential(
        #     nn.Linear(fusion_dim, mlp_hidden),
        #     nn.ReLU(),
        #     nn.Dropout(0.5),
        #     nn.Linear(mlp_hidden, mlp_hidden2), # 512->256
        #     nn.ReLU(),
        #     nn.Dropout(0.5),
        #     nn.Linear(mlp_hidden2, mlp_hidden3),# 256->64
        #     nn.ReLU(),
        #     nn.Dropout(0.5),
        #     nn.Linear(mlp_hidden3, 5)     # 5 classes
        # )
        
        
        # self.classifier = nn.Sequential(
        #     nn.Linear(fusion_dim, mlp_hidden),
        #     nn.ReLU(),
        #     nn.Dropout(0.5),
        #     nn.Linear(mlp_hidden, 5)     # 5 classes
        # )
        # super(MODEL_RobustTexd, self).__init__()
        
        # self.embedding_dim = embedding_dim
        # self.text_encoder = FrozenLanguageModel()
        # # Encoder ECG (mantém igual)
        # self.ecg_encoder = resnet18_1d(in_channels=12, projection_size=self.embedding_dim)
        
        # # MLP Classifier: embedding_dim -> hidden -> num_classes
        # # Nota: A última camada deve ter dimensão de saída igual ao número de classes
        # self.classifier = nn.Sequential(
        #     nn.Linear(self.embedding_dim, mlp_hidden),
        #     nn.ReLU(),
        #     nn.Dropout(0.5),
        #     nn.Linear(mlp_hidden, num_classes) 
        # )
        
        self.text_embedding_dim = self.text_encoder.language_model.config.hidden_size
        self.tokenizer = BertTokenizer.from_pretrained('emilyalsentzer/Bio_ClinicalBERT', cache_dir=bert_pretrain_path)
        self.class_text_representation = None

    def ssl_process_text(self, text_data):
        prompt_list = list(text_data)
        tokens = self.tokenizer(prompt_list, padding=True, truncation=True, return_tensors='pt', max_length=100)
        return tokens


    def forward(self, ecg, text_emb):
        ecg_feat = self.ecg_encoder(ecg)
        text_feat = text_emb

        fused = torch.cat([ecg_feat, text_feat], dim=1)
        return self.classifier(fused)
   
class ECGTextFusion(nn.Module):
    def __init__(self, embedding_dim=512, mlp_hidden=256, stage="train"):
        super(ECGTextFusion, self).__init__()

        # ----- ECG ENCODER -----
        self.embedding_dim = embedding_dim
        self.text_encoder = FrozenLanguageModel()
        self.ecg_encoder = resnet18_1d(
            in_channels=12, 
            projection_size=self.embedding_dim
        )
        self.text_embedding_dim = self.text_encoder.language_model.config.hidden_size  # geralmente 768
        
        
        fusion_dim = self.embedding_dim + self.text_embedding_dim
        mlp_hidden2 = mlp_hidden // 2  # segunda camada oculta menor
        mlp_hidden3 = mlp_hidden2 // 4
        
        # # ----- CLASSIFICADOR (MLP) -----  512 -> 256 -> 64 -> 5
        # self.classifier = nn.Sequential(
        #     nn.Linear(fusion_dim, mlp_hidden), # fusion_dim -> 512
        #     nn.ReLU(),
        #     nn.Dropout(0.5),
        #     nn.Linear(mlp_hidden, mlp_hidden2),# 512 -> 256
        #     nn.ReLU(),
        #     nn.Dropout(0.5),
        #     nn.Linear(mlp_hidden2, mlp_hidden3),# 256 -> 64
        #     nn.ReLU(),
        #     nn.Dropout(0.5),
        #     nn.Linear(mlp_hidden3, 5)     # 5 classes
        # )
        
        #----- CLASSIFICADOR (MLP) ----- 512 -> 256 -> 5
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, mlp_hidden), # fusion_dim -> 512
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(mlp_hidden, mlp_hidden2),# 512 -> 256
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(mlp_hidden2, 5)     # 5 classes
        )
        
        
    def forward(self, ecg, text_emb):
        ecg_feat = self.ecg_encoder(ecg)
        text_feat = text_emb

        fused = torch.cat([ecg_feat, text_feat], dim=1)
        return self.classifier(fused)