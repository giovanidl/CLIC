import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertTokenizer, BertConfig
import sys
from ECGFounder.net1d import Net1D
from Model.ECG_encoder.resnet1d import resnet18_1d

# Caminho absoluto (baseado na localizacao deste arquivo, nao no cwd) --
# um path relativo aqui faz o HuggingFace recriar o cache do zero sempre que
# um script for executado de um cwd diferente da raiz do projeto (ja
# aconteceu: gerou um Model/BERT_pretrain/ duplicado de 832MB dentro de
# runs/resnet18/).
bert_pretrain_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "BERT_pretrain")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


def freeze_batchnorm_running_stats(module):
    """Mantém em .eval() qualquer BatchNorm cujos parâmetros estão
    congelados (requires_grad=False).

    model.train() do PyTorch marca .training=True em TODOS os submódulos
    recursivamente, independente de requires_grad. Sem isso, o BatchNorm de
    camadas "congeladas" (ex: estágios do ECGFounder que não estão sendo
    finetunados) passa a usar a estatística do minibatch atual em vez da
    running stat pré-treinada, e ainda atualiza essa running stat a cada
    passo — corrompendo a calibração do backbone congelado mesmo que seus
    pesos nunca sejam atualizados pelo otimizador. Chamar isso depois de
    todo model.train(mode=True) evita esse problema.
    """
    for m in module.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            if not any(p.requires_grad for p in m.parameters(recurse=False)):
                m.eval()

class LanguageModel(nn.Module):
    def __init__(self, freeze=False):
        super(LanguageModel, self).__init__()
        self.language_model = BertModel.from_pretrained('emilyalsentzer/Bio_ClinicalBERT', cache_dir=bert_pretrain_path)
        
        if freeze:
            for p in self.language_model.parameters():
                p.requires_grad = False
        else:
            # congela tudo
            
            for p in self.language_model.parameters():
                p.requires_grad = False

            # libera apenas as duas últimas camadas
            for p in self.language_model.encoder.layer[-2:].parameters():
                p.requires_grad = True

            # libera também o pooler (opcional)
            for p in self.language_model.pooler.parameters():
                p.requires_grad = True

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

        self.ecg_encoder = resnet18_1d(
            in_channels=12,
            projection_size=self.embedding_dim
        )

        mlp_hidden2 = mlp_hidden // 2  # segunda camada oculta menor
        mlp_hidden3 = mlp_hidden2 // 4
        #512 -> 256 -> 64 -> 5
        self.classifier = nn.Sequential(
            nn.Linear(self.embedding_dim, mlp_hidden),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(mlp_hidden, mlp_hidden2), # 512->256
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(mlp_hidden2, mlp_hidden3),# 256->64
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(mlp_hidden3, 5)     # 5 classes
        )

    def forward(self, ecg_data, text_emb):
        features = self.ecg_encoder(ecg_data)
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

        # ----- TEXT ENCODER (texto já vem pré-computado no forward, então só
        # precisamos do hidden_size do ClinicalBERT, não dos pesos completos) -----
        self.text_embedding_dim = BertConfig.from_pretrained('emilyalsentzer/Bio_ClinicalBERT', cache_dir=bert_pretrain_path).hidden_size  # geralmente 768

        # ----- FUSÃO (CONCAT) -----
        fusion_dim = self.ecg_embedding_dim + self.text_embedding_dim
        new_dim = self.text_embedding_dim
        mlp_hidden2 = mlp_hidden // 2  # segunda camada oculta menor
        mlp_hidden3 = mlp_hidden2 // 4

        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, mlp_hidden),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(mlp_hidden, mlp_hidden2), # 512->256
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(mlp_hidden2, mlp_hidden3),# 256->64
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(mlp_hidden3, 5)     # 5 classes
        )

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
        self.ecg_encoder = resnet18_1d(
            in_channels=12,
            projection_size=self.embedding_dim
        )
        # texto já vem pré-computado no forward, então só precisamos do
        # hidden_size do ClinicalBERT, não dos pesos completos
        self.text_embedding_dim = BertConfig.from_pretrained('emilyalsentzer/Bio_ClinicalBERT', cache_dir=bert_pretrain_path).hidden_size  # geralmente 768
        
        
        fusion_dim = self.embedding_dim + self.text_embedding_dim
        new_dim = self.text_embedding_dim
        mlp_hidden2 = mlp_hidden // 2  # segunda camada oculta menor
        mlp_hidden3 = mlp_hidden2 // 4

        # ----- CLASSIFICADOR (MLP) -----  fusion_dim -> 512 -> 256 -> 64 -> 5
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, mlp_hidden),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(mlp_hidden, mlp_hidden2),# 512 -> 256
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(mlp_hidden2, mlp_hidden3),# 256 -> 64
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(mlp_hidden3, 5)     # 5 classes
        )

    def forward(self, ecg, text_emb):
        ecg_feat = self.ecg_encoder(ecg)
        text_feat = text_emb
        fused = torch.cat([ecg_feat, text_feat], dim=1)
        return self.classifier(fused)


class MODEL_RobustText_Finetuning(nn.Module):
    def __init__(self, embedding_dim=None, mlp_hidden=256, num_classes=5, stage="train", unfreeze_ecg_layers=2):

        super(MODEL_RobustText_Finetuning, self).__init__()

        self.stage = stage
        self.ecg_embedding_dim = embedding_dim

        # ----- ECG ENCODER -----
        self.ecg_encoder = resnet18_1d(
            in_channels=12,
            projection_size=self.ecg_embedding_dim
        )

        # ----- TEXT ENCODER (FROZEN) -----
        self.text_encoder = LanguageModel(freeze=False)
        self.text_embedding_dim = self.text_encoder.language_model.config.hidden_size  # geralmente 768

        # ----- FUSÃO (CONCAT) -----
        fusion_dim = self.ecg_embedding_dim + self.text_embedding_dim
        mlp_hidden2 = mlp_hidden // 2  # segunda camada oculta menor
        mlp_hidden3 = mlp_hidden2 // 4



        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, mlp_hidden),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(mlp_hidden, mlp_hidden2), # 512->256
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(mlp_hidden2, mlp_hidden3),# 256->64
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(mlp_hidden3, 5)     # 5 classes
        )

        self.setup_ecg_grad_status(unfreeze_ecg_layers)

    def setup_ecg_grad_status(self, unfreeze_ecg_layers):
        """
        Congela todo o ecg_encoder e descongela só os N últimos estágios
        (layer1..layer4) + a projeção final (fc). unfreeze_ecg_layers=0
        deixa o ecg_encoder inteiro congelado (só inferência).
        """
        for param in self.ecg_encoder.parameters():
            param.requires_grad = False

        if unfreeze_ecg_layers > 0:
            stages = [
                self.ecg_encoder.layer1,
                self.ecg_encoder.layer2,
                self.ecg_encoder.layer3,
                self.ecg_encoder.layer4,
            ]
            for stage in stages[-unfreeze_ecg_layers:]:
                for param in stage.parameters():
                    param.requires_grad = True
            for param in self.ecg_encoder.fc.parameters():
                param.requires_grad = True

    def train(self, mode=True):
        super().train(mode)
        if mode:
            freeze_batchnorm_running_stats(self.ecg_encoder)
        return self

    def forward(self, ecg, input_ids, attention_mask):
        ecg_feat = self.ecg_encoder(ecg)
        text_feat = self.text_encoder(input_ids, attention_mask)

        fused = torch.cat([ecg_feat, text_feat], dim=1)
        return self.classifier(fused)


class MODEL_ECGLLM_Finetuning(nn.Module):
    def __init__(self, embedding_dim=None, mlp_hidden=256, num_classes=5, stage="train", unfreeze_ecg_layers=2):

        super(MODEL_ECGLLM_Finetuning, self).__init__()

        self.stage = stage
        self.ecg_embedding_dim = embedding_dim

        # ----- ECG ENCODER -----
        self.ecg_encoder = resnet18_1d(
            in_channels=12,
            projection_size=self.ecg_embedding_dim
        )

        # ----- TEXT ENCODER (FROZEN) -----
        self.text_encoder = LanguageModel(freeze=False)
        self.text_embedding_dim = self.text_encoder.language_model.config.hidden_size  # geralmente 768

        # ----- FUSÃO (CONCAT) -----
        fusion_dim = self.ecg_embedding_dim + self.text_embedding_dim
        mlp_hidden2 = mlp_hidden // 2  # segunda camada oculta menor
        mlp_hidden3 = mlp_hidden2 // 4



        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, mlp_hidden),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(mlp_hidden, mlp_hidden2), # 512->256
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(mlp_hidden2, mlp_hidden3),# 256->64
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(mlp_hidden3, 5)     # 5 classes
        )

        self.setup_ecg_grad_status(unfreeze_ecg_layers)

    def setup_ecg_grad_status(self, unfreeze_ecg_layers):
        """Ver MODEL_RobustText_Finetuning.setup_ecg_grad_status."""
        for param in self.ecg_encoder.parameters():
            param.requires_grad = False

        if unfreeze_ecg_layers > 0:
            stages = [
                self.ecg_encoder.layer1,
                self.ecg_encoder.layer2,
                self.ecg_encoder.layer3,
                self.ecg_encoder.layer4,
            ]
            for stage in stages[-unfreeze_ecg_layers:]:
                for param in stage.parameters():
                    param.requires_grad = True
            for param in self.ecg_encoder.fc.parameters():
                param.requires_grad = True

    def train(self, mode=True):
        super().train(mode)
        if mode:
            freeze_batchnorm_running_stats(self.ecg_encoder)
        return self

    def forward(self, ecg, input_ids, attention_mask):
        ecg_feat = self.ecg_encoder(ecg)
        text_feat = self.text_encoder(input_ids, attention_mask)

        fused = torch.cat([ecg_feat, text_feat], dim=1)
        return self.classifier(fused)


class MODEL_FM_RobustText_Finetuning(nn.Module):
    def __init__(self, embedding_dim=1024, mlp_hidden=256, num_classes=5, stage="train"):
        super(MODEL_FM_RobustText_Finetuning, self).__init__()

        self.stage = stage
        self.ecg_embedding_dim = embedding_dim  # dimensão do embedding pré-computado (ECGFounder = 1024)
        self.ecg_encoder, self.ecg_embedding_dim = self._load_ecgfounder_backbone(
            device=device, 
            pth="/home/giovanidl/doutorado/prelim/ECGFounder/12_lead_ECGFounder.pth", 
            freeze=True
        )

        # ----- TEXT ENCODER -----
        self.text_encoder = LanguageModel(freeze=False)
        self.text_embedding_dim = self.text_encoder.language_model.config.hidden_size  # geralmente 768

        # ----- FUSÃO (CONCAT) -----
        fusion_dim = self.ecg_embedding_dim + self.text_embedding_dim
        mlp_hidden2 = mlp_hidden // 2  # segunda camada oculta menor
        mlp_hidden3 = mlp_hidden2 // 4
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, mlp_hidden),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(mlp_hidden, mlp_hidden2),  # 512->256
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(mlp_hidden2, mlp_hidden3),  # 256->64
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(mlp_hidden3, num_classes)  # num_classes (default 5)
        )
        self.setup_grad_status(unfreeze_ecg_layers=1, unfreeze_text_layers=2)

    @staticmethod
    def _load_ecgfounder_backbone(
        device="cuda", 
        pth="./checkpoint/12_lead_ECGFounder.pth", 
        freeze=True
    ):
        # Hiperparâmetros oficiais da arquitetura Net1D do ECGFounder
        model = Net1D(
            in_channels=12,
            base_filters=64,
            ratio=1,
            filter_list=[64, 160, 160, 400, 400, 1024, 1024],
            m_blocks_list=[2, 2, 2, 3, 3, 4, 4],
            kernel_size=16,
            stride=2,
            groups_width=16,
            n_classes=150,  # Pré-treino original em 150 classes
            use_bn=True,
            use_do=True,
            return_features=True  # Garante que retorne (out, deep_features)
        )

        # Carregar pesos do Checkpoint
        checkpoint = torch.load(pth, map_location="cpu",weights_only=False)
        state_dict = checkpoint.get("state_dict", checkpoint.get("model", checkpoint))

        # Tratar prefixo 'module.' se salvo via DataParallel
        new_state_dict = {}
        for k, v in state_dict.items():
            name = k.replace("module.", "")
            new_state_dict[name] = v

        model.load_state_dict(new_state_dict, strict=True)
        print(f"[ECGFounder Net1D] Pesos carregados com sucesso de: {pth}")

        # Atributo utilitário para saber o tamanho da dimensão latente
        model.embedding_dim = model.filter_list[-1] # 1024

        if freeze:
            for param in model.parameters():
                param.requires_grad = False
            model.eval()

        return model.to(device), model.embedding_dim

    def setup_grad_status(self, unfreeze_ecg_layers=1, unfreeze_text_layers=2):
        """
        Congela todo o modelo e descongela apenas as N últimas etapas/estágios.
        """
        # A) Congelar tudo
        for param in self.parameters():
            param.requires_grad = False

        # B) Descongelar as N últimas camadas do ClinicalBERT
        if unfreeze_text_layers > 0:
            for layer in self.text_encoder.language_model.encoder.layer[-unfreeze_text_layers:]:
                for param in layer.parameters():
                    param.requires_grad = True

        # C) Descongelar os N últimos estágios do Net1D (self.stage_list)
        if unfreeze_ecg_layers > 0 and hasattr(self.ecg_encoder, 'stage_list'):
            for stage in self.ecg_encoder.stage_list[-unfreeze_ecg_layers:]:
                for param in stage.parameters():
                    param.requires_grad = True

        # D) A cabeça de classificação sempre treina
        for param in self.classifier.parameters():
            param.requires_grad = True

    def train(self, mode=True):
        super().train(mode)
        if mode:
            freeze_batchnorm_running_stats(self.ecg_encoder)
        return self

    def forward(self, ecg, input_ids, attention_mask):
        # 1. Forward do ECG (Net1D)
        ecg_out = self.ecg_encoder(ecg)
        
        # Como return_features=True no Net1D, ele devolve a tupla (out, deep_features)
        if isinstance(ecg_out, tuple):
            _, ecg_features = ecg_out
        else:
            ecg_features = ecg_out # Backup caso retorne direto
        
        # 2. Forward do Texto (ClinicalBERT - Token CLS [0])
        text_features = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)

        # 3. Concatenar vetores latentes (Shape: [batch, 1024 + 768] = [batch, 1792])
        fused = torch.cat((ecg_features, text_features), dim=-1)
        
        # 4. Predição final
        logits = self.classifier(fused)
        return logits
    
    
class MODEL_FM_LLM_Finetuning(nn.Module):
    def __init__(self, embedding_dim=1024, mlp_hidden=256, num_classes=5, stage="train"):
        super(MODEL_FM_LLM_Finetuning, self).__init__()

        self.stage = stage
        self.ecg_embedding_dim = embedding_dim  # dimensão do embedding pré-computado (ECGFounder = 1024)

        # ----- TEXT ENCODER -----
        self.text_encoder = LanguageModel(freeze=False)
        self.text_embedding_dim = self.text_encoder.language_model.config.hidden_size  # geralmente 768

        # ----- FUSÃO (CONCAT) -----
        fusion_dim = self.ecg_embedding_dim + self.text_embedding_dim
        mlp_hidden2 = mlp_hidden // 2  # segunda camada oculta menor
        mlp_hidden3 = mlp_hidden2 // 4
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, mlp_hidden),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(mlp_hidden, mlp_hidden2),  # 512->256
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(mlp_hidden2, mlp_hidden3),  # 256->64
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(mlp_hidden3, num_classes)  # num_classes (default 5)
        )

    def forward(self, ecg_emb, input_ids, attention_mask):
        text_feat = self.text_encoder(input_ids, attention_mask)
        fused = torch.cat([ecg_emb, text_feat], dim=1)
        return self.classifier(fused)

class MODEL_FM(nn.Module):
    def __init__(self, embedding_dim=None, mlp_hidden=256, num_classes=5):
             
        super(MODEL_FM, self).__init__()
        
        self.embedding_dim = embedding_dim

        mlp_hidden2 = mlp_hidden // 2  # segunda camada oculta menor
        mlp_hidden3 = mlp_hidden2 // 4

        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, mlp_hidden),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(mlp_hidden, mlp_hidden2),  # 512->256
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(mlp_hidden2, mlp_hidden3),  # 256->64
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(mlp_hidden3, num_classes)  # num_classes (default 5)
        )
    
        # )

    def forward(self, ecg_emb, text_emb=None):
        return self.classifier(ecg_emb)

class MODEL_FM_ATTR(nn.Module):
    def __init__(self, embedding_dim=None, mlp_hidden=256, num_classes=5):
        
        
        super(MODEL_FM_ATTR, self).__init__()
        
        
        
        self.embedding_dim = embedding_dim

        mlp_hidden2 = mlp_hidden // 2  # segunda camada oculta menor
        mlp_hidden3 = mlp_hidden2 // 4

        self.classifier = nn.Sequential(
            nn.Linear( self.embedding_dim+7, mlp_hidden),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(mlp_hidden, mlp_hidden2),  # 512->256
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(mlp_hidden2, mlp_hidden3),  # 256->64
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(mlp_hidden3, num_classes)  # num_classes (default 5)
        )

    def forward(self, ecg_emb, text_emb):
        fused = torch.cat([ecg_emb, text_emb], dim=1)
        return self.classifier(fused)


class MODEL_ResNet18_ATTR(nn.Module):
    """ResNet18 treinada do zero (sinal bruto) + atributos numéricos (ATTR,
    7-dim) concatenados. Equivalente à MODEL_FM_ATTR, mas com um ecg_encoder
    treinável em vez de um embedding pré-computado (ECGFounder) como entrada.
    """

    def __init__(self, embedding_dim=512, mlp_hidden=512, num_classes=5, attr_dim=7):
        super(MODEL_ResNet18_ATTR, self).__init__()

        self.embedding_dim = embedding_dim

        self.ecg_encoder = resnet18_1d(
            in_channels=12,
            projection_size=self.embedding_dim
        )

        fusion_dim = self.embedding_dim + attr_dim
        mlp_hidden2 = mlp_hidden // 2  # segunda camada oculta menor
        mlp_hidden3 = mlp_hidden2 // 4

        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, mlp_hidden),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(mlp_hidden, mlp_hidden2),  # 512->256
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(mlp_hidden2, mlp_hidden3),  # 256->64
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(mlp_hidden3, num_classes)  # num_classes (default 5)
        )

    def forward(self, ecg, attr):
        ecg_feat = self.ecg_encoder(ecg)
        fused = torch.cat([ecg_feat, attr], dim=1)
        return self.classifier(fused)


class MODEL_RobustText_FM(nn.Module):
    def __init__(self, embedding_dim=None, mlp_hidden=256, num_classes=5, stage="train"):
        
        super(MODEL_RobustText_FM, self).__init__()

        self.stage = stage
        self.ecg_embedding_dim = embedding_dim

        # ----- TEXT ENCODER (texto já vem pré-computado no forward, então só
        # precisamos do hidden_size do ClinicalBERT, não dos pesos completos) -----
        self.text_embedding_dim = BertConfig.from_pretrained('emilyalsentzer/Bio_ClinicalBERT', cache_dir=bert_pretrain_path).hidden_size  # geralmente 768

        # ----- FUSÃO (CONCAT) -----
        fusion_dim = self.ecg_embedding_dim + self.text_embedding_dim
        #new_dim = self.text_embedding_dim
        mlp_hidden2 = mlp_hidden // 2  # segunda camada oculta menor
        mlp_hidden3 = mlp_hidden2 // 4



        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, mlp_hidden),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(mlp_hidden, mlp_hidden2), # 512->256
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(mlp_hidden2, mlp_hidden3),# 256->64
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(mlp_hidden3, 5)     # 5 classes
        )




        self.tokenizer = BertTokenizer.from_pretrained('emilyalsentzer/Bio_ClinicalBERT', cache_dir=bert_pretrain_path)
        self.class_text_representation = None

    def ssl_process_text(self, text_data):
        prompt_list = list(text_data)
        tokens = self.tokenizer(prompt_list, padding=True, truncation=True, return_tensors='pt', max_length=100)
        return tokens


    def forward(self, ecg_emb, text_emb):

        text_feat = text_emb
        fused = torch.cat([ecg_emb, text_feat], dim=1)
        return self.classifier(fused)


class TextOnlyClassifier(nn.Module):
    """Ablação: classifica só a partir do embedding de texto pré-computado
    (ClinicalBERT congelado), sem nenhuma informação de ECG. `forward`
    recebe `ecg_emb` pra manter a mesma assinatura de
    `engine.forward_adapters.precomputed_forward`/`PTBXLDataset`
    (model(ecg_emb, text_emb)) e simplesmente ignora esse argumento — o
    dataset/ecg_source_fn usado na config pode ser qualquer um pré-computado
    já existente (o valor nunca chega ao classificador).
    """
    def __init__(self, mlp_hidden=256, num_classes=5, stage="train"):
        super(TextOnlyClassifier, self).__init__()

        self.stage = stage
        self.text_embedding_dim = BertConfig.from_pretrained('emilyalsentzer/Bio_ClinicalBERT', cache_dir=bert_pretrain_path).hidden_size  # geralmente 768

        mlp_hidden2 = mlp_hidden // 2  # segunda camada oculta menor
        mlp_hidden3 = mlp_hidden2 // 4

        self.classifier = nn.Sequential(
            nn.Linear(self.text_embedding_dim, mlp_hidden),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(mlp_hidden, mlp_hidden2),  # 512->256
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(mlp_hidden2, mlp_hidden3),  # 256->64
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(mlp_hidden3, num_classes)  # num_classes (default 5)
        )

    def forward(self, ecg_emb, text_emb):
        return self.classifier(text_emb)

