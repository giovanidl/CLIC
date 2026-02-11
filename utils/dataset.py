# -*- coding = utf-8 -*-
# @File : dataset.py
# @Software : PyCharm
import torch
from torch.utils.data import Dataset
import os
import pandas as pd
import wfdb     
import sys
import numpy as np
from transformers import BertModel, BertTokenizer, AutoModelForCausalLM
from ollama import generate

bert_pretrain_path = "Model/BERT_pretrain/"

class PTBXLDataset(Dataset):
    """
    Dataset do PTB-XL para uso com PyTorch.
    Cada item contém um sinal de ECG (12 derivações) e seu respectivo rótulo diagnóstico.
    """
    def __init__(self, data_dir, split="train", sampling_rate=100, transform=None):
        """
        Args:
            data_dir (str): Caminho para a pasta raiz do PTB-XL.
            split (str): 'train' ou 'test' (ou 'validation', se quiser dividir você mesmo).
            sampling_rate (int): Pode ser 100 ou 500 Hz — o PTB-XL tem os dois.
            transform (callable, opcional): Transformação aplicada ao sinal de ECG.
        """
        super(PTBXLDataset, self).__init__()
        self.data_dir = data_dir
        self.sampling_rate = sampling_rate
        self.transform = transform
        self.categories = ["NORM", "MI", "STTC", "HYP", "CD"]
        self.tokenizer = BertTokenizer.from_pretrained('emilyalsentzer/Bio_ClinicalBERT', cache_dir=bert_pretrain_path)
      
        # Carrega o arquivo de metadados principa
        metadata_path = os.path.join(data_dir, "ptbxl_database.csv")
        self.metadata = pd.read_csv(metadata_path)
        # print("\n\metadata MAP\n\n", self.metadata)

        # Seleciona o split desejado
        # O PTB-XL tem uma coluna chamada 'strat_fold' usada para dividir em 10 folds
        if split == "train":
            self.metadata = self.metadata[self.metadata["strat_fold"] < 9]
        elif split == "val":
            self.metadata = self.metadata[self.metadata["strat_fold"] == 9]
        elif split == "test":
            self.metadata = self.metadata[self.metadata["strat_fold"] == 10]

        # Carrega o mapeamento de diagnósticos
        self.label_map_geral = pd.read_csv(os.path.join(data_dir, "scp_statements.csv"), index_col=0)
        self.label_map = self.label_map_geral[self.label_map_geral.diagnostic == 1]  # mantém só diagnósticos

        #print("\n\nTESTE\n\n", self.label_map.loc["LAFB"])
        
        # Cria um dicionário {diagnóstico: índice}diagnostic
        # self.class_names = list(self.label_map.index)
        # print(self.class_names)
        # self.label_dict = {k: i for i, k in enumerate(self.class_names)}
        # # Prepara caminhos dos arquivos de sinais
        if sampling_rate == 100:
            self.metadata["file_path"] = self.metadata["filename_lr"]
        else:
            self.metadata["file_path"] = self.metadata["filename_hr"]

        self.records = self.metadata.reset_index(drop=True)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record = self.records.iloc[idx]
        metadata_robust_text = self.get_metadata_robust_text(record)
        encoding = self.tokenizer(
            metadata_robust_text,
            padding='max_length',
            truncation=True,
            max_length=25,
            return_tensors="pt"
        )
        input_ids = encoding["input_ids"].squeeze(0)       # [seq_len]
        attention_mask = encoding["attention_mask"].squeeze(0)
       
        file_path = os.path.join(self.data_dir, record["file_path"])
        # Lê o sinal de ECG com o wfdb
        signal, _ = wfdb.rdsamp(file_path)
        
        signal = torch.tensor(signal.T, dtype=torch.float32)  # shape (12, length)
        # Extrai rótulos diagnósticos (multi-label)
        labels = self._extract_labels(record["scp_codes"])
        #print(labels)
        superclass = "NORM"  # padrão
        if labels[0] in self.label_map.index:
            superclass = self.label_map.loc[labels[0]].diagnostic_class
        

        idx = self.categories.index(superclass)
        one_hot = torch.zeros(len(self.categories))      # cria vetor cheio de zeros
        one_hot[idx] = 1  
        # label_tensor = torch.zeros(len(self.class_names))
        
                
        if self.transform:
            signal = self.transform(signal)
            
        # print("ECG:", signal.shape)
        # print("input_ids:", input_ids.shape)
        # print("att_mask:", attention_mask.shape)
            
        return signal, input_ids, attention_mask,  one_hot

    def _extract_labels(self, scp_codes_str):
        """
        Converte o campo de string de SCP codes (ex: '{"NORM": 100, "MI": 50}')
        em uma lista de diagnósticos.
        """
        scp_dict = eval(scp_codes_str)
        return list(scp_dict.keys())
    
    def get_metadata_robust_text(self, record):
        age = record['age']
        sex = record['sex'] 
        weight = record['weight']
        collection_device = record['device'].split(" ")[0].replace("-", "")
        height = record['height']

        # Sexo
        text_sex = "Male" if sex == 0 else "Female"

        # Idade
        if age >= 200:
            text_age = "The patient is over 90 years old"
        else:
            text_age = f"Pacient is {int(age)} years old"

        # Peso
        weight_text = (
            ", has unknown weight"
            if np.isnan(weight)
            else f", weight {int(weight)} kg"
        )
        bmi_text = ""
        if np.isnan(height):
            bmi_text = "and with unknown BMI (probably above 40)."
        elif not np.isnan(weight):
            bmi_text = f"and has a BMI of {int(weight / (height / 100) ** 2)}."  
            
        device_text = f"The device used was {collection_device}".strip() + "."

        return f"{text_age}, {text_sex}{weight_text} {bmi_text} {device_text}"
            
class PTBXLDataset_with_prompt(Dataset):
    """
    Dataset do PTB-XL para uso com PyTorch.
    Cada item contém um sinal de ECG (12 derivações) e seu respectivo rótulo diagnóstico.
    """
    def __init__(self, data_dir, split="train", sampling_rate=100, transform=None):
        """
        Args:
            data_dir (str): Caminho para a pasta raiz do PTB-XL.
            split (str): 'train' ou 'test' (ou 'validation', se quiser dividir você mesmo).
            sampling_rate (int): Pode ser 100 ou 500 Hz — o PTB-XL tem os dois.
            transform (callable, opcional): Transformação aplicada ao sinal de ECG.
        """
        super(PTBXLDataset_with_prompt, self).__init__()
        self.data_dir = data_dir
        self.sampling_rate = sampling_rate
        self.transform = transform
        self.categories = ["NORM", "MI", "STTC", "HYP", "CD"]
        self.tokenizer = BertTokenizer.from_pretrained('emilyalsentzer/Bio_ClinicalBERT', cache_dir=bert_pretrain_path)
        #self.llm = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3-8b-instruct", cache_dir=bert_pretrain_path).to(self.device)
        
        # Carrega o arquivo de metadados principa
        metadata_path = os.path.join(data_dir, "ptbxl_database.csv")
        self.metadata = pd.read_csv(metadata_path)

        # Seleciona o split desejado
        # O PTB-XL tem uma coluna chamada 'strat_fold' usada para dividir em 10 folds
        if split == "train":
            self.metadata = self.metadata[self.metadata["strat_fold"] < 9]
        elif split == "val":
            self.metadata = self.metadata[self.metadata["strat_fold"] == 9]
        elif split == "test":
            self.metadata = self.metadata[self.metadata["strat_fold"] == 10]

        # Carrega o mapeamento de diagnósticos
        self.label_map_geral = pd.read_csv(os.path.join(data_dir, "scp_statements.csv"), index_col=0)
        self.label_map = self.label_map_geral[self.label_map_geral.diagnostic == 1]  # mantém só diagnósticos

        if sampling_rate == 100:
            self.metadata["file_path"] = self.metadata["filename_lr"]
        else:
            self.metadata["file_path"] = self.metadata["filename_hr"]

        self.records = self.metadata.reset_index(drop=True)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record = self.records.iloc[idx]
        
        labels = self._extract_labels(record["scp_codes"])
        #print(labels)
      
        generated_text = self.generate_prompt_from_metadata(record)

        encoding = self.tokenizer(
            generated_text,
            padding='max_length',
            truncation=True,
            max_length=25,
            return_tensors="pt"
        )
        input_ids = encoding["input_ids"].squeeze(0)       # [seq_len]
        attention_mask = encoding["attention_mask"].squeeze(0)
       
        file_path = os.path.join(self.data_dir, record["file_path"])
        # Lê o sinal de ECG com o wfdb
        signal, _ = wfdb.rdsamp(file_path)
        
        signal = torch.tensor(signal.T, dtype=torch.float32)  # shape (12, length)
        # Extrai rótulos diagnósticos (multi-label)

        superclass = "NORM"  # padrão
        if labels[0] in self.label_map.index:
            superclass = self.label_map.loc[labels[0]].diagnostic_class
        

        idx = self.categories.index(superclass)
        one_hot = torch.zeros(len(self.categories))      # cria vetor cheio de zeros
        one_hot[idx] = 1  
        # label_tensor = torch.zeros(len(self.class_names))
        
                
        if self.transform:
            signal = self.transform(signal)
        
            
        return signal, input_ids, attention_mask,  one_hot

    def _extract_labels(self, scp_codes_str):
        """
        Converte o campo de string de SCP codes (ex: '{"NORM": 100, "MI": 50}')
        em uma lista de diagnósticos.
        """
        scp_dict = eval(scp_codes_str)
        return list(scp_dict.keys())
    
            
    def generate_prompt_from_metadata(self, record):
        age = record['age']
        sex = record['sex'] 
        weight = record['weight']
        collection_device = record['device'].split(" ")[0].replace("-", "")
        height = record['height']
        
        

        # Sexo
        text_sex = "Male" if sex == 0 else "Female"

        # Idade
        if age >= 200:
            text_age = "The patient is over 90 years old"
        else:
            text_age = f"Pacient is {int(age)} years old"

        # Peso
        weight_text = (
            ", has unknown weight"
            if np.isnan(weight)
            else f", weight {int(weight)} kg"
        )
        bmi_text = ""
        if np.isnan(height):
            bmi_text = "and with unknown BMI (probably above 40)."
        elif not np.isnan(weight):
            bmi_text = f"and has a BMI of {int(weight / (height / 100) ** 2)}."  
            
        device_text = f"The device used was {collection_device}".strip() + "."

        return f"{text_age}, {text_sex}{weight_text} {bmi_text} {device_text}"
    
class PTBXLDataset_with_generated_prompt(Dataset):
    """
    Dataset do PTB-XL para uso com PyTorch.
    Cada item contém um sinal de ECG (12 derivações) e seu respectivo rótulo diagnóstico.
    """
    def __init__(self, data_dir, split="train", sampling_rate=100, transform=None):
        """
        Args:
            data_dir (str): Caminho para a pasta raiz do PTB-XL.
            split (str): 'train' ou 'test' (ou 'validation', se quiser dividir você mesmo).
            sampling_rate (int): Pode ser 100 ou 500 Hz — o PTB-XL tem os dois.
            transform (callable, opcional): Transformação aplicada ao sinal de ECG.
        """
        super(PTBXLDataset_with_generated_prompt, self).__init__()
        self.data_dir = data_dir
        self.sampling_rate = sampling_rate
        self.transform = transform
        self.categories = ["NORM", "MI", "STTC", "HYP", "CD"]
        self.tokenizer = BertTokenizer.from_pretrained('emilyalsentzer/Bio_ClinicalBERT', cache_dir=bert_pretrain_path)
        #self.llm = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3-8b-instruct", cache_dir=bert_pretrain_path).to(self.device)
        
        # Carrega o arquivo de metadados principa
        metadata_path = os.path.join(data_dir, "ptbxl_database.csv")
        self.metadata = pd.read_csv(metadata_path)
        # print("\n\metadata MAP\n\n", self.metadata)

        # Seleciona o split desejado
        # O PTB-XL tem uma coluna chamada 'strat_fold' usada para dividir em 10 folds
        if split == "train":
            self.metadata = self.metadata[self.metadata["strat_fold"] < 9]
        elif split == "val":
            self.metadata = self.metadata[self.metadata["strat_fold"] == 9]
        elif split == "test":
            self.metadata = self.metadata[self.metadata["strat_fold"] == 10]

        # Carrega o mapeamento de diagnósticos
        self.label_map_geral = pd.read_csv(os.path.join(data_dir, "scp_statements.csv"), index_col=0)
        self.label_map = self.label_map_geral[self.label_map_geral.diagnostic == 1]  # mantém só diagnósticos

        if sampling_rate == 100:
            self.metadata["file_path"] = self.metadata["filename_lr"]
        else:
            self.metadata["file_path"] = self.metadata["filename_hr"]

        self.records = self.metadata.reset_index(drop=True)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record = self.records.iloc[idx]
        
        labels = self._extract_labels(record["scp_codes"])
        #print(labels)
        form_text= ""
        rhythm_text= ""
        for label in labels:
            if not np.isnan(self.label_map_geral.loc[label].form):
    
                form_text += self.label_map_geral.loc[label].description + ", "

        for label in labels:
            if not np.isnan(self.label_map_geral.loc[label].rhythm):
                rhythm_text += self.label_map_geral.loc[label].description + ", "


        generated_text = self.generate_prompt_from_metadata(record, form_text, rhythm_text)
        encoding = self.tokenizer(
            generated_text,
            padding='max_length',
            truncation=True,
            max_length=25,
            return_tensors="pt"
        )
        input_ids = encoding["input_ids"].squeeze(0)       # [seq_len]
        attention_mask = encoding["attention_mask"].squeeze(0)
       
        file_path = os.path.join(self.data_dir, record["file_path"])
        # Lê o sinal de ECG com o wfdb
        signal, _ = wfdb.rdsamp(file_path)
        
        signal = torch.tensor(signal.T, dtype=torch.float32)  # shape (12, length)
        # Extrai rótulos diagnósticos (multi-label)

        superclass = "NORM"  # padrão
        if labels[0] in self.label_map.index:
            superclass = self.label_map.loc[labels[0]].diagnostic_class
        

        idx = self.categories.index(superclass)
        one_hot = torch.zeros(len(self.categories))      # cria vetor cheio de zeros
        one_hot[idx] = 1  
        # label_tensor = torch.zeros(len(self.class_names))
        
                
        if self.transform:
            signal = self.transform(signal)
            
        # print("ECG:", signal.shape)
        # print("input_ids:", input_ids.shape)
        # print("att_mask:", attention_mask.shape)
            
        return signal, input_ids, attention_mask,  one_hot

    def _extract_labels(self, scp_codes_str):
        """
        Converte o campo de string de SCP codes (ex: '{"NORM": 100, "MI": 50}')
        em uma lista de diagnósticos.
        """
        scp_dict = eval(scp_codes_str)
        return list(scp_dict.keys())
    
    def generate_medical_text(self,
            age,
            sex,
            weight,
            bmi,
            collection_device,
            morphology_text,
            rhythm_text,
            num_predict=180
        ):
            prompt = f"""
        You are a cardiology specialist.

        Generate a concise, single-paragraph clinical ECG report based on the information below.
        Use formal medical English, objective tone, and clear clinical reasoning.

        Patient information:
        - Age: {age} years
        - Sex: {sex}
        - Weight: {weight} kg
        - Body Mass Index: {bmi}
        - Recording device: {collection_device}

        Electrocardiographic findings:
        - Signal morphology: {morphology_text}
        - Cardiac rhythm: {rhythm_text}

        End the report with a complete sentence and avoid bullet points or lists, and use all the information given above. 
        Don't calculate the BMI yourself, always use the given BMI, just use the height and weight information if available.
        Don't start the report with "Here is the clinical report" or similar phrases.
        Don't ever provide information, such as 70 bpm heart rate, that is not given in the input. Only assumptions that can be made using the given input.
        If the height is missing, it has a high chance of being above 40, according to the dataset paper, so it's safe to assume that the Body Mass Index of a patient with missing height data is above 40.
        Don't include the unit of the Body Mass Index in the report, just say "has a BMI of 32", for example.
        """

            response = generate(
                model="llama3.1:8b",
                prompt=prompt.strip(),
                options={
                    "num_predict": num_predict,
                    "temperature": 0.0,
                    "top_p": 0.9,
                    "repeat_penalty": 1.1
                }
            )["response"]

            return response.strip()
    
            
    def generate_prompt_from_metadata(self, record, morphology_text, rhythm_text):
        age = record['age']
        sex = record['sex'] 
        weight = record['weight']
        collection_device = record['device'].split(" ")[0].replace("-", "")
        height = record['height']
        
        

        # Sexo
        text_sex = "Male" if sex == 0 else "Female"

        # Idade
        if age >= 200:
            text_age = "The patient is over 90 years old"
        else:
            text_age = f"Pacient is {int(age)} years old"

        # Peso
        weight_text = (
            "Has unknown weight"
            if np.isnan(weight)
            else f"Weight {int(weight)} kg"
        )
        bmi_text = ""
        if np.isnan(height):
            bmi_text = "Unknown Body Mass Index (probably above 40)."
        elif not np.isnan(weight):
            bmi_text = f"Has a BMI of {int(weight / (height / 100) ** 2)}."  
            
        device_text = f"The device used was {collection_device}".strip() + "."
        
        
        response = self.generate_medical_text(
            text_age,
            text_sex,
            weight_text,
            bmi_text,
            device_text,
            morphology_text,
            rhythm_text
        )
        
        # print("Age:", text_age)
        # print("Sex:", text_sex)
        # print("Weight:", weight_text)
        # print("BMI:", bmi_text)
        # print("Collection device:", device_text)
        # print("Morphology text:", morphology_text)
        # print("Rhythm text:", rhythm_text)
        
        # print("\nGenerating medical text...\n")
        
        # print(response)
        
        # exit(0)
        return response
    
class PTBXLDatasetWithTextEmbeddingNPY(Dataset):
    """
    Dataset do PTB-XL com embeddings textuais pré-computados (.npy).
    """
    def __init__(
        self,
        data_dir,
        text_embedding_path,
        split="train",
        sampling_rate=100,
        transform=None
    ):
        super(PTBXLDatasetWithTextEmbeddingNPY,self).__init__()

        self.data_dir = data_dir
        self.sampling_rate = sampling_rate
        self.transform = transform

        self.categories = ["NORM", "MI", "STTC", "HYP", "CD"]

        # ---- carrega embeddings textuais ----
        self.text_embeddings = np.load(
            text_embedding_path,
            allow_pickle=True
        ).item()

        # ---- metadata PTB-XL ----
        metadata_path = os.path.join(data_dir, "ptbxl_database.csv")
        self.metadata = pd.read_csv(metadata_path)

        # splits oficiais
        if split == "train":
            self.metadata = self.metadata[self.metadata["strat_fold"] < 9]
        elif split == "val":
            self.metadata = self.metadata[self.metadata["strat_fold"] == 9]
        elif split == "test":
            self.metadata = self.metadata[self.metadata["strat_fold"] == 10]

        # diagnósticos
        self.label_map_geral = pd.read_csv(
            os.path.join(data_dir, "scp_statements.csv"),
            index_col=0
        )
        self.label_map = self.label_map_geral[
            self.label_map_geral.diagnostic == 1
        ]

        # arquivos ECG
        if sampling_rate == 100:
            self.metadata["file_path"] = self.metadata["filename_lr"]
        else:
            self.metadata["file_path"] = self.metadata["filename_hr"]

        self.records = self.metadata.reset_index(drop=True)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record = self.records.iloc[idx]

        # ---------- ECG ----------
        file_path = os.path.join(self.data_dir, record["file_path"])
        signal, _ = wfdb.rdsamp(file_path)
        signal = torch.tensor(signal.T, dtype=torch.float32)  # (12, L)

        if self.transform:
            signal = self.transform(signal)

        # ---------- TEXT EMBEDDING ----------
        # use o mesmo ID usado na geração offline
        ecg_fn = record["filename_hr"] if "filename_hr" in record else record["file_path"]
        
        # ---------- TEXT EMBEDDING ----------
        # use o mesmo ID usado na geração offline
        ecg_fn = record["filename_hr"] if "filename_hr" in record else record["file_path"]
        


        text_emb = torch.tensor(
            self.text_embeddings[ecg_fn],
            dtype=torch.float32
        )

        # ---------- LABEL ----------
        labels = self._extract_labels(record["scp_codes"])

        superclass = "NORM"
        if labels[0] in self.label_map.index:
            superclass = self.label_map.loc[labels[0]].diagnostic_class

        label_idx = self.categories.index(superclass)
        one_hot = torch.zeros(len(self.categories))
        one_hot[label_idx] = 1.0

        return signal, text_emb, one_hot

    def _extract_labels(self, scp_codes_str):
        scp_dict = eval(scp_codes_str)
        return list(scp_dict.keys())