# -*- coding = utf-8 -*-
# @File : dataset.py
# @Software : PyCharm
import os

import numpy as np
import pandas as pd
import torch
from ollama import generate
from torch.utils.data import Dataset

from utils.labels import CATEGORIES, build_split_records, parse_scp_codes


class PTBXLDataset(Dataset):
    """Dataset do PTB-XL, composto por uma ecg_source e uma text_source.

    ecg_source / text_source: objetos __call__(record) -> tensor|str, ver
    utils/ecg_sources.py e utils/text_sources.py. Substitui as antigas
    PTBXLDataset / PTBXLDatasetWithTextEmbeddingNPY / ...JSON / ...JSONandFM /
    ...TextandFMNPY, que só variavam nessas duas fontes.
    """

    def __init__(
        self,
        data_dir,
        ecg_source,
        text_source,
        split="train",
        sampling_rate=500,
        categories=CATEGORIES,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.ecg_source = ecg_source
        self.text_source = text_source
        self.categories = categories

        self.records = build_split_records(data_dir, split, sampling_rate, categories)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record = self.records.iloc[idx]

        ecg_repr = self.ecg_source(record)
        text_repr = self.text_source(record)
        label = torch.tensor(record["label"], dtype=torch.float32)

        return ecg_repr, text_repr, label


class PTBXLDataset_with_generated_prompt(Dataset):
    """
    Dataset auxiliar usado apenas por generate_text_cache.py para gerar
    (offline, via LLM) os caches de texto clínico. Não é usado para
    treino/avaliação, por isso mantém todos os registros do split (mesmo
    sem código diagnóstico) e preserva a lógica de construção de texto
    exatamente como estava.
    """

    def __init__(self, data_dir, split="train", sampling_rate=100, transform=None):
        super(PTBXLDataset_with_generated_prompt, self).__init__()
        self.data_dir = data_dir
        self.sampling_rate = sampling_rate
        self.transform = transform
        self.categories = CATEGORIES

        # Carrega o arquivo de metadados principal
        metadata_path = os.path.join(data_dir, "ptbxl_database.csv")
        self.metadata = pd.read_csv(metadata_path)

        # Seleciona o split desejado
        if split == "train":
            self.metadata = self.metadata[self.metadata["strat_fold"] < 9]
        elif split == "val":
            self.metadata = self.metadata[self.metadata["strat_fold"] == 9]
        elif split == "test":
            self.metadata = self.metadata[self.metadata["strat_fold"] == 10]

        # Carrega o mapeamento de diagnósticos
        self.label_map_geral = pd.read_csv(os.path.join(data_dir, "scp_statements.csv"), index_col=0)
        self.label_map = self.label_map_geral[self.label_map_geral.diagnostic == 1]

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
        # Códigos que também são diagnósticos (diagnostic==1) são excluídos daqui:
        # NDT/NST_/DIG/LNGQT têm form==1 E diagnostic==1 (mapeando pra STTC), então
        # incluí-los vazaria o próprio rótulo disfarçado de achado de forma.
        form_text = ""
        rhythm_text = ""
        for label in labels:
            if label not in self.label_map.index and not np.isnan(self.label_map_geral.loc[label].form):

                form_text += self.label_map_geral.loc[label].description + ", "

        for label in labels:
            if label not in self.label_map.index and not np.isnan(self.label_map_geral.loc[label].rhythm):
                rhythm_text += self.label_map_geral.loc[label].description + ", "

        generated_text = self.generate_prompt_from_metadata(record, form_text, rhythm_text)

        return generated_text

    def _extract_labels(self, scp_codes_str):
        """
        Converte o campo de string de SCP codes (ex: '{"NORM": 100, "MI": 50}')
        em uma lista de diagnósticos.
        """
        return parse_scp_codes(scp_codes_str)

    def generate_medical_text(
            self,
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

        return response


def get_metadata_robust_text(record):
    """Texto demográfico curto usado como texto DtT (ex: RawTextSource)."""
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
