import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from utils.dataset import PTBXLDataset_with_generated_prompt
import numpy as np
import sys
from ollama import generate
import pandas as pd
import os
def generate_medical_text(
            age,
            sex,
            weight,
            bmi,
            collection_device,
            morphology_text,
            rhythm_text,
            num_predict=180,
            model="llama3.1:8b"
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
                model=model,
                prompt=prompt.strip(),
                think=False,  # alguns modelos (ex: qwen3.5) gastam tokens "pensando"
                              # antes da resposta; sem isso o num_predict pode
                              # estourar antes do texto final ser gerado.
                options={
                    "num_predict": num_predict,
                    "temperature": 0.0,
                    "top_p": 0.9,
                    "repeat_penalty": 1.1
                }
            )["response"]

            return response.strip()
        
# TEXTO PARA ADICIONAR POSSIVELMENTE NO FUTURO
# Generate a concise clinical ECG report. Vary sentence structure and phrasing across patients, while keeping all clinical information accurate and faithful.
def generate_prompt_from_metadata(record, morphology_text, rhythm_text, model="llama3.1:8b"):
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
        
        
        response = generate_medical_text(
            text_age,
            text_sex,
            weight_text,
            bmi_text,
            device_text,
            morphology_text,
            rhythm_text,
            model=model
        )


        return response

def generate_robust_text_from_metadata(record):
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
            else f", weighs {int(weight)} kg"
        )
        bmi_text = ""
        if np.isnan(height):
            bmi_text = "and with unknown BMI (probably above 40)."
        elif not np.isnan(weight):
            bmi_text = f"and has a BMI of {int(weight / (height / 100) ** 2)}."  
            
        device_text = f"The device used was {collection_device}".strip() + "."

        return f"{text_age}, {text_sex}{weight_text} {bmi_text} {device_text}"
    

def generate_cache_llm(split='train', cache_filename="text_cache.json", model="llama3.1:8b", max_workers=8):
    dataset = PTBXLDataset_with_generated_prompt(
        data_dir="/home/giovanidl/Datasets/PTBXL",
        split=split,
        sampling_rate=500
    )

    text_cache = {}

    print(f"Generating LLM text cache (model={model}) for split:", split)

    def generate_one(idx):
        record = dataset.records.iloc[idx]
        labels = dataset._extract_labels(record["scp_codes"])

        #print(labels)
        # Códigos que também são diagnósticos (diagnostic==1) são excluídos daqui:
        # NDT/NST_/DIG/LNGQT têm form==1 E diagnostic==1 (mapeando pra STTC), então
        # incluí-los em form_text vazaria o próprio rótulo disfarçado de achado de forma.
        form_text = ""
        rhythm_text = ""
        for label in labels:
            if label not in dataset.label_map.index and not np.isnan(dataset.label_map_geral.loc[label].form):
                form_text += dataset.label_map_geral.loc[label].description + ", "

        for label in labels:
            if label not in dataset.label_map.index and not np.isnan(dataset.label_map_geral.loc[label].rhythm):
                rhythm_text += dataset.label_map_geral.loc[label].description + ", "

        generated_text = generate_prompt_from_metadata(record, form_text, rhythm_text, model=model)

        ecg_filename = record["filename_hr"]  # use um ID único
        return ecg_filename, generated_text

    # Ollama processa varias requisicoes concorrentes na mesma GPU/modelo ja
    # carregado (o modelo NAO e reinstanciado a cada chamada) — despachar em
    # paralelo (I/O-bound, aguardando resposta HTTP) da um ganho real de
    # throughput em vez de mandar uma chamada de cada vez.
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(generate_one, idx) for idx in range(len(dataset))]
        for future in tqdm(as_completed(futures), total=len(futures)):
            ecg_filename, generated_text = future.result()
            text_cache[ecg_filename] = generated_text

    with open(cache_filename, "w") as f:
        json.dump(text_cache, f)



def generate_cache_robust_text2(split='train', cache_filename="text_cache.json"):
    dataset = PTBXLDataset_with_generated_prompt(
        data_dir="/home/giovanidl/Datasets/PTBXL",
        split=split,
        sampling_rate=250
    )

    text_cache = {} 

    print("Generating robust text cache for split:", split)
       
    for idx in tqdm(range(len(dataset))):
        
        
        
        record = dataset.records.iloc[idx]
        labels = dataset._extract_labels(record["scp_codes"])
        label_map_geral = pd.read_csv(os.path.join(dataset.data_dir, "scp_statements.csv"), index_col=0)
        diagnostic_codes = label_map_geral[label_map_geral.diagnostic == 1].index

        age = record['age']
        sex = record['sex'] 
        weight = record['weight']
        collection_device = record['device'].split(" ")[0].replace("-", "")
        height = record['height']
        
        if np.isnan(height):
            height = -1  # valor médio aproximado para pacientes adultos, pode ser ajustado conforme necessário
        if np.isnan(weight):
            weight = -1  # valor médio aproximado para pacientes adultos, pode ser ajustado conforme necessário
        
        
        labels_device = ["AT60", "CS12", "CS100","AT6"]
        collection_device_nmb = labels_device.index(collection_device)

  
        labels_form = [
         "NDT","NST_","DIG","LNGQT","ABQRS","PVC","STD_","VCLVH","QWAVE",
         "LOWT","NT_","PAC","LPR","INVT","LVOLT","HVOLT","TAB_","STE_","PRC(S)"]

        labels_rhythm= [
        "SR","AFIB","STACH","SARRH","SBRAD","PACE","SVARR","BIGU","AFLT","SVTAC","PSVT","TRIGU"]
        
        # Códigos que também são diagnósticos (diagnostic==1) são excluídos daqui:
        # NDT/NST_/DIG/LNGQT têm form==1 E diagnostic==1 (mapeando pra STTC), então
        # incluí-los vazaria o próprio rótulo disfarçado de atributo de forma.
        form_nmb = -1
        rhythm_nmb = -1
        for label in labels:
            if label in diagnostic_codes:
                continue
            if not np.isnan(label_map_geral.loc[label].form):
                form_nmb = labels_form.index(label)
            if not np.isnan(label_map_geral.loc[label].rhythm):
                rhythm_nmb = labels_rhythm.index(label)
        
        
        generated_text = f"{age},{sex},{weight},{height},{collection_device_nmb},{form_nmb},{rhythm_nmb}" 

        ecg_filename = record["filename_hr"]  # use um ID único
        text_cache[ecg_filename] = generated_text
    

    with open(cache_filename, "w") as f:
        json.dump(text_cache, f)        
        
        
def generate_cache_robust_text(split='train', cache_filename="text_cache.json"):
    dataset = PTBXLDataset_with_generated_prompt(
        data_dir="/home/giovanidl/Datasets/PTBXL",
        split=split,
        sampling_rate=250
    )

    text_cache = {} 

    print("Generating robust text cache for split:", split)
       
    for idx in tqdm(range(len(dataset))):
        
        
        
        record = dataset.records.iloc[idx]
        labels = dataset._extract_labels(record["scp_codes"])
        label_map_geral = pd.read_csv(os.path.join(dataset.data_dir, "scp_statements.csv"), index_col=0)
        diagnostic_codes = label_map_geral[label_map_geral.diagnostic == 1].index

        generated_text = generate_robust_text_from_metadata(record)
        #print(labels)
        # Códigos que também são diagnósticos (diagnostic==1) são excluídos daqui:
        # NDT/NST_/DIG/LNGQT têm form==1 E diagnostic==1 (mapeando pra STTC), então
        # incluí-los vazaria o próprio rótulo disfarçado de achado de forma.
        form_text= ""
        rhythm_text= ""
        for label in labels:
            if label in diagnostic_codes:
                continue
            if not np.isnan(label_map_geral.loc[label].form):
                form_text += label_map_geral.loc[label].description + ", "


        for label in labels:
            if label in diagnostic_codes:
                continue
            if not np.isnan(label_map_geral.loc[label].rhythm):
                rhythm_text += label_map_geral.loc[label].description + ", "
        
        generated_text = generated_text + f" {form_text.strip(', ')} {rhythm_text.strip(', ')}"


        ecg_filename = record["filename_hr"]  # use um ID único
        text_cache[ecg_filename] = generated_text
    

    with open(cache_filename, "w") as f:
        json.dump(text_cache, f)


def main():
    # ja gerados anteriormente com llama3.1:8b / DtT estruturado:
    # generate_cache_llm(split='train', cache_filename="/home/giovanidl/doutorado/prelim/cache/json_cache/llm_text_cache_train.json")
    # generate_cache_llm(split='val', cache_filename="/home/giovanidl/doutorado/prelim/cache/json_cache/llm_text_cache_val.json")
    # generate_cache_llm(split='test', cache_filename="/home/giovanidl/doutorado/prelim/cache/json_cache/llm_text_cache_test.json")
    # generate_cache_robust_text(split='train', cache_filename="/home/giovanidl/doutorado/prelim/cache/json_cache/robust_text_cache_train.json")
    # generate_cache_robust_text(split='val', cache_filename="/home/giovanidl/doutorado/prelim/cache/json_cache/robust_text_cache_val.json")
    # generate_cache_robust_text(split='test', cache_filename="/home/giovanidl/doutorado/prelim/cache/json_cache/robust_text_cache_test.json")

    # ja gerado com qwen3.5:9b:
    # generate_cache_llm(split='train', cache_filename="/home/giovanidl/doutorado/prelim/cache/json_cache/qwen_text_cache_train.json", model="qwen3.5:9b")
    # generate_cache_llm(split='val', cache_filename="/home/giovanidl/doutorado/prelim/cache/json_cache/qwen_text_cache_val.json", model="qwen3.5:9b")
    # generate_cache_llm(split='test', cache_filename="/home/giovanidl/doutorado/prelim/cache/json_cache/qwen_text_cache_test.json", model="qwen3.5:9b")

    # novo: mesmo prompt/pipeline, mas gerado com gemma4:12b
    # max_workers=6 (nao 8): o Gemma é maior que o Qwen (7.6GB vs 6.6GB) e
    # ocupou mais VRAM concorrente no teste (8.7GB de 12.3GB com 8 workers) —
    # margem de seguranca menor, dado o travamento anterior.
    generate_cache_llm(
        split='train',
        cache_filename="/home/giovanidl/doutorado/prelim/cache/json_cache/gemma_text_cache_train.json",
        model="gemma4:12b",
        max_workers=6,
    )
    generate_cache_llm(
        split='val',
        cache_filename="/home/giovanidl/doutorado/prelim/cache/json_cache/gemma_text_cache_val.json",
        model="gemma4:12b",
        max_workers=6,
    )
    generate_cache_llm(
        split='test',
        cache_filename="/home/giovanidl/doutorado/prelim/cache/json_cache/gemma_text_cache_test.json",
        model="gemma4:12b",
        max_workers=6,
    )

if __name__ == "__main__":
    main()
