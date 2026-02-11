import json
from tqdm import tqdm
from utils.dataset import PTBXLDataset_with_generated_prompt
import numpy as np
import sys
from ollama import generate

def generate_medical_text(
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
        
# TEXTO PARA ADICIONAR POSSIVELMENTE NO FUTURO
# Generate a concise clinical ECG report. Vary sentence structure and phrasing across patients, while keeping all clinical information accurate and faithful.
def generate_prompt_from_metadata(record, morphology_text, rhythm_text):
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
            rhythm_text
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
    

def generate_cache_llm(split='train', cache_filename="text_cache.json"):
    dataset = PTBXLDataset_with_generated_prompt(
        data_dir="/home/giovanidl/Datasets/PTBXL",
        split=split,
        sampling_rate=500
    )

    text_cache = {} 

    print("Generating LLM text cache for split:", split)
       
    for idx in tqdm(range(len(dataset))):
        
        
        
        record = dataset.records.iloc[idx]
        labels = dataset._extract_labels(record["scp_codes"])
        
        #print(labels)
        form_text= ""
        rhythm_text= ""
        for label in labels:
            if not np.isnan(dataset.label_map_geral.loc[label].form):

                form_text += dataset.label_map_geral.loc[label].description + ", "

        for label in labels:
            if not np.isnan(dataset.label_map_geral.loc[label].rhythm):
                rhythm_text += dataset.label_map_geral.loc[label].description + ", "


        generated_text = generate_prompt_from_metadata(record, form_text, rhythm_text)



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
    

        generated_text = generate_robust_text_from_metadata(record)



        ecg_filename = record["filename_hr"]  # use um ID único
        text_cache[ecg_filename] = generated_text
    

    with open(cache_filename, "w") as f:
        json.dump(text_cache, f)


def main():
    #generate_cache_llm(split='train', cache_filename="text_cache_train.json")
    #generate_cache_llm(split='val', cache_filename="/home/giovanidl/doutorado/prelim/cache/json_cache/text_cache_val.json")
    #generate_cache_llm(split='test', cache_filename="/home/giovanidl/doutorado/prelim/cache/json_cache/text_cache_test.json")
    generate_cache_robust_text(split='train', cache_filename="/home/giovanidl/doutorado/prelim/cache/json_cache/robust_text_cache_train.json")
    generate_cache_robust_text(split='val', cache_filename="/home/giovanidl/doutorado/prelim/cache/json_cache/robust_text_cache_val.json")
    generate_cache_robust_text(split='test', cache_filename="/home/giovanidl/doutorado/prelim/cache/json_cache/robust_text_cache_test.json")

if __name__ == "__main__":
    main()
