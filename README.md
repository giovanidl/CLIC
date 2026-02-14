# CLIC


## Prompt for CLIC-LLM

"""
    You are a cardiology specialist.

    Generate a concise, single-paragraph clinical ECG report based on the information below.
    Use formal medical English, objective tone, and clear clinical reasoning.

    Patient information:
        

        Age: {age} years
        Sex: {sex}
        Weight: {weight} kg
        Body Mass Index: {bmi}
        Recording device: {collection_device}


    Electrocardiographic findings:
        

        Signal morphology: {morphology_text}
        Cardiac rhythm: {rhythm_text}


    End the report with a complete sentence and avoid bullet points or lists, and use all the information given above. 
    Don't calculate the BMI yourself, always use the given BMI, just use the height and weight information if available.
    Don't start the report with "Here is the clinical report" or similar phrases.
    Don't ever provide information, such as 70 bpm heart rate, that is not given in the input. Only assumptions that can be made using the given input.
    If the height is missing, it has a high chance of being above 40, according to the dataset paper, so it's safe to assume that the Body Mass Index of a patient with missing height data is above 40.
    Don't include the unit of the Body Mass Index in the report, just say "has a BMI of 32", for example.
"""