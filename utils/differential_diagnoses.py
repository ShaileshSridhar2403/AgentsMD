import os
import json
from datetime import datetime
from utils.query_model import query_model

def generate_differential_diagnoses(case_id, assessment_results, output_dir="differential_diagnoses"):
    """
    Generate potential differential diagnoses based on the assessment
    
    Args:
        case_id (str): The case ID
        assessment_results (dict): The assessment results
        output_dir (str): Directory to save the output file
        
    Returns:
        str: Path to the generated file
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Ensure assessment_results is a dictionary
    if not isinstance(assessment_results, dict):
        print(f"Error: assessment_results is not a dictionary, got {type(assessment_results)}")
        return None
    
    # Extract relevant information
    esi_level = assessment_results.get("esi_level", "Unknown")
    justification = assessment_results.get("justification", "")
    chief_complaint = assessment_results.get("chief_complaint", "")
    
    # Format the timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create the filename
    filename = f"{output_dir}/{case_id}_differential_{timestamp}.txt"
    
    # Generate the differential diagnoses using a model
    from utils.query_model import query_model
    
    # Create a prompt for generating differential diagnoses
    prompt = f"""
    Based on the following patient information, provide a list of potential differential diagnoses:
    
    ESI Level: {esi_level}
    Chief Complaint: {chief_complaint}
    Clinical Assessment: {justification}
    
    Please list 3-5 potential diagnoses in order of likelihood, with a brief explanation for each.
    """
    
    # Get the API key from environment
    api_key = os.getenv('OPENAI_API_KEY')
    
    # Generate the differential diagnoses
    try:
        response = query_model(
            model_str="o1-mini",  # Use a consistent model for this task
            system_prompt="You are an expert emergency medicine physician. Your task is to generate differential diagnoses based on patient information.",
            prompt=prompt,
            openai_api_key=api_key
        )
        
        # Format the output
        output = f"""# Differential Diagnoses

Case ID: {case_id}
Generated: {timestamp}

## Patient Information
- ESI Level: {esi_level}
- Chief Complaint: {chief_complaint}

## Potential Diagnoses
{response}

## Disclaimer
This is an AI-generated list of potential diagnoses for consideration only. 
Clinical judgment and further evaluation are required for definitive diagnosis.
"""
        
        # Save to file
        with open(filename, "w") as f:
            f.write(output)
        
        return filename
    except Exception as e:
        print(f"Error generating differential diagnoses: {str(e)}")
        return None 