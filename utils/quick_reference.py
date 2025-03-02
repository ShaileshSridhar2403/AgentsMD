import os
from datetime import datetime

def generate_quick_reference(case_id, esi_level, confidence, actions, chief_complaint=None, output_dir="quick_ref"):
    """
    Generate a quick reference file for nurses in action
    
    Args:
        case_id (str): The case identifier
        esi_level (str): The determined ESI level
        confidence (int): Confidence in the ESI determination
        actions (list): List of recommended actions
        chief_complaint (str, optional): The patient's chief complaint
        output_dir (str, optional): Directory to save the quick reference file
        
    Returns:
        str: Path to the generated quick reference file
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Format the quick reference
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{output_dir}/{case_id}_{timestamp}_QUICK.txt"
    
    with open(filename, "w") as f:
        f.write("="*40 + "\n")
        f.write("EMERGENCY TRIAGE - QUICK REFERENCE\n")
        f.write("="*40 + "\n\n")
        
        f.write(f"ESI LEVEL: {esi_level}\n")
        f.write(f"CONFIDENCE: {confidence}%\n")
        
        if chief_complaint:
            f.write(f"CHIEF COMPLAINT: {chief_complaint}\n")
        
        f.write("\nRECOMMENDED ACTION ITEMS:\n")
        for i, action in enumerate(actions, 1):
            f.write(f"{i}. {action}\n")
        
        f.write("\n" + "="*40 + "\n")
    
    return filename