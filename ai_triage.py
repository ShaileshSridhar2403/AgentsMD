import os
import argparse
import json
from datetime import datetime
import time
import re

# Import using direct paths
import sys
sys.path.append('.')  # Add current directory to path

from agents.triage_nurse import TriageNurseAgent
from agents.emergency_physician import EmergencyPhysicianAgent
from agents.medical_consultant import MedicalConsultantAgent
from utils.agent_discussion import AgentDiscussion

class ClinicalTriageSystem:
    def __init__(self, api_key=None, llm_backend="o1-mini", verbose=False):
        """
        Initialize the Clinical Triage System
        
        Args:
            api_key (str): API key for the LLM service
            llm_backend (str): LLM model to use
            verbose (bool): Whether to print verbose output
        """
        self.api_key = api_key
        self.llm_backend = llm_backend
        self.verbose = verbose
        
        # Set output directories
        self.output_dirs = {
            "results": "results",
            "discussions": "discussions",
            "quick_ref": "quick_ref"
        }
        
        # Initialize agents
        self.triage_nurse = TriageNurseAgent(model=llm_backend, api_key=api_key)
        self.emergency_physician = EmergencyPhysicianAgent(model=llm_backend, api_key=api_key)
        self.medical_consultant = MedicalConsultantAgent(model=llm_backend, api_key=api_key)
        
        # Initialize discussion framework
        self.agents = [self.triage_nurse, self.emergency_physician, self.medical_consultant]
        self.discussion = AgentDiscussion(
            agents=[self.triage_nurse, self.emergency_physician, self.medical_consultant],
            model=llm_backend,
            api_key=api_key
        )
        
        # Initialize assessment results
        self.assessment_results = None
        self.case_id = None
    
    def _generate_case_id(self):
        """Generate a unique case ID"""
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        return f"CASE-{timestamp}"
    
    def process_conversation(self, conversation_text):
        """
        Process a patient conversation and determine ESI level
        
        Args:
            conversation_text (str): The patient conversation text
            
        Returns:
            dict: The assessment results
        """
        # Generate a unique case ID
        case_id = f"case_{int(time.time())}"
        
        # Start the agent discussion
        discussion_file = self.discussion.deliberate(
            conversation_text=conversation_text,
            case_id=case_id
        )
        
        # Extract the assessment results from the discussion
        assessment_results = self.extract_assessment(discussion_file)
        
        # Store the results
        self.case_id = case_id
        self.conversation_text = conversation_text
        self.assessment_results = assessment_results
        
        # Generate the quick reference
        quick_ref_file = self.generate_quick_reference()
        
        # Generate detailed output
        detailed_output_file = self.generate_detailed_output()
        
        # Generate differential diagnoses
        differential_diagnoses_file = self.generate_differential_diagnoses()
        
        return assessment_results
    
    def save_assessment_results(self):
        """Save the assessment results to a file"""
        # Create results directory if it doesn't exist
        os.makedirs(self.output_dirs["results"], exist_ok=True)
        
        # Format the results for saving
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.output_dirs['results']}/{self.case_id}_{timestamp}.json"
        
        # Save as JSON
        with open(filename, 'w') as f:
            json.dump(self.assessment_results, f, indent=2)
        
        # Also save as human-readable text
        text_filename = f"{self.output_dirs['results']}/{self.case_id}_{timestamp}.txt"
        with open(text_filename, 'w') as f:
            f.write(f"Case ID: {self.case_id}\n")
            f.write(f"Timestamp: {timestamp}\n\n")
            f.write(f"ESI Level: {self.assessment_results['esi_level']}\n")
            f.write(f"Confidence: {self.assessment_results['confidence']}%\n\n")
            f.write(f"Justification:\n{self.assessment_results['justification']}\n\n")
            f.write("Recommended Actions:\n")
            for i, action in enumerate(self.assessment_results['recommended_actions'], 1):
                f.write(f"{i}. {action}\n")
        
        return text_filename
    
    def print_assessment(self):
        """Print the triage assessment in a formatted way"""
        if not self.assessment_results:
            print("No assessment has been performed yet.")
            return
        
        result = self.assessment_results
        
        print("\n" + "="*60)
        print(f"CLINICAL TRIAGE ASSESSMENT")
        print("="*60)
        print(f"Case ID: {result['case_id']}")
        print(f"Timestamp: {result['timestamp']}")
        print("\n")
        print(f"ESI LEVEL: {result['esi_level']}")
        print(f"Confidence: {result['confidence']}%")
        print("\n")
        print("CLINICAL JUSTIFICATION:")
        print(result['justification'])
        print("\n")
        print("RECOMMENDED ACTIONS:")
        for action in result['recommended_actions']:
            print(f"- {action}")
        print("\n")
        print("AGENT DISCUSSION SUMMARY:")
        print(result['discussion_summary'])
        print("="*60)

    def generate_quick_reference(self):
        """Generate a quick reference file for nurses in action"""
        from utils.quick_reference import generate_quick_reference
        
        # Extract chief complaint if available
        chief_complaint = None
        if hasattr(self, 'nurse_assessment') and self.nurse_assessment:
            chief_complaint = self.nurse_assessment.get('chief_complaint')
        
        # Generate the quick reference file
        quick_ref_file = generate_quick_reference(
            case_id=self.case_id,
            esi_level=self.assessment_results["esi_level"],
            confidence=self.assessment_results["confidence"],
            actions=self.assessment_results["recommended_actions"],
            chief_complaint=chief_complaint,
            output_dir=self.output_dirs["quick_ref"]  # Pass the custom directory
        )
        
        return quick_ref_file

    def generate_differential_diagnoses(self):
        """Generate potential differential diagnoses based on the assessment"""
        from utils.differential_diagnoses import generate_differential_diagnoses
        
        try:
            # Generate the differential diagnoses file
            diff_dx_file = generate_differential_diagnoses(
                case_id=self.case_id,
                assessment_results=self.assessment_results,
                output_dir="differential_diagnoses"  # Create a new directory for these files
            )
            
            # Ensure the result is a string
            if not isinstance(diff_dx_file, (str, bytes, os.PathLike)):
                if diff_dx_file is None:
                    return None
                return str(diff_dx_file)
            
            return diff_dx_file
        except Exception as e:
            print(f"Error generating differential diagnoses: {str(e)}")
            return None

    def extract_assessment(self, discussion_file):
        """
        Extract the assessment results from the discussion file
        
        Args:
            discussion_file (str): Path to the discussion file
            
        Returns:
            dict: The assessment results
        """
        # Read the discussion file
        with open(discussion_file, 'r') as f:
            discussion_text = f.read()
        
        # Extract the ESI level
        esi_match = re.search(r'ESI Level: (\d)', discussion_text)
        esi_level = esi_match.group(1) if esi_match else None
        
        # Extract the justification
        justification_match = re.search(r'Justification:(.*?)(?=\n\n|$)', discussion_text, re.DOTALL)
        justification = justification_match.group(1).strip() if justification_match else ""
        
        # Extract the recommended actions - THIS IS THE KEY PART TO FIX
        actions_match = re.search(r'Recommended Actions:(.*?)(?=\n\n|$)', discussion_text, re.DOTALL)
        if actions_match:
            actions_text = actions_match.group(1).strip()
            # Split by numbered items or bullet points and clean up
            actions = [re.sub(r'^\d+\.\s*|\*\s*', '', action.strip()) 
                      for action in re.split(r'\n\d+\.|\n\*', actions_text) 
                      if action.strip()]
        else:
            # If no actions found, generate them based on the ESI level and justification
            actions = self.generate_actions_from_assessment(esi_level, justification)
        
        # Extract the discussion summary
        summary_match = re.search(r'Discussion Summary:(.*?)(?=\n\n|$)', discussion_text, re.DOTALL)
        summary = summary_match.group(1).strip() if summary_match else ""
        
        # Calculate confidence
        confidence = 80  # Default confidence
        confidence_match = re.search(r'Confidence: (\d+)%', discussion_text)
        if confidence_match:
            confidence = int(confidence_match.group(1))
        
        # Extract chief complaint if available
        chief_complaint_match = re.search(r'Chief Complaint:(.*?)(?=\n\n|$)', discussion_text, re.DOTALL)
        chief_complaint = chief_complaint_match.group(1).strip() if chief_complaint_match else None
        
        return {
            "case_id": self.case_id,
            "esi_level": esi_level,
            "confidence": confidence,
            "justification": justification,
            "recommended_actions": actions,
            "discussion_summary": summary,
            "chief_complaint": chief_complaint
        }

    def generate_actions_from_assessment(self, esi_level, justification):
        """
        Generate recommended actions based on ESI level and justification
        when no actions are found in the discussion
        
        Args:
            esi_level (str): The ESI level (1-5)
            justification (str): The justification text
            
        Returns:
            list: List of recommended actions
        """
        from utils.query_model import query_model
        
        # Create a prompt for generating actions
        prompt = f"""
        Based on the following patient assessment with ESI level {esi_level}, 
        generate a list of 3-5 specific recommended actions for the healthcare team.
        
        Assessment: {justification}
        
        The actions should be specific to this patient's condition and ESI level.
        Format your response as a numbered list of actions only, without any introduction or explanation.
        """
        
        # System prompt for the model
        system_prompt = """
        You are an expert emergency medicine physician. 
        Your task is to recommend specific actions for the healthcare team based on a patient assessment.
        Focus on practical, actionable steps that are appropriate for the patient's ESI level and condition.
        Provide only the numbered list of actions, without any additional text.
        """
        
        # Generate the actions using the model
        model_response = query_model(
            model_str=self.llm_backend,
            system_prompt=system_prompt,
            prompt=prompt,
            openai_api_key=self.api_key
        )
        
        # Parse the response into a list of actions
        actions = []
        for line in model_response.strip().split('\n'):
            # Remove numbering and clean up
            clean_line = re.sub(r'^\d+\.\s*|\*\s*', '', line.strip())
            if clean_line:
                actions.append(clean_line)
        
        return actions

def main():
    parser = argparse.ArgumentParser(description="Clinical Triage System")
    parser.add_argument("--api-key", required=True, help="API key for LLM service")
    parser.add_argument("--input-file", help="Path to conversation text file")
    parser.add_argument("--input-text", help="Direct conversation text input")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    
    args = parser.parse_args()
    
    # Initialize the system
    triage_system = ClinicalTriageSystem(
        api_key=args.api_key,
        verbose=args.verbose
    )
    
    # Get conversation text
    conversation_text = ""
    if args.input_file:
        with open(args.input_file, 'r') as f:
            conversation_text = f.read()
    elif args.input_text:
        conversation_text = args.input_text
    else:
        print("Please provide conversation text via --input-file or --input-text")
        return
    
    # Process the conversation
    triage_system.process_conversation(conversation_text)
    
    # Print the assessment
    triage_system.print_assessment()

if __name__ == "__main__":
    main() 