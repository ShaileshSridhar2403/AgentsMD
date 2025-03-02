import os
import argparse
import json
from datetime import datetime

# Import using direct paths
import sys
sys.path.append('.')  # Add current directory to path

from agents.triage_nurse import TriageNurseAgent
from agents.emergency_physician import EmergencyPhysicianAgent
from agents.medical_consultant import MedicalConsultantAgent
from utils.agent_discussion import AgentDiscussion

class ClinicalTriageSystem:
    def __init__(self, api_key, llm_backend="gpt-4o-mini", verbose=True):
        """
        Initialize the Clinical Triage System
        
        Args:
            api_key (str): API key for the LLM service
            llm_backend (str): LLM model to use
            verbose (bool): Whether to print detailed output
        """
        self.api_key = api_key
        self.llm_backend = llm_backend
        self.verbose = verbose
        
        # Initialize agents
        self.triage_nurse = TriageNurseAgent(model=llm_backend, api_key=api_key)
        self.emergency_physician = EmergencyPhysicianAgent(model=llm_backend, api_key=api_key)
        self.medical_consultant = MedicalConsultantAgent(model=llm_backend, api_key=api_key)
        
        # Initialize agent discussion
        self.agent_discussion = AgentDiscussion(
            agents=[self.triage_nurse, self.emergency_physician, self.medical_consultant],
            model=llm_backend,
            api_key=api_key
        )
        
        # Session data
        self.case_id = self._generate_case_id()
        self.timestamp = datetime.now()
        self.assessment_results = {}
    
    def _generate_case_id(self):
        """Generate a unique case ID"""
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        return f"CASE-{timestamp}"
    
    def process_conversation(self, conversation_text):
        """
        Process a patient-nurse conversation through agent discussion
        
        Args:
            conversation_text (str): The text of the conversation
            
        Returns:
            dict: Triage assessment results with ESI level
        """
        if self.verbose:
            print(f"Processing case {self.case_id}...")
            print("Beginning agent discussion for ESI determination...")
        
        # Conduct agent discussion to determine ESI level
        discussion_result = self.agent_discussion.deliberate(
            conversation_text=conversation_text,
            case_id=self.case_id
        )
        
        # Store and return results
        self.assessment_results = {
            "case_id": self.case_id,
            "timestamp": self.timestamp.isoformat(),
            "esi_level": discussion_result["esi_level"],
            "confidence": discussion_result["confidence"],
            "justification": discussion_result["justification"],
            "recommended_actions": discussion_result["recommended_actions"],
            "discussion_summary": discussion_result["discussion_summary"]
        }
        
        # Save the assessment results to a file
        self.save_assessment_results()
        
        # Generate quick reference for nurses
        self.generate_quick_reference()
        
        if self.verbose:
            print(f"ESI determination complete: Level {discussion_result['esi_level']}")
        
        return self.assessment_results
    
    def save_assessment_results(self):
        """Save the assessment results to a file"""
        # Create results directory if it doesn't exist
        os.makedirs("results", exist_ok=True)
        
        # Format the results for saving
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"results/{self.case_id}_{timestamp}.json"
        
        # Save as JSON for structured data
        with open(filename, "w") as f:
            json.dump(self.assessment_results, f, indent=2)
        
        # Also save a human-readable text version
        text_filename = f"results/{self.case_id}_{timestamp}.txt"
        with open(text_filename, "w") as f:
            result = self.assessment_results
            
            f.write("="*60 + "\n")
            f.write(f"CLINICAL TRIAGE ASSESSMENT\n")
            f.write("="*60 + "\n")
            f.write(f"Case ID: {result['case_id']}\n")
            f.write(f"Timestamp: {result['timestamp']}\n\n")
            
            f.write(f"ESI LEVEL: {result['esi_level']}\n")
            f.write(f"Confidence: {result['confidence']}%\n\n")
            
            f.write("CLINICAL JUSTIFICATION:\n")
            f.write(f"{result['justification']}\n\n")
            
            f.write("RECOMMENDED ACTIONS:\n")
            for action in result['recommended_actions']:
                f.write(f"- {action}\n")
            f.write("\n")
            
            f.write("AGENT DISCUSSION SUMMARY:\n")
            f.write(f"{result['discussion_summary']}\n")
            f.write("="*60 + "\n")
        
        if self.verbose:
            print(f"Assessment results saved to {filename} and {text_filename}")
        
        return filename
    
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
        if hasattr(self, 'triage_nurse') and hasattr(self.triage_nurse, 'assessment'):
            chief_complaint = self.triage_nurse.assessment.get("chief_complaint", None)
        
        # Generate the quick reference file
        quick_ref_file = generate_quick_reference(
            case_id=self.case_id,
            esi_level=self.assessment_results["esi_level"],
            confidence=self.assessment_results["confidence"],
            actions=self.assessment_results["recommended_actions"],
            chief_complaint=chief_complaint
        )
        
        if self.verbose:
            print(f"Quick reference for nurses generated: {quick_ref_file}")
        
        return quick_ref_file

def main():
    parser = argparse.ArgumentParser(description="Clinical Triage System")
    parser.add_argument("--api-key", required=True, help="API key for LLM service")
    parser.add_argument("--llm-backend", default="gpt-4o-mini", help="LLM model to use")
    parser.add_argument("--input-file", help="Path to conversation text file")
    parser.add_argument("--input-text", help="Direct conversation text input")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    
    args = parser.parse_args()
    
    # Initialize the system
    triage_system = ClinicalTriageSystem(
        api_key=args.api_key,
        llm_backend=args.llm_backend,
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