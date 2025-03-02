import re
import json
import os
from datetime import datetime
from utils.query_model import query_model

class AgentDiscussion:
    def __init__(self, agents, model="gpt-4o-mini", api_key=None, max_rounds=3):
        """
        Initialize the Agent Discussion framework
        
        Args:
            agents (list): List of agent objects
            model (str): LLM model to use
            api_key (str): API key for the LLM service
            max_rounds (int): Maximum number of discussion rounds
        """
        self.agents = agents
        self.model = model
        self.api_key = api_key
        self.max_rounds = max_rounds
        
        # ESI level descriptions for reference
        self.esi_descriptions = {
            "1": "Requires immediate life-saving intervention",
            "2": "High risk situation; severe pain/distress",
            "3": "Requires multiple resources but stable vital signs",
            "4": "Requires one resource",
            "5": "Requires no resources"
        }
    
    def deliberate(self, conversation_text, case_id=None, progress_callback=None):
        """
        Conduct a deliberation among agents to determine ESI level
        
        Args:
            conversation_text (str): The text of the conversation
            case_id (str, optional): Case identifier
            progress_callback (callable, optional): Callback function to report progress
            
        Returns:
            dict: Results of the deliberation
        """
        # Initialize discussion
        discussion_history = []
        
        # Initial assessments
        if progress_callback:
            progress_callback("Triage Nurse is analyzing the conversation...", 15)
        
        nurse_assessment = self.agents[0].assess_conversation(conversation_text)
        
        if progress_callback:
            # Get a summary from the assessment, safely handling different formats
            nurse_summary = nurse_assessment.get('summary', 'Assessment completed')
            progress_callback(f"Triage Nurse: {nurse_summary[:100]}...", 25)
        
        if progress_callback:
            progress_callback("Emergency Physician is evaluating the case...", 35)
        
        physician_assessment = self.agents[1].assess_conversation(conversation_text)
        
        if progress_callback:
            # Get a summary from the assessment, safely handling different formats
            physician_summary = physician_assessment.get('summary', 'Assessment completed')
            progress_callback(f"Emergency Physician: {physician_summary[:100]}...", 45)
        
        if progress_callback:
            progress_callback("Medical Consultant is reviewing the case...", 55)
        
        consultant_assessment = self.agents[2].assess_conversation(conversation_text)
        
        if progress_callback:
            # Get a summary from the assessment, safely handling different formats
            consultant_summary = consultant_assessment.get('summary', 'Assessment completed')
            progress_callback(f"Medical Consultant: {consultant_summary[:100]}...", 65)
        
        # Add to discussion history
        discussion_history.append({
            "role": "Triage Nurse",
            "content": f"Initial assessment: {self._summarize_assessment(nurse_assessment)}"
        })
        discussion_history.append({
            "role": "Emergency Physician",
            "content": f"Initial assessment: {self._summarize_assessment(physician_assessment)}"
        })
        discussion_history.append({
            "role": "Medical Consultant",
            "content": f"Initial assessment: {self._summarize_assessment(consultant_assessment)}"
        })
        
        # Round 2: Agents respond to each other's assessments
        print("Round 2: Agents responding to each other's assessments...")
        
        # Create a dictionary of all assessments
        all_assessments = {
            self.agents[0].role: nurse_assessment,
            self.agents[1].role: physician_assessment,
            self.agents[2].role: consultant_assessment
        }
        
        for agent in self.agents:
            print(f"  - {agent.role} is responding to other assessments...")
            response = agent.respond_to_assessments(conversation_text, all_assessments)
            
            # Add to discussion history
            discussion_history.append({
                "role": agent.role,
                "content": response
            })
        
        # Round 3: Final deliberation and consensus
        print("Round 3: Final deliberation and consensus...")
        consensus_prompt = self._create_consensus_prompt(discussion_history, conversation_text)
        
        consensus_result = query_model(
            model_str=self.model,
            system_prompt=self._get_consensus_system_prompt(),
            prompt=consensus_prompt,
            openai_api_key=self.api_key
        )
        
        # Parse the consensus result
        final_result = self._parse_consensus_result(consensus_result)
        
        # Add discussion summary
        final_result["discussion_summary"] = self._generate_discussion_summary(discussion_history)
        
        # Save the full discussion to a file
        self._save_discussion(discussion_history, case_id, final_result)
        
        # During discussion
        if progress_callback:
            progress_callback("Agents are discussing ESI determination...", 75)
        
        # After reaching consensus
        if progress_callback:
            progress_callback(f"Consensus reached: ESI Level {final_result['esi_level']} with {final_result['confidence']}% confidence", 85)
        
        return final_result
    
    def _save_discussion(self, discussion_history, case_id, final_result):
        """Save the full discussion to a file"""
        # Create discussions directory if it doesn't exist
        os.makedirs("discussions", exist_ok=True)
        
        # Format the discussion for saving
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"discussions/{case_id}_{timestamp}.txt"
        
        with open(filename, "w") as f:
            f.write(f"CASE ID: {case_id}\n")
            f.write(f"TIMESTAMP: {datetime.now().isoformat()}\n\n")
            f.write("FULL AGENT DISCUSSION:\n")
            f.write("="*80 + "\n\n")
            
            for entry in discussion_history:
                f.write(f"[{entry['role']}]\n")
                f.write(f"{entry['content']}\n\n")
                f.write("-"*80 + "\n\n")
            
            f.write("="*80 + "\n\n")
            f.write("FINAL CONSENSUS:\n")
            f.write(f"ESI Level: {final_result['esi_level']}\n")
            f.write(f"Confidence: {final_result['confidence']}%\n")
            f.write(f"Justification: {final_result['justification']}\n\n")
            f.write("Recommended Actions:\n")
            for action in final_result['recommended_actions']:
                f.write(f"- {action}\n")
        
        print(f"Full discussion saved to {filename}")
        
        return filename
    
    def _summarize_assessment(self, assessment):
        """Create a summary of an agent's assessment"""
        # If the assessment already has a summary, use it
        if assessment.get('summary'):
            return assessment['summary']
        
        # Otherwise, try to create a summary based on available fields
        esi_level = ""
        if assessment.get('recommended_esi'):
            esi_match = re.search(r'(\d+)', assessment['recommended_esi'])
            if esi_match:
                esi_level = esi_match.group(1)
        elif assessment.get('esi_level'):
            esi_match = re.search(r'(\d+)', assessment['esi_level'])
            if esi_match:
                esi_level = esi_match.group(1)
        elif assessment.get('esi_evaluation'):
            esi_match = re.search(r'(\d+)', assessment['esi_evaluation'])
            if esi_match:
                esi_level = esi_match.group(1)
        
        # Get a rationale or assessment
        rationale = ""
        if assessment.get('rationale'):
            rationale = assessment['rationale'][:100]
        elif assessment.get('clinical_assessment'):
            rationale = assessment['clinical_assessment'][:100]
        elif assessment.get('specialist_impression'):
            rationale = assessment['specialist_impression'][:100]
        elif assessment.get('initial_impression'):
            rationale = assessment['initial_impression'][:100]
        
        if esi_level and rationale:
            return f"ESI Level: {esi_level}. Rationale: {rationale}..."
        elif esi_level:
            return f"ESI Level: {esi_level}."
        elif rationale:
            return f"Assessment: {rationale}..."
        else:
            return "Assessment completed."
    
    def _create_consensus_prompt(self, discussion_history, conversation_text):
        """Create a prompt for the final consensus"""
        prompt = [
            "Based on the patient-nurse conversation and the discussion between medical professionals below, "
            "determine the most appropriate ESI (Emergency Severity Index) level for this patient.",
            "",
            "PATIENT-NURSE CONVERSATION:",
            conversation_text,
            "",
            "DISCUSSION TRANSCRIPT:"
        ]
        
        for entry in discussion_history:
            prompt.append(f"{entry['role']}: {entry['content']}")
        
        prompt.append("")
        prompt.append("Please analyze the discussion and determine:")
        prompt.append("1. The final ESI level (1-5)")
        prompt.append("2. Confidence level (0-100%)")
        prompt.append("3. Clinical justification for this ESI level that references specific findings from this case")
        prompt.append("4. Recommended immediate actions (provide at least 3-5 specific actions)")
        prompt.append("")
        prompt.append("IMPORTANT: Your recommended actions MUST be specific to this patient's condition and presentation.")
        prompt.append("Do NOT provide generic recommendations like 'establish IV access' or 'monitor vital signs' without specifying WHY and HOW these actions relate to this specific patient.")
        prompt.append("Each recommendation should include the specific reason for the action based on the patient's symptoms or condition.")
        
        return "\n".join(prompt)
    
    def _get_consensus_system_prompt(self):
        """Get the system prompt for the consensus model"""
        return """
        You are an experienced emergency department medical director with over 25 years of experience.
        Your role is to review discussions between medical professionals and make the final determination
        on Emergency Severity Index (ESI) levels for patients.
        
        The ESI is a 5-level triage system:
        - Level 1: Requires immediate life-saving intervention
        - Level 2: High risk situation or severe pain/distress
        - Level 3: Requires multiple resources but stable vital signs
        - Level 4: Requires one resource
        - Level 5: Requires no resources
        
        IMPORTANT: You MUST provide at least 3-5 SPECIFIC recommended actions for the patient based on their ESI level AND the specific details of their case.
        
        DO NOT provide generic recommendations. Each recommendation must be directly related to the patient's specific symptoms, history, and presentation.
        
        For example:
        - Instead of "Establish IV access" → "Establish IV access for fluid resuscitation due to signs of dehydration"
        - Instead of "Order appropriate tests" → "Order CBC, BMP, and urinalysis to evaluate suspected UTI"
        - Instead of "Administer pain medication" → "Administer acetaminophen 1000mg PO for fever of 101.3°F"
        
        For ESI Level 1-2 patients, include immediate interventions specific to their life-threatening condition.
        For ESI Level 3 patients, include diagnostic and treatment recommendations targeting their specific presentation.
        For ESI Level 4-5 patients, include appropriate care instructions that address their specific complaint.
        
        Your final determination must include:
        1. ESI Level (1-5)
        2. Confidence level (0-100%)
        3. Detailed clinical justification referencing specific findings from the case
        4. Specific recommended actions (at least 3-5) tailored to this exact patient
        
        Be decisive and consider all perspectives from the discussion, but ensure your recommendations are highly specific to this individual patient case.
        """
    
    def _parse_consensus_result(self, result):
        """Parse the consensus result into a structured format"""
        # Extract ESI level - try multiple patterns
        esi_match = re.search(r'(?:ESI Level|Level|Final ESI Level):\s*(\d)', result, re.IGNORECASE)
        if not esi_match:
            # Try to find any digit after ESI or Level
            esi_match = re.search(r'ESI.*?(\d)', result, re.IGNORECASE)
        if not esi_match:
            # Try to find any standalone digit that might be the ESI level
            esi_match = re.search(r'(?:^|\n|\s)(\d)(?:$|\n|\s)', result)
        
        # If we found a match, use it; otherwise default to level 3 (middle ground)
        esi_level = esi_match.group(1) if esi_match else "3"
        
        # Validate ESI level (must be 1-5)
        if esi_level not in ["1", "2", "3", "4", "5"]:
            # Default to level 3 if invalid
            esi_level = "3"
        
        # Extract confidence
        confidence_match = re.search(r'Confidence(?:\s*Level)?:\s*(\d+)%?', result, re.IGNORECASE)
        confidence = int(confidence_match.group(1)) if confidence_match else 80
        
        # Extract justification
        justification_match = re.search(r'(?:Justification|Clinical Justification|Rationale|Clinical Justification for ESI Level):(.*?)(?=Recommended(?:\s*Immediate)?\s*Actions|\Z)', result, re.DOTALL | re.IGNORECASE)
        justification = justification_match.group(1).strip() if justification_match else "No justification provided."
        
        # Extract recommended actions
        actions = []
        actions_match = re.search(r'Recommended(?:\s*Immediate)?\s*Actions:(.*?)(?=\Z|\n\s*\d+\.)', result, re.DOTALL | re.IGNORECASE)
        if actions_match:
            actions_text = actions_match.group(1).strip()
            # Extract actions as a list - look for bullet points or numbered items
            actions_list = re.findall(r'(?:^|\n)\s*(?:-|\d+\.)\s*(.*?)(?=\n\s*(?:-|\d+\.)|\Z)', actions_text, re.DOTALL)
            actions = [a.strip() for a in actions_list if a.strip() and not a.startswith("**")]
        
        # If no actions found or actions contain meta-instructions, try a different approach
        if not actions or any("**" in action for action in actions):
            # Look for any bullet points or numbered items in the entire text
            actions_list = re.findall(r'(?:^|\n)\s*(?:-|\d+\.)\s*(.*?)(?=\n\s*(?:-|\d+\.)|\Z)', result, re.DOTALL)
            # Filter out meta-instructions and keep only reasonable-length action items
            actions = [a.strip() for a in actions_list if a.strip() 
                      and not a.startswith("**") 
                      and "ESI Level" not in a 
                      and "Confidence" not in a
                      and "Justification" not in a
                      and "Recommended Actions" not in a
                      and len(a) < 200]  # Allow longer actions for more specificity
        
        # If still no actions, generate default actions based on ESI level and justification
        if not actions:
            # Extract key symptoms or conditions from the justification
            key_symptoms = []
            if "chest pain" in justification.lower():
                key_symptoms.append("chest pain")
            if "shortness of breath" in justification.lower() or "sob" in justification.lower():
                key_symptoms.append("respiratory distress")
            if "fever" in justification.lower():
                key_symptoms.append("fever")
            if "bleeding" in justification.lower():
                key_symptoms.append("bleeding")
            if "trauma" in justification.lower() or "injury" in justification.lower():
                key_symptoms.append("trauma")
            if "pain" in justification.lower():
                key_symptoms.append("pain")
            
            # Default symptom if none detected
            if not key_symptoms:
                key_symptoms = ["presenting condition"]
            
            # Generate more specific actions based on ESI level and symptoms
            if esi_level == "1":
                actions = [
                    f"Immediate intervention by emergency physician for {' and '.join(key_symptoms)}",
                    f"Prepare resuscitation equipment appropriate for {' and '.join(key_symptoms)}",
                    f"Establish two large-bore IV access for immediate medication administration and fluid resuscitation",
                    "Continuous cardiac monitoring and vital sign checks every 2-3 minutes",
                    f"Notify critical care team for possible ICU admission due to {' and '.join(key_symptoms)}"
                ]
            elif esi_level == "2":
                actions = [
                    f"Urgent assessment by emergency physician within 10 minutes to evaluate {' and '.join(key_symptoms)}",
                    "Establish IV access for medication and fluid administration",
                    "Continuous vital sign monitoring every 5-10 minutes",
                    f"Administer appropriate medication for {' and '.join(key_symptoms)} after physician assessment",
                    f"Order diagnostic studies specific to {' and '.join(key_symptoms)} including labs and imaging"
                ]
            elif esi_level == "3":
                actions = [
                    f"Assessment by emergency physician within 30 minutes to evaluate {' and '.join(key_symptoms)}",
                    "Obtain baseline vital signs and repeat every 1-2 hours",
                    f"Order diagnostic tests appropriate for {' and '.join(key_symptoms)}",
                    "Establish IV access if needed for medication administration",
                    f"Provide symptomatic treatment for {' and '.join(key_symptoms)} as ordered"
                ]
            elif esi_level == "4":
                actions = [
                    f"Assessment by provider within 60 minutes to evaluate {' and '.join(key_symptoms)}",
                    "Obtain baseline vital signs",
                    f"Focused examination of {' and '.join(key_symptoms)}",
                    f"Consider appropriate testing for {' and '.join(key_symptoms)} if clinically indicated",
                    f"Provide symptomatic relief for {' and '.join(key_symptoms)} as appropriate"
                ]
            else:  # ESI level 5
                actions = [
                    f"Assessment by provider when available to evaluate {' and '.join(key_symptoms)}",
                    "Obtain baseline vital signs once",
                    f"Focused examination of {' and '.join(key_symptoms)}",
                    f"Provide education on home management of {' and '.join(key_symptoms)}",
                    "Arrange appropriate follow-up care as needed"
                ]
        
        return {
            "esi_level": esi_level,
            "confidence": confidence,
            "justification": justification,
            "recommended_actions": actions
        }
    
    def _generate_discussion_summary(self, discussion_history):
        """Generate a summary of the discussion"""
        summary = []
        
        for entry in discussion_history:
            # Extract the first sentence or up to 100 characters
            content = entry["content"]
            first_sentence = content.split('.')[0] + '.'
            if len(first_sentence) > 100:
                first_sentence = first_sentence[:97] + '...'
            
            summary.append(f"{entry['role']}: {first_sentence}")
        
        return "\n".join(summary) 