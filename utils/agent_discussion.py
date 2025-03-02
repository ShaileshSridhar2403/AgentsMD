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
    
    def deliberate(self, conversation_text, case_id):
        """
        Conduct a deliberation between agents to determine ESI level
        
        Args:
            conversation_text (str): The raw nurse-patient conversation
            case_id (str): Unique case identifier
            
        Returns:
            dict: Discussion result with ESI level and justification
        """
        # Initialize discussion
        discussion_history = []
        
        # Round 1: Initial assessments from each agent
        print("Round 1: Initial assessments from each agent...")
        initial_assessments = {}
        for agent in self.agents:
            print(f"  - {agent.role} is assessing the conversation...")
            assessment = agent.assess_conversation(conversation_text)
            initial_assessments[agent.role] = assessment
            
            # Add to discussion history
            discussion_history.append({
                "role": agent.role,
                "content": f"Initial assessment: {self._summarize_assessment(assessment)}"
            })
        
        # Round 2: Agents respond to each other's assessments
        print("Round 2: Agents responding to each other's assessments...")
        for agent in self.agents:
            print(f"  - {agent.role} is responding to other assessments...")
            response = agent.respond_to_assessments(conversation_text, initial_assessments)
            
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
        """Summarize an agent's assessment"""
        if "recommended_esi" in assessment:
            esi = assessment.get("recommended_esi", "Unknown")
            rationale = assessment.get("rationale", "No rationale provided")
            return f"ESI Level: {esi}. Rationale: {rationale}"
        elif "esi_level" in assessment:
            esi = assessment.get("esi_level", "Unknown")
            clinical = assessment.get("clinical_assessment", "No assessment provided")
            return f"ESI Level: {esi}. Assessment: {clinical}"
        else:
            return "No clear ESI recommendation"
    
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
        prompt.append("3. Clinical justification for this ESI level")
        prompt.append("4. Recommended immediate actions (provide at least 3-5 specific actions)")
        prompt.append("")
        prompt.append("IMPORTANT: Your recommended actions must be specific, practical steps that the emergency department staff should take. Do not leave this section empty or vague.")
        
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
        
        IMPORTANT: You MUST provide at least 3-5 specific recommended actions for the patient based on their ESI level.
        These actions should be concrete, practical steps that the emergency department staff should take.
        
        For ESI Level 1-2 patients, include immediate interventions.
        For ESI Level 3 patients, include diagnostic and treatment recommendations.
        For ESI Level 4-5 patients, include appropriate care instructions.
        
        Provide your final determination in a structured format with clear reasoning.
        Be decisive and consider all perspectives from the discussion.
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
                      and len(a) < 150]
        
        # If still no actions, generate default actions based on ESI level
        if not actions:
            if esi_level == "1":
                actions = [
                    "Immediate intervention by emergency physician",
                    "Prepare resuscitation equipment",
                    "Establish IV access immediately",
                    "Continuous vital sign monitoring",
                    "Prepare for possible intubation/resuscitation"
                ]
            elif esi_level == "2":
                actions = [
                    "Urgent assessment by emergency physician within 10 minutes",
                    "Establish IV access",
                    "Continuous vital sign monitoring",
                    "Administer appropriate pain medication if needed",
                    "Prepare for diagnostic studies"
                ]
            elif esi_level == "3":
                actions = [
                    "Assessment by emergency physician within 30 minutes",
                    "Obtain baseline vital signs",
                    "Order appropriate diagnostic tests",
                    "Establish IV access if needed",
                    "Reassess patient condition regularly"
                ]
            elif esi_level == "4":
                actions = [
                    "Assessment by provider within 60 minutes",
                    "Obtain baseline vital signs",
                    "Focused examination of affected area",
                    "Consider appropriate imaging or simple tests",
                    "Provide symptomatic relief as needed"
                ]
            else:  # ESI level 5
                actions = [
                    "Assessment by provider when available",
                    "Obtain baseline vital signs",
                    "Focused examination of affected area",
                    "Provide education and home care instructions",
                    "Arrange follow-up care as needed"
                ]
        
        return {
            "esi_level": esi_level,
            "confidence": confidence,
            "justification": justification,
            "recommended_actions": actions
        }
    
    def _generate_discussion_summary(self, discussion_history):
        """Generate a summary of the agent discussion"""
        summary_prompt = "Summarize the following discussion between medical professionals about a patient's ESI level:\n\n"
        
        for entry in discussion_history:
            summary_prompt += f"{entry['role']}: {entry['content']}\n\n"
        
        summary_prompt += "Provide a concise summary of the key points and areas of agreement/disagreement."
        
        summary = query_model(
            model_str=self.model,
            system_prompt="You are a medical scribe who creates concise summaries of clinical discussions.",
            prompt=summary_prompt,
            openai_api_key=self.api_key
        )
        
        return summary 