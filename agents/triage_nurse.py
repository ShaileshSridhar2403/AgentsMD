import re
import json
import os
from utils.query_model import query_model

class TriageNurseAgent:
    def __init__(self, model="gpt-4o-mini", api_key=None):
        """
        Initialize the Triage Nurse Agent
        
        Args:
            model (str): LLM model to use
            api_key (str): API key for the LLM service
        """
        self.model = model
        self.api_key = api_key
        self.role = "Triage Nurse"
    
    def assess_conversation(self, conversation_text):
        """
        Perform initial assessment based on the conversation
        
        Args:
            conversation_text (str): The nurse-patient conversation
            
        Returns:
            dict: Initial assessment results
        """
        # Create a system prompt for the triage nurse role
        system_prompt = """
        You are an experienced emergency department triage nurse with over 15 years of experience.
        Your task is to perform an initial assessment of a patient based on the provided conversation.
        Focus on:
        1. Identifying immediate life threats
        2. Assessing vital signs stability (if mentioned)
        3. Determining the severity of the chief complaint
        4. Estimating resource needs
        5. Providing an initial ESI (Emergency Severity Index) level recommendation
        6. Recommending specific nursing interventions
        
        The ESI is a 5-level triage system:
        - Level 1: Requires immediate life-saving intervention
        - Level 2: High risk situation or severe pain/distress
        - Level 3: Requires multiple resources but stable vital signs
        - Level 4: Requires one resource
        - Level 5: Requires no resources
        
        IMPORTANT: Include at least 2-3 specific nursing interventions or actions that should be taken immediately.
        
        Provide your assessment in a structured format with clear reasoning.
        """
        
        # Create the user prompt
        user_prompt = f"""
        Please perform an initial triage assessment based on the following patient-nurse conversation:
        
        {conversation_text}
        
        Provide your assessment in the following format:
        1. Initial Impression:
        2. Chief Complaint (as you understand it):
        3. Concerning Findings:
        4. Estimated Resource Needs:
        5. Recommended ESI Level:
        6. Rationale:
        7. Immediate Nursing Interventions (list at least 2-3 specific actions):
        8. Additional Notes:
        """
        
        # Query the model
        response = query_model(
            model_str=self.model,
            system_prompt=system_prompt,
            prompt=user_prompt,
            openai_api_key=self.api_key
        )
        
        # Parse the response
        assessment = self._parse_assessment(response)
        
        return assessment
    
    def respond_to_assessments(self, conversation_text, assessments):
        """
        Respond to other agents' assessments
        
        Args:
            conversation_text (str): The nurse-patient conversation
            assessments (dict): Assessments from all agents
            
        Returns:
            str: Response to other assessments
        """
        # Create a system prompt for the response
        system_prompt = """
        You are an experienced emergency department triage nurse with over 15 years of experience.
        Your task is to review the assessments from other medical professionals and provide your perspective.
        You should:
        1. Identify any points you agree with
        2. Note any concerns or disagreements you have
        3. Provide additional insights from a triage nurse perspective
        4. Clarify or defend your ESI recommendation if needed
        
        Be professional but direct in your assessment. Your primary concern is patient safety and appropriate triage.
        """
        
        # Format the assessments for the prompt
        formatted_assessments = []
        for role, assessment in assessments.items():
            if role != self.role:  # Don't include own assessment
                formatted_assessments.append(f"{role} Assessment:")
                if isinstance(assessment, dict):
                    for key, value in assessment.items():
                        formatted_assessments.append(f"- {key}: {value}")
                else:
                    formatted_assessments.append(f"- {assessment}")
        
        formatted_assessments_text = "\n".join(formatted_assessments)
        
        # Create the user prompt
        user_prompt = f"""
        Please review the following assessments from other medical professionals regarding this patient conversation:
        
        {formatted_assessments_text}
        
        The original conversation was:
        
        {conversation_text}
        
        Provide your response to these assessments, noting agreements, disagreements, and additional insights from your perspective as a triage nurse.
        """
        
        # Query the model
        response = query_model(
            model_str=self.model,
            system_prompt=system_prompt,
            prompt=user_prompt,
            openai_api_key=self.api_key
        )
        
        return response
    
    def _parse_assessment(self, response):
        """Parse the LLM response into a structured assessment"""
        assessment = {
            "initial_impression": "",
            "chief_complaint": "",
            "concerning_findings": "",
            "resource_needs": "",
            "recommended_esi": "",
            "rationale": "",
            "notes": ""
        }
        
        # Extract sections using regex
        impression_match = re.search(r'1\.\s*Initial Impression:(.*?)(?=2\.|\Z)', response, re.DOTALL)
        if impression_match:
            assessment["initial_impression"] = impression_match.group(1).strip()
        
        complaint_match = re.search(r'2\.\s*Chief Complaint.*?:(.*?)(?=3\.|\Z)', response, re.DOTALL)
        if complaint_match:
            assessment["chief_complaint"] = complaint_match.group(1).strip()
        
        findings_match = re.search(r'3\.\s*Concerning Findings:(.*?)(?=4\.|\Z)', response, re.DOTALL)
        if findings_match:
            assessment["concerning_findings"] = findings_match.group(1).strip()
        
        resources_match = re.search(r'4\.\s*Estimated Resource Needs:(.*?)(?=5\.|\Z)', response, re.DOTALL)
        if resources_match:
            assessment["resource_needs"] = resources_match.group(1).strip()
        
        esi_match = re.search(r'5\.\s*Recommended ESI Level:(.*?)(?=6\.|\Z)', response, re.DOTALL)
        if esi_match:
            assessment["recommended_esi"] = esi_match.group(1).strip()
        
        rationale_match = re.search(r'6\.\s*Rationale:(.*?)(?=7\.|\Z)', response, re.DOTALL)
        if rationale_match:
            assessment["rationale"] = rationale_match.group(1).strip()
        
        notes_match = re.search(r'7\.\s*Additional Notes:(.*?)(?=\Z)', response, re.DOTALL)
        if notes_match:
            assessment["notes"] = notes_match.group(1).strip()
        
        return assessment 