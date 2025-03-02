import re
from utils.query_model import query_model

class EmergencyPhysicianAgent:
    def __init__(self, model="gpt-4o-mini", api_key=None):
        """
        Initialize the Emergency Physician Agent
        
        Args:
            model (str): LLM model to use
            api_key (str): API key for the LLM service
        """
        self.model = model
        self.api_key = api_key
        self.role = "Emergency Physician"
    
    def assess_conversation(self, conversation_text):
        """
        Perform detailed clinical assessment based on the conversation
        
        Args:
            conversation_text (str): The nurse-patient conversation
            
        Returns:
            dict: Detailed clinical assessment
        """
        # Create a system prompt for the emergency physician role
        system_prompt = """
        You are an experienced emergency medicine physician with over 20 years of experience.
        Your task is to perform a detailed clinical assessment of a patient based on the provided conversation.
        Focus on:
        1. Identifying potential diagnoses and their likelihood
        2. Determining the appropriate ESI (Emergency Severity Index) level
        3. Recommending immediate diagnostic tests and interventions
        4. Assessing risk factors and potential complications
        5. Determining disposition (admission, discharge, observation)
        
        The ESI is a 5-level triage system:
        - Level 1: Requires immediate life-saving intervention
        - Level 2: High risk situation or severe pain/distress
        - Level 3: Requires multiple resources but stable vital signs
        - Level 4: Requires one resource
        - Level 5: Requires no resources
        
        IMPORTANT: Provide at least 3-5 specific medical interventions, diagnostic tests, or treatments that should be initiated.
        Be concrete and practical in your recommendations.
        
        Provide your assessment in a structured format with clear medical reasoning.
        """
        
        # Create the user prompt
        user_prompt = f"""
        Please perform a detailed emergency physician assessment based on the following patient-nurse conversation:
        
        {conversation_text}
        
        Provide your assessment in the following format:
        1. Clinical Assessment:
        2. Potential Diagnoses (in order of likelihood):
        3. ESI Level Recommendation:
        4. Immediate Actions/Interventions (list at least 3 specific actions):
        5. Diagnostic Studies (be specific):
        6. Risk Assessment:
        7. Disposition Recommendation:
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
        You are an experienced emergency medicine physician with over 20 years of experience.
        Your task is to review the assessments from other medical professionals and provide your perspective.
        You should:
        1. Identify any points you agree with
        2. Note any concerns or disagreements you have
        3. Provide additional insights from an emergency physician perspective
        4. Clarify or defend your ESI recommendation if needed
        
        Be professional but direct in your assessment. Your primary concern is accurate diagnosis and appropriate treatment.
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
        
        Provide your response to these assessments, noting agreements, disagreements, and additional insights from your perspective as an emergency physician.
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
            "clinical_assessment": "",
            "potential_diagnoses": [],
            "esi_level": "",
            "immediate_actions": [],
            "diagnostic_studies": [],
            "risk_assessment": "",
            "disposition": ""
        }
        
        # Extract sections using regex
        clinical_match = re.search(r'1\.\s*Clinical Assessment:(.*?)(?=2\.|\Z)', response, re.DOTALL)
        if clinical_match:
            assessment["clinical_assessment"] = clinical_match.group(1).strip()
        
        diagnoses_match = re.search(r'2\.\s*Potential Diagnoses.*?:(.*?)(?=3\.|\Z)', response, re.DOTALL)
        if diagnoses_match:
            diagnoses_text = diagnoses_match.group(1).strip()
            # Extract diagnoses as a list
            diagnoses_list = re.findall(r'(?:^|\n)\s*(?:-|\d+\.)\s*(.*?)(?=\n\s*(?:-|\d+\.)|\Z)', diagnoses_text, re.DOTALL)
            assessment["potential_diagnoses"] = [d.strip() for d in diagnoses_list if d.strip()]
        
        esi_match = re.search(r'3\.\s*ESI Level.*?:(.*?)(?=4\.|\Z)', response, re.DOTALL)
        if esi_match:
            assessment["esi_level"] = esi_match.group(1).strip()
        
        actions_match = re.search(r'4\.\s*Immediate Actions.*?:(.*?)(?=5\.|\Z)', response, re.DOTALL)
        if actions_match:
            actions_text = actions_match.group(1).strip()
            # Extract actions as a list
            actions_list = re.findall(r'(?:^|\n)\s*(?:-|\d+\.)\s*(.*?)(?=\n\s*(?:-|\d+\.)|\Z)', actions_text, re.DOTALL)
            assessment["immediate_actions"] = [a.strip() for a in actions_list if a.strip()]
        
        studies_match = re.search(r'5\.\s*Diagnostic Studies.*?:(.*?)(?=6\.|\Z)', response, re.DOTALL)
        if studies_match:
            studies_text = studies_match.group(1).strip()
            # Extract studies as a list
            studies_list = re.findall(r'(?:^|\n)\s*(?:-|\d+\.)\s*(.*?)(?=\n\s*(?:-|\d+\.)|\Z)', studies_text, re.DOTALL)
            assessment["diagnostic_studies"] = [s.strip() for s in studies_list if s.strip()]
        
        risk_match = re.search(r'6\.\s*Risk Assessment:(.*?)(?=7\.|\Z)', response, re.DOTALL)
        if risk_match:
            assessment["risk_assessment"] = risk_match.group(1).strip()
        
        disposition_match = re.search(r'7\.\s*Disposition.*?:(.*?)(?=\Z)', response, re.DOTALL)
        if disposition_match:
            assessment["disposition"] = disposition_match.group(1).strip()
        
        return assessment 