import os
import json
from openai import OpenAI

def query_model(model_str, system_prompt, prompt, openai_api_key=None):
    """
    Query an LLM model with the given prompts using the OpenAI API v1.0+
    
    Args:
        model_str (str): The model to use
        system_prompt (str): The system prompt
        prompt (str): The user prompt
        openai_api_key (str, optional): API key for OpenAI
        
    Returns:
        str: The model's response
    """
    # Set up the client
    if openai_api_key:
        client = OpenAI(api_key=openai_api_key)
    elif "OPENAI_API_KEY" in os.environ:
        client = OpenAI()  # Uses OPENAI_API_KEY from environment
    else:
        raise ValueError("No OpenAI API key provided")
    
    try:
        # Create the chat completion with the new API
        response = client.chat.completions.create(
            model=model_str,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,  # Lower temperature for more deterministic responses
            max_tokens=2000
        )
        
        # Extract the content from the response
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error querying model: {e}")
        return f"Error: {str(e)}"