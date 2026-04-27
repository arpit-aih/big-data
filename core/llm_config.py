import os
from typing import Optional
from dotenv import load_dotenv
from pandasai_openai import AzureOpenAI
from azure.identity import ClientSecretCredential
import logging


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_llm() -> Optional[AzureOpenAI]:
    
    try:
        load_dotenv()
        
        engine = os.getenv("Engine")
        tenant_id = os.getenv("Tenant_ID")
        client_id = os.getenv("Client_ID")
        client_secret = os.getenv("Secret_Value") 
        
        
        if not client_secret:
            client_secret = os.getenv("secret_key")
            
        azure_endpoint = os.getenv("End_point")
        api_version = os.getenv("API_version")

        if not all([engine, tenant_id, client_id, client_secret, azure_endpoint, api_version]):
            logger.error("Azure OpenAI configuration is incomplete. Check your environment variables.")
            return None

        credential = ClientSecretCredential(tenant_id, client_id, client_secret)

        
        llm = AzureOpenAI(
            azure_endpoint=azure_endpoint,
            api_version=api_version,
            deployment_name=engine,
            azure_ad_token=credential.get_token("https://cognitiveservices.azure.com/.default").token,
            max_tokens=None,
            temperature=0,
        )
        return llm
    except Exception as e:
        logger.error(f"Error initializing Azure OpenAI: {str(e)}")
        return None
