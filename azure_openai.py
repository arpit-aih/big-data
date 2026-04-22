from typing import Any, Dict, List
from openai import AzureOpenAI
from aih_automaton.ai_models.model_base import AIModel
from aih_automaton.data_models import FileResponse
from aih_automaton.utils.resource_handler import ResourceBox
import os
from dotenv import load_dotenv
from azure.identity import ClientSecretCredential


load_dotenv()

class AzureOpenAIModel(AIModel):
    def __init__(
        self,
        azure_api_key=None,
        azure_api_version=None,
        parameters: Dict[str, Any] = None,
        azure_endpoint: str = None,
    ):
        self.parameters = parameters
        azure_api_version = azure_api_version or os.getenv("API_version")
        azure_endpoint = azure_endpoint or os.getenv("End_point")
        engine = os.getenv("Engine", "nexa1-gpt-4.1-global-std")

        tenant_id = os.getenv("tenant_id")
        client_id = os.getenv("client_id")
        client_secret = os.getenv("Secret_Value") or os.getenv("Secret_ID")

        if tenant_id and client_id and client_secret:
            
            credential = ClientSecretCredential(tenant_id, client_id, client_secret)
            self.client = AzureOpenAI(
                azure_endpoint=azure_endpoint,
                api_version=azure_api_version,
                azure_ad_token=credential.get_token("https://cognitiveservices.azure.com/.default").token
            )
        else:
            self.client = AzureOpenAI(
                api_key=azure_api_key or os.getenv("AZURE_OPENAI_API_KEY"),
                api_version=azure_api_version,
                azure_endpoint=azure_endpoint
            )
        
        self.api_key = azure_api_key
        self.model = engine

    def generate_text(
        self,
        task_id: str=None,
        system_persona: str=None,
        prompt: str=None,
        messages: List[dict] = None,
        functions: List[Dict[str, Any]] = None,
        function_call: str = None,
        model=None,
    ):
        
        if messages is None:
            messages = [
                {"role": "system", "content": system_persona},
                {"role": "user", "content": prompt},
            ]

        response = self.client.chat.completions.create(
             messages=messages,
             functions=functions,
             function_call=function_call,
             model=model or self.model or 'gpt-5.1'
        )
        if function_call:
            return response.choices[0].message.function_call
        
        return response.choices[0].message.content

    def generate_image(
        self, task_id: str, prompt: str, resource_box: ResourceBox
    ) -> FileResponse:
        response = self.client.images.generate(**self.parameters, prompt=prompt)
        return resource_box.save_from_url(url=response.data[0].url, subfolder=task_id)