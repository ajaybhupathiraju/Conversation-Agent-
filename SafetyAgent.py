import os
from DocumentRetrival import DocumentRetrieval
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI,AzureChatOpenAI
from langchain_core.messages import HumanMessage,SystemMessage,AnyMessage,AIMessage
from langgraph.graph import StateGraph,START,END
from typing import TypedDict
from typing_extensions import Annotated,Literal
from langgraph.graph.message import add_messages
from azure.identity import ClientSecretCredential
import logging
logging.basicConfig(
    level=logging.INFO, # Set the global threshold to INFO
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()] # Log to console
)
logger = logging.getLogger(__name__)

class SafetyAgent:
    """This class is used to create a safety agent."""
    def __init__(self):
        load_dotenv()
        try:
            self.tenant_id = os.getenv("IQVIA_TENANT_ID")
            self.client_id = os.getenv("CLIENT_ID")
            self.client_secret = os.getenv("CLIENT_SECRET")
            self.ad_programatic_scope = os.getenv("AD_PROGRAMMATIC_SCOPE")
            self.endpoint = os.getenv("BASE_OPENAI_API_ENDPOINT")
            self.azure_deployment_name = os.getenv("AZURE_CHAT_OPENAI_DEPLOYMENT_NAME")
            self.azure_api_version = os.getenv("AZURE_CHAT_OPENAI_API_VERSION")
        except Exception as e:
            logger.error(f"Error initializing Safety Agent: {e}")
            return e 

    def get_token(self):
        client = ClientSecretCredential(
                    tenant_id=self.tenant_id,
                    client_id=self.client_id,
                    client_secret=self.client_secret
                )
        token = client.get_token(self.ad_programatic_scope)
        return token.token   

    def get_llm(self):
        return AzureChatOpenAI(
                azure_deployment=self.azure_deployment_name,
                api_version=self.azure_api_version,
                azure_endpoint=self.endpoint,
                azure_ad_token_provider=self.get_token,
                temperature=0
        )

    def agent_response(self, study_name: str, drug_name: str,user_query: str):
        """This method is used to get the llm response."""

        document_context = DocumentRetrieval().get_documents(study_name, drug_name, user_query)
        if document_context is not None:
            system_prompt = f"""
                You are an expert Pharmacovigilance Safety Specialist and Medical Analyst.
                You need to respond to the user query based on the relevent context provided.
                
                Here is the context : {document_context} \n\n 
                study name / protocol name : {study_name} \n\n 
                drug name : {drug_name}"""
        
            try:
                llm = self.get_llm()
                response = llm.invoke(system_prompt)
                return response.content
            except Exception as e:
                logger.error(f"Error answering the user query: {e}")
                return e
    
    

            


