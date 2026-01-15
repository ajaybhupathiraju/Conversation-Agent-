from openai import AzureOpenAI
from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI,AzureChatOpenAI
from azure.identity import ClientSecretCredential
import time
from langchain_community.document_loaders.directory import DirectoryLoader
from langchain_text_splitters import CharacterTextSplitter,RecursiveCharacterTextSplitter
from langchain_community.document_loaders.unstructured import UnstructuredFileLoader
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from qdrant_client.models import PointStruct
from qdrant_client.http.exceptions import ResponseHandlingException

import logging
logging.basicConfig(
    level=logging.INFO, # Set the global threshold to INFO
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()] # Log to console
)
logger = logging.getLogger(__name__)

class DocumentRetrieval:
    """ This class is used to retrieve documents from the qdrant vector database """
    def __init__(self):
       logger.info("Initializing DocumentRetrieval constructor")
       load_dotenv()
       try:
           self.tenant_id = os.getenv('IQVIA_TENANT_ID')
           self.client_id = os.getenv('CLIENT_ID')
           self.client_secret = os.getenv('CLIENT_SECRET')
           self.ad_programmatic_scope = os.getenv('AD_PROGRAMMATIC_SCOPE')
           self.api_version = os.getenv('OPENAI_API_VERSION')
           self.endpoint = os.getenv('OPENAI_API_BASE')
           self.embedding_deployment_id = os.getenv('EMBEDDING_DEPLOYMENT_ID')
           self.collection_name = os.getenv('DOCUMENT_COLLECTION_NAME')
           self.qdrant_host = os.getenv('QDRANT_HOST')
           self.qdrant_port = os.getenv('QDRANT_PORT')
       except Exception as e:
            logger.error(f"Error initializing DocumentRetrieval: {e}")
            return e

    def token_provider(self):
        """Function to provide the token for authentication."""
        credential = ClientSecretCredential(tenant_id=self.tenant_id,client_id=self.client_id,client_secret=self.client_secret)
        return credential.get_token(self.ad_programmatic_scope).token


    def get_llm(self):
        return AzureOpenAI( # Use AzureOpenAI client for embeddings API calls
                api_version=self.api_version,
                azure_endpoint=self.endpoint,
                azure_ad_token_provider=self.token_provider,
                azure_deployment="text-embedding-3-large", # Use embedding deployment id
            )

    def get_documents(self, study_name: str, drug_name: str,query_text: str)-> str:
        """Get the documents from the qdrant vector database"""
        print("get documents invoked...")
        try:
            model = self.get_llm()
            client = QdrantClient(host=os.getenv('QDRANT_HOST'),port=os.getenv('QDRANT_PORT'))
            embed = model.embeddings.create(input=f"Study :{study_name} and drug :{drug_name} and user query is :{query_text}",model="text-embedding-3-large")
            points = client.query_points(collection_name=self.collection_name,query=embed.data[0].embedding,limit=5)
            documents = []
            for point in points.points:
                text = point.payload['text']['page_content'].strip()
                documents.append(text)
            logger.info(f"No of documents retrieved for study / protocol id :{study_name} and drug  :{drug_name} is :{len(documents)}")
            return "\n".join(documents)
            return documents 
        except ResponseHandlingException as e:
               logger.error(f"ResponseHandlingException Qdrant Client : {e}")
               return None
        except Exception as e:
            print(type(e))
            logger.error(f"Exception :{e}")
            return None
        
if __name__ == "__main__":
    document_retrieval = DocumentRetrieval()
    documents = document_retrieval.get_documents(study_name="CNIS793B12301", drug_name="NAB PACLITAXEL",query_text="Can you explain information aboutNAB PACLITAXEL?")
    if documents is not None:
       print(documents)
    else:
        print("No documents found")