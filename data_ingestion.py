from openai import AzureOpenAI
from dotenv import load_dotenv
import os
from azure.identity import ClientSecretCredential
import time
from langchain_community.document_loaders.directory import DirectoryLoader
from langchain_text_splitters import CharacterTextSplitter,RecursiveCharacterTextSplitter
from langchain_community.document_loaders.unstructured import UnstructuredFileLoader
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from qdrant_client.models import PointStruct
from qdrant_client.http.exceptions import ResponseHandlingException
from openai.types.create_embedding_response import CreateEmbeddingResponse
from langchain_core.documents import Document

import logging
logging.basicConfig(
    level=logging.INFO, # Set the global threshold to INFO
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()] # Log to console
)
logger = logging.getLogger(__name__)

class DataIngestion:
    """This class used to ingest data reading from a source to a vector database."""
    def __init__(self, data_source: str = os.getenv('DOCUMENTS_PATH')):
        load_dotenv()
        self.data_source = data_source
        self.tenant_id = os.getenv('IQVIA_TENANT_ID')
        self.client_id = os.getenv('CLIENT_ID')
        self.client_secret = os.getenv('CLIENT_SECRET')
        self.ad_programmatic_scope = os.getenv('AD_PROGRAMMATIC_SCOPE')
        self.api_version = os.getenv('OPENAI_API_VERSION')
        self.endpoint = os.getenv('OPENAI_API_BASE')
        self.embedding_deployment_id = os.getenv('EMBEDDING_DEPLOYMENT_ID')
        self.client = ClientSecretCredential(self.tenant_id, self.client_id, self.client_secret)
        self.qdrant_host = os.getenv('QDRANT_HOST')
        self.qdrant_port = os.getenv('QDRANT_PORT')

    def token_provider(self):
        """This method used to get the token from the Azure AD."""
        try:
            return self.credential.get_token(self.ad_programmatic_scope).token
        except Exception as e:
            logger.error(f"Error getting token from Azure AD: {e}")
            return None

    def create_documents(self) -> list[Document]:
        """This method used to create documents from the data source (read all pdf files) and return documents"""
        try:
            loader = DirectoryLoader(path=self.data_source, glob="**/*.pdf",show_progress=True,loader_cls=UnstructuredFileLoader)
            documents = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=100)
            final_documents = text_splitter.split_documents(documents)
            return final_documents
        except Exception as e:
            logger.error(f"Error creating documents: {e}")
            return None

    def create_embeddings(self, documents) -> CreateEmbeddingResponse:
        """This method used to create vector embeddings from the documents and return embeddings size 1536"""
        try:
            client = AzureOpenAI(api_version=self.api_version, azure_endpoint=self.endpoint,azure_ad_token_provider=self.token_provider,azure_deployment="text-embedding-3-large")
            embeddings = client.embeddings.create(input=[doc.page_content for doc in documents], model="text-embedding-3-large")
            return embeddings
        except Exception as e:
            logger.error(f"Error creating embeddings: {e}")
            return None
    
    def create_qdrant_collection(self, collection_name: str, embeddings: CreateEmbeddingResponse,actual_documents: list[Document]) -> bool: 
        """This method used to create a qdrant collection and return the collection
        size of the vector is 1536 becz we are using text-embedding-3-large model for embeddings."""
        try:
            client = QdrantClient(host=self.qdrant_host,port=self.qdrant_port)
            client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=1536,
                    distance=Distance.COSINE,
                ),
            )
            points = [PointStruct(id=idx,vector=data.embedding,payload={"text": text})
                for idx, (data, text) in enumerate(zip(embeddings.data, actual_documents))
            ]
            client.upsert(collection_name, points)
            logger.info("documents created to qdrant vector database done.")
            return True
        except Exception as e:
            logger.error(f"Error creating qdrant collection: {e}")
            return False


if __name__ == "__main__":
    data_ingestion = DataIngestion(data_source=os.getenv('DOCUMENTS_PATH'))
    documents = data_ingestion.create_documents()
    embeddings = data_ingestion.create_embeddings(documents)
    result = data_ingestion.create_qdrant_collection(collection_name="safety_collection", embeddings=embeddings, actual_documents=documents)
    if result:
        logger.info("Data ingestion completed.")
    else:
        logger.error("Data ingestion failed.")
    



    