from langchain_google_vertexai import VertexAIEmbeddings
from google.cloud.aiplatform import telemetry
import logging

# Configure the logging module
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Constants
USER_AGENT = 'cloud-solutions/genai-for-developers-v1'

# Create Vertex AI embeddings instance
embeddings = VertexAIEmbeddings(model_name='textembedding-gecko@001')


# Use telemetry's context manager if required
with telemetry.tool_context_manager(USER_AGENT):
    text = "This is a test document."
    query_result = embeddings.embed_query(text)
    doc_result = embeddings.embed_documents([text])

    # Log results for demonstration purposes
    logging.info('Query Embedding Result: %s', query_result[0] if query_result else 'No result')
    logging.info('Document Embedding Result: %s', doc_result[0] if doc_result else 'No result')
