import os  # For environment variable management
from ibm_watsonx_ai.foundation_models.utils.enums import ModelTypes  # For specifying model types
from ibm_watsonx_ai import APIClient, Credentials  # For API client and credentials management
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams  # For managing model parameters
from ibm_watsonx_ai.foundation_models.utils.enums import DecodingMethods  # For defining decoding methods
from langchain_ibm import WatsonxLLM, WatsonxEmbeddings  # For interacting with IBM's LLM and embeddings
from ibm_watsonx_ai.foundation_models.utils import get_embedding_model_specs  # For retrieving model specifications
from ibm_watsonx_ai.foundation_models.utils.enums import EmbeddingTypes  # For specifying types of embeddings

# Set up the LLM model
def setup_credentials():
    model_id = "meta-llama/llama-3-2-3b-instruct"

    url = os.getenv("IBM_WATSONX_URL", "https://us-south.ml.cloud.ibm.com")
    api_key = os.getenv("IBM_API_KEY")

    if not api_key:
        raise RuntimeError("IBM_API_KEY is not set in your environment.")

    # For watsonx.ai on IBM Cloud, project_id is effectively REQUIRED.
    # If you omit it, most foundation-model calls will fail with
    # 'Project ID cannot be empty' or 'Missing header x-watsonx-project-id'.
    project_id = os.getenv("IBM_PROJECT_ID")
    if not project_id:
        raise RuntimeError(
            "IBM_PROJECT_ID is not set. For watsonx.ai, project_id is required."
        )

    # SDK client uses the Credentials object (expects keyword args, not a dict)
    creds_obj = Credentials(url=url, api_key=api_key)
    client = APIClient(creds_obj)
    
    # LangChain wrappers just need a simple dict
    credentials = {"url": url, "apikey": api_key}
    return model_id, credentials, client, project_id

def define_parameters():
    # Return a dictionary containing the parameters for the WatsonX model
    return {
        # Set the decoding method to GREEDY for generating text
        GenParams.DECODING_METHOD: DecodingMethods.GREEDY,
        
        # Specify the maximum number of new tokens to generate
        GenParams.MAX_NEW_TOKENS: 900,
    }

def initialize_watsonx_llm(model_id, credentials, project_id, parameters):
    # Create and return an instance of the WatsonxLLM with the specified configuration
    return WatsonxLLM(
        model_id=model_id,          # Set the model ID for the LLM
        url=credentials.get("url"), # Retrieve the service URL from credentials
        apikey=credentials.get("apikey"), # Retrieve the API key from credentials
        project_id=project_id,            # Set the project ID for accessing resources
        params=parameters                  # Pass the parameters for model behavior
    )

# Setup the embedding model used to embed transcript chunks
def setup_embedding_model(credentials, project_id):
    # Create and return an instance of WatsonxEmbeddings with the specified configuration
    return WatsonxEmbeddings(
        model_id=EmbeddingTypes.IBM_SLATE_30M_ENG.value,  # Set the model ID for the SLATE-30M embedding model
        url=credentials["url"],
        apikey=credentials["apikey"],
        # Retrieve the service URL from the provided credentials
        project_id=project_id                               # Set the project ID for accessing resources in the Watson environment
    )