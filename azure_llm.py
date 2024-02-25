from langchain.text_splitter import Language
from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers import LanguageParser

repo_path = "../commons-lang/"
SUB_PATH = "src/main/java/org/apache/commons/lang3"

# Load
loader = GenericLoader.from_filesystem(
    repo_path + SUB_PATH,
    glob="**/*",
    suffixes=[".java"],
    exclude=["**/non-utf8-encoding.py"],
    parser=LanguageParser(language=Language.JAVA, parser_threshold=500),
)
documents = loader.load()

from langchain.text_splitter import RecursiveCharacterTextSplitter

python_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON, chunk_size=2000, chunk_overlap=200
)
texts = python_splitter.split_documents(documents)

import os

from langchain_community.vectorstores.azuresearch import AzureSearch
from langchain_openai import AzureOpenAIEmbeddings

embeddings: AzureOpenAIEmbeddings = AzureOpenAIEmbeddings(
    # azure_deployment="text-embedding-ada-002",
    azure_deployment="text-embedding-3-large",
    openai_api_version="2023-05-15",
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    api_key=os.environ["AZURE_OPENAI_KEY"],
)

vector_store_address: str = os.environ["AZURE_SEARCH_ENDPOINT"]
vector_store_password: str = os.environ["AZURE_SEARCH_ADMIN_KEY"]

index_name: str = "langchain-vector-demo"
vector_store: AzureSearch = AzureSearch(
    azure_search_endpoint=vector_store_address,
    azure_search_key=vector_store_password,
    index_name=index_name,
    embedding_function=embeddings.embed_query,
)

vector_store.add_documents(documents=texts)