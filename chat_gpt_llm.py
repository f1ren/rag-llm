# from git import Repo
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
len(documents)

from langchain.text_splitter import RecursiveCharacterTextSplitter

python_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON, chunk_size=2000, chunk_overlap=200
)
texts = python_splitter.split_documents(documents)

from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

db = Chroma.from_documents(texts, OpenAIEmbeddings(disallowed_special=()))
retriever = db.as_retriever(
    search_type="mmr",  # Also test "similarity"
    search_kwargs={"k": 8},
)