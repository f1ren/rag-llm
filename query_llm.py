# https://python.langchain.com/docs/integrations/llms/azure_openai

import os

from langchain_community.vectorstores.azuresearch import AzureSearch
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI

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

retriever = vector_store.as_retriever(
    search_type="similarity",  # Also test "similarity"
    # search_kwargs={"k": 8},
    k=8
)

from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationSummaryMemory
# from langchain_openai import ChatOpenAI

llm = AzureChatOpenAI(
    openai_api_version="2023-05-15",
    azure_deployment="gpt-35-turbo-instruct",
    openai_api_type="azure",
)


# memory = ConversationSummaryMemory(
#     llm=llm, memory_key="chat_history", return_messages=True
# )
# qa = ConversationalRetrievalChain.from_llm(llm, retriever=retriever, memory=memory)

# qa = ConversationalRetrievalChain.from_llm(llm, retriever=retriever)
condense_question_llm = AzureChatOpenAI(
    openai_api_version="2023-05-15", model="gpt-35-turbo-instruct", temperature=0)
qa = ConversationalRetrievalChain.from_llm(llm,
                                           retriever=retriever,
                                           return_source_documents=True,
                                           verbose=True,
                                           output_key='answer',
                                           # combine_docs_chain_kwargs={'prompt': prompt},
                                           condense_question_llm=condense_question_llm)

question = "How can I compare two instances of my own custom class?"
chat_history = []
result = qa.invoke({"question": question, "chat_history": chat_history})

print(result["answer"])
