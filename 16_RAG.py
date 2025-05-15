import os
os.environ["USER_AGENT"] = "MyCustomAgent/1.0"

import faiss
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from typing import TypedDict, List, Dict
from langchain_core.documents import Document
from langchain import hub

from langgraph.graph import START, StateGraph


# States and Node functions
class State(TypedDict):
    question: str
    context: List[Document]
    answer: str

def retriever(state: State):
    result = vectorstore.similarity_search(state['question'])
    return {'context': result}

def generate(state: State):
    context = "\n\n".join([docs.page_content for docs in state['context']])
    message = prompt.invoke({
        "question": state['question'],
        "context": context,
    })
    response = llm.invoke(message)
    return {"answer": response.content}

if __name__ == "__main__":
    # constants
    web_path = "https://lilianweng.github.io/posts/2023-06-23-agent/"

    # Step 1: Load documents
    doc_loader = WebBaseLoader(
        web_path=web_path
    )
    docs = doc_loader.load()
    
    
    # Step 2: chunking
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    chunks = splitter.split_documents(docs)

    # Step 3: Store (docstore, embedder, index, vectorstore)
    docstore = InMemoryDocstore()
    embedding_function = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    index = faiss.IndexFlatL2(len(embedding_function.embed_query("placeholder")))
    vectorstore = FAISS(
        docstore=docstore,
        embedding_function=embedding_function,
        index=index,
        index_to_docstore_id={}
    )
    vectorstore.add_documents(chunks)

    # Step 4: Model & prompt
    llm = ChatGoogleGenerativeAI(model='gemini-1.5-flash')
    prompt = hub.pull("rlm/rag-prompt")

    # Step 5: Define graph 
    graph_builder = StateGraph(State)
    graph_builder.add_node("retrieve", retriever)
    graph_builder.add_node("generate", generate)
    graph_builder.add_edge(START, "retrieve")
    graph_builder.add_edge("retrieve", "generate")
    graph = graph_builder.compile()

    # Step 6: Generate Output
    output = graph.invoke({
        "question": "What is Task decomposition?"
    })
    print(output['answer'])