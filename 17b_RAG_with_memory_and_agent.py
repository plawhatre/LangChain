import os
os.environ["USER_AGENT"] = "MyCustomAgent/1.0"

import faiss
from langchain_core.tools import tool
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import InMemorySaver

@tool
def retrieve(query: str):
    """Tool to retrieve documents that are relevant to the query"""
    docs = vectorstore.similarity_search(query, k=2)
    serialised = "\n\n\n".join(
        [f"Source: \n\t{doc.metadata}\n\n Content:\n\t{doc.page_content}"
        for doc in docs]
    )
    return serialised

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

    # Step 4: Model
    llm = ChatGoogleGenerativeAI(model='gemini-2.0-flash')

    # Step 5: Create agent
    memory = InMemorySaver()
    agent = create_react_agent(
        llm,
        tools=[retrieve],
        checkpointer=memory
    )

    # Step 6: Generate Output
    config = {"configurable": {"thread_id": '123abc'}}
    inp1 = {"messages": HumanMessage("Hello")}
    inp2 = {"messages": HumanMessage("What is Task Decomposition?")}
    inp3 = {"messages": HumanMessage("What are some common ways to do that?")}    

    for step in agent.stream(inp1, stream_mode='values', config=config):
        step["messages"][-1].pretty_print()

    for step in agent.stream(inp2, stream_mode='values', config=config):
        step["messages"][-1].pretty_print()
    
    for step in agent.stream(inp3, stream_mode='values', config=config):
        step["messages"][-1].pretty_print()