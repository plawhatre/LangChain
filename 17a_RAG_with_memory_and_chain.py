import os
os.environ["USER_AGENT"] = "MyCustomAgent/1.0"

import faiss
from langchain_core.tools import tool
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage

from langgraph.graph import START, StateGraph, MessagesState, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import InMemorySaver


# Node Functions
@tool
def retrieve(query: str):
    """Tool to retrieve documents that are relevant to the query"""
    docs = vectorstore.similarity_search(query, k=2)
    serialised = "\n\n\n".join(
        [f"Source: \n\t{doc.metadata}\n\n Content:\n\t{doc.page_content}"
        for doc in docs]
    )
    return serialised

def query_or_respond(state: MessagesState):
    llm_with_tools = llm.bind_tools([retrieve])
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}

def generate(state: MessagesState):
    # Messages: tools
    tool_messages = []
    for msg in reversed(state['messages']):
        if msg.type == 'tool':
            tool_messages.append(msg)
        else:
            break
    tool_messages = tool_messages[::-1]
    docs_content = "\n\n".join([doc.content for doc in tool_messages])

    # Messages: system, conversation history
    system_message = SystemMessage(
        "You are a helpful assistant that answers question"
        "Use the following piece of retrieved context to answer"
        "If you don't know the answer then you say so."
        "Keep the answer concise. Max three sentences"
        "\n\n"
        f"{docs_content}"
    )
    conversation_history_without_tools = [
        msg
        for msg in state['messages']
        if msg.type in ("human", "system")
        or (msg.type == 'ai' and not msg.tool_calls)
    ]

    # Prompt for LLM
    prompt = [system_message] + conversation_history_without_tools
    response = llm.invoke(prompt)
    return {"messages": [response]}

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

    # Step 5: Define graph 
    graph_builder = StateGraph(MessagesState)

    graph_builder.add_node(query_or_respond)
    graph_builder.add_node("tools", ToolNode([retrieve]))
    graph_builder.add_node(generate)

    graph_builder.set_entry_point("query_or_respond")
    graph_builder.add_conditional_edges(
        "query_or_respond",
        tools_condition,
        {   
            "tools": "tools",
            END: END
        }
    )
    graph_builder.add_edge("tools", "generate")
    graph_builder.add_edge("generate", END)

    graph = graph_builder.compile(checkpointer=InMemorySaver())

    # Step 6: Generate Output
    config = {"configurable": {"thread_id": '123abc'}}
    inp1 = {"messages": HumanMessage("Hello")}
    inp2 = {"messages": HumanMessage("What is Task Decomposition?")}
    inp3 = {"messages": HumanMessage("What are some common ways to do that?")}    

    for step in graph.stream(inp1, stream_mode='values', config=config):
        step["messages"][-1].pretty_print()

    for step in graph.stream(inp2, stream_mode='values', config=config):
        step["messages"][-1].pretty_print()
    
    for step in graph.stream(inp3, stream_mode='values', config=config):
        step["messages"][-1].pretty_print()