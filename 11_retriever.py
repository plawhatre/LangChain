from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from langchain_core.runnables import chain

if __name__ == "__main__":
    filename = 'attention_is_all_you_need.pdf'
    # Step 1: Load the document
    docs = PyPDFLoader(filename).load()
    # Step 2: chunk the data
    chunks = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True
    ).split_documents(docs)
    # Step 3: Add docs to vector store
    vector_store = FAISS.from_documents(documents=chunks, embedding=OllamaEmbeddings(model="mistral:v0.3"))

    # Step 4.1: Retriever (chain decorator)
    @chain
    def retriever1(query):
        return vector_store.similarity_search(query, k=1)

    # Step 4.2: Retriever (as_retriever function)
    retriever2 = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={'k': 1}
    )

    output1 = retriever1.batch([
        "What is masked attention?",
        "What is self attention?"
    ])
    print("chain decorator: \n", output1)

    output2 = retriever2.batch([
        "What is masked attention?",
        "What is self attention?"
    ])
    print("as_retriever func: \n", output2)
