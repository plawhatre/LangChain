import faiss
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain.vectorstores import FAISS


if __name__ == "__main__":

    filename = "attention_is_all_you_need.pdf"
    query ="What is self attention?"
    vector_store_options = {
        "option_1": 'InMemoryVectorStore',
        "option_2": 'FAISS',
    }
    choice = vector_store_options.get(
        f"option_{str(input("Press \n 1. for InMemoryVectorStore\n 2. for FAISS\n"))}"
    )
    if choice == 'FAISS':
        similarity_metric = {
            "inner_product": 'IndexFlatIP',
            "eucledian": 'IndexFlatL2'
        }.get(
            f"option_{str(input("Press \n 1. for Inner Product\n 2. for Eucledian\n"))}"
        )

    # Step1: Load the document
    loader = PyPDFLoader(filename)
    documents = loader.load()

    # Step2: Split the document
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True
    )
    chunks = text_splitter.split_documents(documents)
    
    # Step3: Add the embedding model
    embedder = OllamaEmbeddings(model='mistral:v0.3')

    # Step4: Add the document to vector store
    if choice == 'InMemoryVectorStore':
        vector_store = InMemoryVectorStore(embedding=embedder)
    else:
        # define index
        if similarity_metric == 'IndexFlatL2':
            index = faiss.IndexFlatL2(
                len(embedder.embed_query("placeholder"))
            )
        else:
            index = faiss.IndexFlatIP(
                len(embedder.embed_query("placeholder"))
            )
        vector_store = FAISS(
            docstore=InMemoryDocstore(),
            embedding_function=embedder,
            index=index,
            index_to_docstore_id={}
        )
    # Step4.a: Add the documents
    ids = vector_store.add_documents(chunks) 
    # Step4.b: Delete the documents
    vector_store.delete(ids=ids[-2:]) 
    # Step4.c: Similarity search
    similar_docs = vector_store.similarity_search(query, k=3) 
    print(similar_docs)