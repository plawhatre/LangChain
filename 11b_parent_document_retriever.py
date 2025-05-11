import httpx
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain.storage import InMemoryStore
from langchain.retrievers import ParentDocumentRetriever

if __name__ == "__main__":
    # Constants
    url1 = "https://arxiv.org/pdf/1706.03762"
    url2 = "https://arxiv.org/pdf/1810.04805"
    filename1 = "attention_is_all_you_need.pdf"
    filename2 = "bert.pdf"

    # Step0: Get the data
    if not os.path.isfile(filename1):
        content1 = httpx.get(url1).content
        with open(filename1, "wb") as file:
            file.write(content1)

    if not os.path.isfile(filename2):
        content2 = httpx.get(url2).content
        with open("bert.pdf", "wb") as file:
            file.write(content2)

    # Step1: Load the documents
    docs = []
    docs1 = PyPDFLoader(filename1).load()
    docs2 = PyPDFLoader(filename2).load()
    docs.extend(docs1)
    docs.extend(docs2)

    # Step2: Splitter for child documents
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=200,
        add_start_index=False
    )

    # Step3: Embedder
    embedder = OllamaEmbeddings(model="mistral:v0.3")

    # Step4: Vector Store
    vectorstore = Chroma(
        collection_name="full_documents",
        embedding_function=embedder
    )

    # Step5: Storage for parent documents
    docstore = InMemoryStore()

    # Step6: Retriever
    retriever = ParentDocumentRetriever(
        child_splitter=splitter,
        vectorstore=vectorstore,
        docstore=docstore
    )
    retriever.add_documents(docs, ids=None)

    # Query
    query = "what is Masked Language Modeling?"
    sub_doc = vectorstore.similarity_search(query)
    print("\nsub_doc[0].page_content\n", sub_doc[0].page_content)
    retrieved_doc = retriever.invoke(query)
    print("\nretrieved_doc[0].page_content\n", retrieved_doc[0].page_content)

