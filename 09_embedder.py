import os
from getpass import getpass
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings

if __name__ == '__main__':
    # Step 1: load the documents
    filename = 'attention_is_all_you_need.pdf'
    loader = PyPDFLoader(filename)
    documents = loader.load()

    # Step 2: split documents
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True
    )
    splits = splitter.split_documents(documents)

    # Step 3: Vector Embedding
    embedder1 = OllamaEmbeddings(model='mistral:v0.3')
    vector11 = embedder1.embed_query(splits[0].page_content)
    vector12 = embedder1.embed_query(splits[1].page_content)
    vector1 = embedder1.embed_documents([split.page_content for split in splits])

    if 'GOOGLE_API_KEY' not in os.environ:
        os.environ['GOOGLE_API_KEY'] = getpass("Enter the google API key: ")

    embedder2 = GoogleGenerativeAIEmbeddings(model='models/embedding-001', temperature=0.8)
    vector21 = embedder2.embed_query(splits[0].page_content)
    vector22 = embedder2.embed_query(splits[1].page_content)
    vector2 = embedder2.embed_documents([split.page_content for split in splits])

