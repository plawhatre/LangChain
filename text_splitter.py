import os
import httpx
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter, CharacterTextSplitter, TokenTextSplitter

if __name__ == '__main__':

    # Step1: Load the documents
    url = "https://arxiv.org/pdf/1706.03762"
    filename = "attention_is_all_you_need.pdf"

    if not os.path.isfile(filename):
        content = httpx.get(url).content
        with open(filename, 'wb') as file:
            file.write(content)

    documents = PyPDFLoader(filename).load()

    # Step2: Split the document
    all_splits1 = TokenTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True
    ).split_documents(documents)

    all_splits2 = CharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True
    ).split_documents(documents)

    all_splits3 = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True
    ).split_documents(documents)

    print(all_splits1[0].page_content)
    print(all_splits2[0].page_content)
    print(all_splits3[0].page_content)
    