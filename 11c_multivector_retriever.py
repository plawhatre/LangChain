import uuid
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain.storage import InMemoryByteStore
from langchain.retrievers import MultiVectorRetriever


if __name__ == '__main__':
    # constant
    filename1 = "attention_is_all_you_need.pdf"
    filename2 = "bert.pdf"
    query = "what is attention?"

    # Step1: Load the pdfs
    docs = []
    doc1 = PyPDFLoader(filename1).load()
    doc2 = PyPDFLoader(filename2).load()
    docs.extend(doc1)
    docs.extend(doc2)

    # Step2: Split the text
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True
    )
    docs = splitter.split_documents(docs)

    # Perform: Multivector technique -> a. Smaller Chunks
    # Step3.a.1: Get sub docs and doc_ids
    id_key = 'doc_id' 
    child_text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300, 
        chunk_overlap=100, 
        add_start_index=False
    )
    
    sub_docs = []
    doc_ids = [str(uuid.uuid4()) for _ in docs]
    for i, doc in enumerate(docs):
        _sub_docs = child_text_splitter.split_documents([doc])
        _id = doc_ids[i]
        for _doc in _sub_docs:
            _doc.metadata[id_key] = _id
        sub_docs.extend(_sub_docs)

    # Step3.a.2: Vectorstore, Docstore, Retriever
    vectorstore = Chroma(
        collection_name="all_docs", 
        embedding_function=OllamaEmbeddings(model='mistral:v0.3')
    )
    byte_store = InMemoryByteStore()
    retriever = MultiVectorRetriever(
        vectorstore=vectorstore,
        byte_store=byte_store,
        id_key=id_key
    )

    retriever.vectorstore.add_documents(sub_docs)
    retriever.docstore.mset(list(zip(doc_ids, docs)))

    # Step3.a.3: Get output
    otuput_smaller_chunks = retriever.invoke(query)
    print(otuput_smaller_chunks[0].page_content)


