from langchain_ollama import OllamaEmbeddings
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_core.runnables import ConfigurableField


if __name__ == "__main__":
    docs1 = [
        "I like apples",
        "I like oranges",
        "Apple and orange are fruits"
    ]
    docs2 = [
        "You like apple",
        "You like oranges"
    ]

    # Step1: Initialise the retrievers
    bm25_retriever = BM25Retriever.from_texts(
        texts=docs1,
        metadatas=[{'source': 'docs1'}] * len(docs1)
    )
    bm25_retriever.k = 2

    faiss_vectorstore = FAISS.from_texts(
        texts=docs2,
        embedding=OllamaEmbeddings(model='mistral:v0.3'),
        metadatas=[{'source': 'docs2'}] * len(docs2)
    )
    faiss_retriever = faiss_vectorstore.as_retriever(search_kwargs={'k': 2}).configurable_fields(
        search_kwargs=ConfigurableField(
            id='search_kwargs_faiss',
            name='Search Kwargs',
            description="search kwargs for faiss"
        )
    )

    # Step2: Combine them both
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, faiss_retriever],
        weights=[0.5, 0.5]
    )

    # Step3: Invoke retrievers (with optional config)
    config = {"configurable": {"search_kwargs_faiss": {'k': 1}}}
    docs = ensemble_retriever.invoke("apple", config=config)
    for doc in docs:
        print(doc)
