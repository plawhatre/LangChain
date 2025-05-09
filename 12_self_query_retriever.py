from langchain_core.documents import Document
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.schema import AttributeInfo
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma

if __name__ == "__main__":
    # Step 1: Documents
    docs = [
        Document(
            page_content="Kewa Datshi: Potatoes cooked with yak butter",
            metadata={
                "cuisine": "Bhutanese",
                "type": "Veg",
                "spiciness": "Low"
            }
        ),
        Document(
            page_content="Ema Datshi: Chillies cooked with yak butter",
            metadata={
                "cuisine": "Bhutanese",
                "type": "Veg",
                "spiciness": "High"
            }
        ),
        Document(
            page_content="Butter Chicken: Chicken curry cooked in butter",
            metadata={
                "cuisine": "Indian",
                "type": "Non-veg",
                "spiciness": "Moderate"
            }
        ),
        Document(
            page_content="Ramen: Noodles cooked in broth",
            metadata={
                "cuisine": "Japanese",
                "type": "Non-veg",
                "spiciness": "Low"
            }
        )
    ]

    # Step 2: Describe the document and its metadata
    document_content_description = "Breif description of the dish"
    metadata_fields_info = [
        AttributeInfo(
            name="cuisine",
            description="Describes the country of origin",
            type="string"
        ),
        AttributeInfo(
            name="type",
            description="Describes if the dish is vegetarian or non-vegetarian",
            type="string"
        ),
        AttributeInfo(
            name="spiciness",
            description="Describe the spice level",
            type="string"
        )
    ]

    # Step 3: Model
    embedder = OllamaEmbeddings(model="mistral:v0.3")
    vector_store = Chroma.from_documents(
        docs,
        embedder
    )

    # Step 4: Model
    model = ChatOllama(
        model="mistral:v0.3",
        temperature=0.2
    )

    # Step 4: Retriever
    retriever = SelfQueryRetriever.from_llm(
        model,
        vector_store,
        document_content_description,
        metadata_fields_info
    )

    # Step 5: Query
    query = "Suggest some dish in Japanese cuisine"
    output = retriever.invoke(query)
    print(output)