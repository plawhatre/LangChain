import os
import httpx
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader


if __name__ == "__main__":
    load_local = True

    if not load_local:
        # from Document class: on the fly
        document1 = Document(
            page_content="Dogs are great companions.",
            metadata={"source": "pet_docs"}
        )

        document2 = Document(
            page_content="Cats are independent pet",
            metadata={"source": "pet_docs"}
        )

        documents = [document1, document2]
        print(documents) 

    else:
        # from document_loaders module: from local
        url = "https://arxiv.org/pdf/1706.03762"
        filename = "attention_is_all_you_need.pdf"

        if not os.path.isfile(filename):
            content = httpx.get(url).content
            with open(filename, 'wb') as file:
                file.write(content)

        documents = PyPDFLoader(filename).load()
        print(documents)