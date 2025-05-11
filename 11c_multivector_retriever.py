import uuid
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain.storage import InMemoryByteStore
from langchain.retrievers import MultiVectorRetriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from pydantic import BaseModel, Field
from typing import List


class HypoQuestions(BaseModel):
    """Generate a list of three hypothetical questions"""
    questions: List[str] = Field("List of 3 questions")


if __name__ == '__main__':
    # constant
    filename = "attention_is_all_you_need.pdf"
    query = "what is attention?"
    choices = {
        "smaller_chunks": False,
        "summaries": False,
        "hypothetical_question": False
    }
    options = {
        1: "smaller_chunks",
        2: "summaries",
        3: "hypothetical_question"
    }
    inp = int(input("Enter\n\t1. for smaller chunks\n\t2. for summaries\n\t3. for hypothetical question\n"))
    try:
        choices[options[inp]] = True
    except:
        raise Exception("Enter correct input")

    # Step1: Load the pdfs
    docs = []
    doc1 = PyPDFLoader(filename).load()
    docs.extend(doc1)

    # Step2: Split the text
    # splitter = RecursiveCharacterTextSplitter(
    #     chunk_size=1000,
    #     chunk_overlap=200,
    #     add_start_index=True
    # )
    # docs = splitter.split_documents(docs)

    # Perform: Multivector technique -> a. Smaller Chunks
    if choices[options[1]]:
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
            collection_name="small_chunks", 
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

    # Perform: Multivector technique -> b. Summaries
    if choices[options[2]]:
        # Step3.b.1: Create Chain
        prompt = ChatPromptTemplate.from_template("Get the summary of the document:\n\n{doc}")
        model = ChatOllama(model='mistral:v0.3', temperature=0.0)
        output_parser = StrOutputParser()
        chain = {'doc': lambda x: x.page_content} | prompt | model | output_parser
        summaries = chain.batch(docs,  {"max_concurrency": 5})

        # Step3.b.2: Vectorstore, Docstore, Retriever
        vectorstore = Chroma(
            collection_name="summaries",
            embedding_function=OllamaEmbeddings(model='mistral:v0.3')
        )
        byte_store = InMemoryByteStore()
        id_key = 'doc_id' 
        retriever = MultiVectorRetriever(
            vectorstore=vectorstore,
            byte_store=byte_store,
            id_key=id_key
        )

        # Step3.b.3: Attach summaries with their docs
        doc_ids = [str(uuid.uuid4()) for  _ in docs]
        summaries_docs = []
        for i, summary in enumerate(summaries):
            doc =  Document(
                page_content=summary,
                metadata={
                    id_key: doc_ids[i]
                }
            )
            summaries_docs.append(doc)
        
        retriever.vectorstore.add_documents(summaries_docs)
        retriever.docstore.mset(list(zip(doc_ids, docs)))

    # Perform: Multivector technique -> c. Hypothetical Questions
    if choices[options[3]]:
        # Step3.c.1: Create chain
        prompt = ChatPromptTemplate.from_template(
            "Generate exactly three questions that the below dopcument would be able to answer:\n\n{doc}"
        )
        model = ChatOllama(model='mistral:v0.3', temperature=0.0)
        chain = (
            {'doc': lambda x: x.page_content}
            | prompt 
            | model.with_structured_output(HypoQuestions)
            | (lambda x: x.questions)
        )
        ques = chain.batch(docs, {'max_concurrency': 5})

        # Step3.c.2: Vectorstore, Docstore, Retriever
        vectorstore = Chroma(
            collection_name="hypothetical_questions",
            embedding_function=OllamaEmbeddings(model='mistral:v0.3')
        )
        byte_store = InMemoryByteStore()
        id_key = 'doc_id'
        retriever = MultiVectorRetriever(
            vectorstore=vectorstore,
            byte_store=byte_store,
            id_key=id_key
        )

        # Step3.c.3: Attach questions with their docs
        doc_ids = [str(uuid.uuid4()) for _ in docs]
        questions = []
        for i, que in enumerate(ques):
            _id = doc_ids[i]
            questions.extend([Document(page_content=q, metadata={id_key: _id}) for q in que])
        
        retriever.vectorstore.add_documents(questions)
        retriever.docstore.mset(list(zip(doc_ids, docs)))

    # Step4: Get output
    otuput_smaller_chunks = retriever.invoke(query)
    print(otuput_smaller_chunks[0].page_content) 



