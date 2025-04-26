import os
from getpass import getpass
from langchain_ollama import ChatOllama
from langchain_google_genai import ChatGoogleGenerativeAI

if __name__ == '__main__':
    run_local = True
    if run_local:
        # Mistral-7b-v0.3 or deepseek-r1:1.5b
        llm = ChatOllama(
            # model="mistral:v0.3",
            model="deepseek-r1:1.5b",
            temperature=0.8,
            num_predict=256,
        )
    else:
        # Gemini 2.0 Flash
        if 'GOOGLE_API_KEY' not in os.environ:
            os.environ['GOOGLE_API_KEY'] = getpass("Enter the API key: ")
        llm = ChatGoogleGenerativeAI(model='gemini-2.0-flash', temperature = 0.8)

    # prompt
    messages = [
        ("system", "You are a helpful translator. Translate the user sentence to Japanese."),
        ("human", "I love programming."),
    ]

    # generate response
    print(llm.invoke(messages).content)