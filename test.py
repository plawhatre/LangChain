from langchain_ollama import ChatOllama


if __name__ == '__main__':
    # model
    llm = ChatOllama(
        model = "mistral:v0.3",
        temperature = 0.8,
        num_predict = 256,
    )

    # prompt
    messages = [
        ("system", "You are a helpful translator. Translate the user sentence to Japanese."),
        ("human", "I love programming."),
    ]

    # generate response
    print(llm.invoke(messages))