import os
from getpass import getpass
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI

from pydantic import BaseModel, Field
from typing import Optional

class Joke(BaseModel):
    topic: str = Field(default="", description="topic on which joke was written")
    joke: str = Field(default="", description="the joke on the topic")
    rating: Optional[int] = Field(default=None, description="What is the rating of the joke on a scale of 1 to 10")


if __name__ == '__main__':
    # Setting
    run_local = True
    stream = False
    output_parser = False
    
    # LLM
    if run_local:
        # Mistral-7b-v0.3 or deepseek-r1:1.5b
        llm = ChatOllama(
            model="mistral:v0.3",
            # model="deepseek-r1:1.5b",
            temperature=0.8,
            num_predict=256,
        )
    else:
        # Gemini 2.0 Flash
        if 'GOOGLE_API_KEY' not in os.environ:
            os.environ['GOOGLE_API_KEY'] = getpass("Enter the API key: ")
        llm = ChatGoogleGenerativeAI(model='gemini-2.0-flash', temperature = 0.8)

    # Prompt
    prompt_template  = ChatPromptTemplate([
        SystemMessagePromptTemplate.from_template("You are a funny comedian."),
        HumanMessagePromptTemplate.from_template("tell me a joke on {topic}. Also, rate teh joke on a scale of 1 to 10.")
    ])

    # Chain
    if not stream:
        # Known issue: https://github.com/langchain-ai/langchain/issues/24225#issuecomment-2387323137
        # this might not work with ChatGoogleGenerativeAI
        # Parser
        if output_parser:
            parser = PydanticOutputParser(pydantic_object=Joke)
            chain = prompt_template.partial(format_instructions=parser.get_format_instructions()) | llm
        else:
            chain = prompt_template | llm.with_structured_output(Joke)
            # chain = prompt_template | llm.bind_tools([Joke]) # not all models support this

    else:
        chain = prompt_template | llm

    # Generate Response: Invocation
    if not stream:
        print(chain.invoke([{"topic": "Birds"}]))
        # print(chain.invoke([{"topic": "Birds"}]).tool_calls[0]["args"])
    else:
        for chunk in chain.stream([{"topic": "Birds"}]):
            print(chunk.content)