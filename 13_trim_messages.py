from langchain_core.messages import (
    SystemMessage, 
    HumanMessage, 
    AIMessage,
    trim_messages
)
from langchain_core.messages.utils import count_tokens_approximately
from langchain_google_genai import ChatGoogleGenerativeAI

# Step1: Have a chat history in list
messages = [
    SystemMessage("You are a helpful assistant"),
    HumanMessage("Hi, my name is Prashant"),
    AIMessage("Hi Prashant, How can I help you?"),
    HumanMessage("What is 2+2?"),
    AIMessage("2+2 is 4"),
    HumanMessage("what is 2/2?"),
    AIMessage("It is 1"),
    HumanMessage("What is my Name?")
]

# Step2: Model
llm = ChatGoogleGenerativeAI(model='gemini-1.5-flash')

# Step3.1: Create a trimmer object (based on token length)
trimmer1 = trim_messages(
    start_on='human',
    end_on=('human', 'tool'),
    include_system=True,
    allow_partial=False,
    strategy='last',
    token_counter=count_tokens_approximately,
    max_tokens=45
)

# Step3.2: Create a trimmer object (based on LLM's token count)
trimmer2 = trim_messages(
    start_on='human',
    end_on=('human', 'tool'),
    include_system=True,
    allow_partial=False,
    strategy='last',
    token_counter=llm,
    max_tokens=200
)

# Step3.3: Create a trimmer object (based on messages count)
trimmer3 = trim_messages(
    start_on='human',
    end_on=('human', 'tool'),
    include_system=True,
    allow_partial=False,
    strategy='last',
    token_counter=len,
    max_tokens=4
)

# Step4: Create chain
chain1 = trimmer1 | llm
chain2 = trimmer2 | llm
chain3 = trimmer3 | llm

print("chain1:\n", chain1.invoke(messages).content, "\n")
print("chain2:\n", chain2.invoke(messages).content, "\n")
print("chain3:\n", chain3.invoke(messages).content, "\n")
