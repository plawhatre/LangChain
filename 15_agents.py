from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool

from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

@tool
def multiply(a: float, b:float) -> float:
    """This function takes two floats and then multiplies them"""
    return a*b

@tool
def addition(a: float, b: float) -> float:
    """This function takes two floats and then adds them"""
    return a+b


if __name__ == '__main__':

    # Step 1: Create Model, Tool and Memory
    model = ChatGoogleGenerativeAI(model='gemini-1.5-flash')
    tools = [multiply, addition]
    memory = MemorySaver()

    # Step 2: Create Agent
    agent_executor = create_react_agent(
        model, 
        tools=tools, 
        checkpointer=memory
    )

    # Step 3: Use Agent
    config = {"configurable": {"thread_id": '123abc'}}
    messages = [
        SystemMessage("You are a helpful assistant who helps with any question asked of you."),
        HumanMessage("Hi, my name is Prashant"),
        AIMessage("Hi Prashant. How can I help you?"),
        HumanMessage("Multiply 5.2 by 63")
    ]

    # Step 3.a: First call 
    response = agent_executor.invoke(
        {
            "messages": messages
        },
        config=config
    )

    # Step 3.b: Second call
    response["messages"].extend([HumanMessage("Add the result from the last query to 23")])
    response = agent_executor.invoke(
        {
            "messages": response["messages"]
        },
        config=config
    )

    # Step 4: Print Message History
    for message in response["messages"]:
        message.pretty_print()
