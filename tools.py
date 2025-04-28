import os
import getpass

from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage

@tool
def multiply(a: float, b:float) -> float:
    """This function takes two floats and then multiplies them"""
    return a*b

@tool
def addition(a: float, b: float) -> float:
    """This function takes two floats and then adds them"""
    return a+b


if __name__ == '__main__':
    # Messages and Prompt
    system_message = SystemMessage("You are a helpful assistant. You help with any question that is asked to you.")
    human_message = HumanMessage("What is multiplication of 5.2 and 63. Then add that number with 23 ?")
    messages = [system_message, human_message]

    # Model
    if 'GOOGLE_API_KEY' not in os.environ:
        os.environ['GOOGLE_API_KEY'] = getpass.getpass("Enter the api key for gemini-2.0-flash: ")
    llm = ChatOllama(model='mistral:v0.3', temperature=0.0)
    llm_with_tools = llm.bind_tools([addition, multiply])

    # invocation
    output = llm_with_tools.invoke(messages)

    if len(output.tool_calls) == 0:
        print("No tool calling")
        final_output = output
    else:
        print("Tool called")
        # loop through all the functions that were called and manually invoke them
        for tool_call in output.tool_calls:
            print(f"Processing tool: {tool_call['name']}, {tool_call}")
            selected_tool = {'addition': addition, 'multiply': multiply}[tool_call['name'].lower()]
            tool_output = selected_tool.invoke(tool_call)
            messages.append(tool_output)

        # invoke the chain one last time with the ToolMessage
        final_output = llm_with_tools.invoke(messages)

    print(f"final_output content: {final_output.content}")



