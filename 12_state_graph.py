from langgraph.graph import (
    START, 
    END, 
    StateGraph
)
from typing import TypedDict

class InputState(TypedDict):
    user_input: str

class OutputState(TypedDict):
    graph_output: str

class OverallState(TypedDict):
    overall: str
    user_input: str
    graph_output: str

class PrivateState(TypedDict):
    temp: str

def node1(state: InputState) -> OverallState:
    return {"overall": state["user_input"] + " name"}

def node2(state: OverallState) -> PrivateState:
    return {"temp": state['overall'] + " is"}

def node3(state: PrivateState) -> OutputState:
    return {"graph_output": state['temp'] + " Prashant"}


if __name__ == "__main__":
    # Step 1: Initialise graph
    builder = StateGraph(OverallState, input=InputState, output=OutputState)

    # Step 2: Add nodes
    builder.add_node("node1", node1)
    builder.add_node("node2", node2)
    builder.add_node("node3", node3)

    # Step3: Add edges
    builder.add_edge(START, 'node1')
    builder.add_edge('node1', 'node2')
    builder.add_edge('node2', 'node3')
    builder.add_edge('node3', END)

    # Step 4: Compile graph
    graph = builder.compile()
    response = graph.invoke({"user_input": "My"})
    print(response)

