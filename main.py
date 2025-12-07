from langchain.tools import tool
from langchain_xai import ChatXAI
from dotenv import load_dotenv
from typing import Annotated, TypedDict
from langchain_core.messages import AnyMessage
import operator
from langgraph.graph import StateGraph, START, END
from langchain.messages import ToolMessage
from langchain.messages import SystemMessage
load_dotenv()
class GrokJRAgent(TypedDict):
        messages: Annotated[list[AnyMessage], operator.add]
        robot_state: str

# Define tools
@tool
def multiply(a: int, b: int) -> int:
    """Multiply `a` and `b`.

    Args:
        a: First int
        b: Second int
    """
    return a * b


@tool
def add(a: int, b: int) -> int:
    """Adds `a` and `b`.

    Args:
        a: First int
        b: Second int
    """
    return a + b


@tool
def divide(a: int, b: int) -> float:
    """Divide `a` and `b`.

    Args:
        a: First int
        b: Second int
    """
    return a / b



model = ChatXAI(model="grok-4-1-fast-non-reasoning")
    # Augment the LLM with tools
tools = [add, multiply, divide]
tools_by_name = {tool.name: tool for tool in tools}
model = model.bind_tools(tools)

# execute the tool calls
def tool_node(state: GrokJRAgent):
    """Performs the tool call"""

    result = []
    for tool_call in state["messages"][-1].tool_calls:
        tool = tools_by_name[tool_call["name"]]
        observation = tool.invoke(tool_call["args"])
        result.append(ToolMessage(content=observation, tool_call_id=tool_call["id"]))
    return {"messages": result}



def llm_call(state: GrokJRAgent):
    """LLM decides whether to call a tool or not"""
    return {
        "messages": [
            model.invoke(
                [
                    SystemMessage(
                        content="You are a helpful assistant tasked with performing arithmetic on a set of inputs."
                    )
                ]
                + state["messages"]
            )
        ],
    }

def print_state(state: GrokJRAgent):
    print(state["messages"][-1].content)
    return state

def main():

    graph = StateGraph(GrokJRAgent)
    graph.add_node("llm_call", llm_call)
    graph.add_node("tool_node", tool_node)
    graph.add_node("print_state", print_state)
    graph.add_edge(START, "llm_call")
    graph.add_edge("llm_call", "print_state")
    graph.add_edge("print_state", "tool_node")
    graph.add_edge("tool_node", END)
    graph = graph.compile()
    graph.invoke({"messages": [{"role": "user", "content": "hi!"}]})


if __name__ == "__main__":
    main()