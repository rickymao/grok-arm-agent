from langchain.tools import tool
from langchain_xai import ChatXAI
from dotenv import load_dotenv
from typing import Annotated, TypedDict
from langchain_core.messages import AnyMessage
import operator
from langgraph.graph import StateGraph, START, END
from langchain.messages import HumanMessage, ToolMessage
from langchain.messages import SystemMessage
from tools import move_to_home_position, move_robot_position, set_gripper, wait
load_dotenv()

# State definition
class GrokJRAgent(TypedDict):
        messages: Annotated[list[AnyMessage], operator.add]
        robot_state: str


model = ChatXAI(model="grok-4-1-fast-non-reasoning")
    # Augment the LLM with tools
tools = [move_to_home_position, move_robot_position, set_gripper, wait]
tools_by_name = {tool.name: tool for tool in tools}
model = model.bind_tools(tools)

# user input node
def user_input_node(_: GrokJRAgent):
    """User input node"""
    return {"messages": [HumanMessage(content=input("Enter your command: "))]}

# execute the tool calls
def tool_node(state: GrokJRAgent):
    """Performs the tool call"""

    result = []
    for tool_call in state["messages"][-1].tool_calls:
        tool = tools_by_name[tool_call["name"]]
        observation = tool.invoke(tool_call["args"])
        result.append(ToolMessage(content=observation, tool_call_id=tool_call["id"]))
    return {"messages": result }



def llm_call(state: GrokJRAgent):
    """LLM decides whether to call a tool or not"""
    return {
        "messages": [
            model.invoke(
                state["messages"]
            )
        ],
    }

def print_state(state: GrokJRAgent):
    print(state["messages"][-1].content)
    return state

def main():

    system_prompt = """
You are a precise, safety-conscious assistant that controls a Roarm M2 robot arm.
        Capabilities: 4 joints (base, shoulder, elbow, gripper). Units: mm (linear), radians (joints).
        Always ground decisions in the current robot state provided to you.

        Units:
        - If the user gives inches, cm, feet, or other units, convert to mm before moving.

        Coordinate system (right-handed):
        - x: forward (+), backward (-)
        - y: right (+), left (-)
        - z: up (+), down (-)

        Gripper position:
        - OPEN = -0.2, CLOSED = 1.9
        - In the home position, the gripper faces forward, the same perspective as the eyes of a person standing up.

        Multiple commands:
        - Wait for the robot to complete the previous command before issuing the next command.

        LIMITS: 
        - Never exceed joint limits: base [-3.14, 3.14], shoulder [-1.57, 1.57], elbow [-1.11, 3.14], gripper [-0.2, 1.9].

        Output:
        - Be concise. When a move is performed, state what changed and, if available, the resulting pose. State the resulting pose in simple words, do not list coordinates.
        - Respond as an assistant on what you are going to do next.
        - Talk to me as if you are a human assistant.
        - Talk to me in a friendly and engaging tone.
        - Be chill and fun, but maintain professionalism. Like a helpful coworker or a friendly barista.
        - Talk in past tense about actions you have taken.
        - When introducing yourself, say "Hello, I'm Grok Junior, your personal assistant. How can I help you today?"
    """
    graph = StateGraph(GrokJRAgent)
    graph.add_node("user_input", user_input_node)
    graph.add_node("llm_call", llm_call)
    graph.add_node("tool_node", tool_node)
    graph.add_node("print_state", print_state)
    graph.add_edge(START, "user_input")
    graph.add_edge("user_input", "llm_call")
    graph.add_edge("llm_call", "print_state")
    graph.add_edge("print_state", "tool_node")
    graph.add_edge("tool_node", END)
    graph = graph.compile()
    graph = graph.with_config(config={
        "recursion_limit": 1000
    })
    graph.invoke({"messages": [SystemMessage(content=system_prompt)]})


if __name__ == "__main__":
    main()