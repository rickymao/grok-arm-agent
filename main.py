from langchain.tools import tool
from langchain_xai import ChatXAI
from dotenv import load_dotenv
from typing import Annotated, TypedDict
from langchain_core.messages import AnyMessage
import operator
from langgraph.graph import StateGraph, START, END
from langchain.messages import AIMessage, HumanMessage, ToolMessage
from langchain.messages import SystemMessage
from tools import move_to_home_position, move_robot_position, pick_up_object, set_gripper, set_led_brightness
from roarm import RoarmClient
from voice import text_to_speech
load_dotenv()

# State definition
class GrokJRAgent(TypedDict):
        messages: Annotated[list[AnyMessage], operator.add]
        robot_state: str


model = ChatXAI(model="grok-4-1-fast-non-reasoning")
roarm_client = RoarmClient()
    # Augment the LLM with tools
tools = [move_to_home_position, move_robot_position, set_gripper, set_led_brightness, pick_up_object]
tools_by_name = {tool.name: tool for tool in tools}
model = model.bind_tools(tools)

# get the current robot state
def get_robot_state_node(_: GrokJRAgent):
    """Get the current robot state and joint radians"""
    pose = roarm_client.pose_get()
    pose = [round(x, 2) for x in pose]

    joint_radians = roarm_client.joints_radian_get()
    joint_radians = [round(x, 2) for x in joint_radians]

    robot_state = {
        "x": pose[0],
        "y": pose[1],
        "z": pose[2],
        "gripper": pose[3],
        "base_joint": joint_radians[0],
        "shoulder_joint": joint_radians[1],
        "elbow_joint": joint_radians[2],
        "gripper_joint": joint_radians[3]
    }

    return {
        "messages": [SystemMessage(content=f"The current robot state is: {str(robot_state)}")],
        "robot_state": robot_state
    }

# user input node
def user_input_node(_: GrokJRAgent):
    """User input node"""
    return {"messages": [HumanMessage(content=input("Enter your command: "))]}

# execute the tool calls
def tool_node(state: GrokJRAgent):
    """Performs the tool call"""

    result = []
    print("Tool calls to execute:", state["messages"][-1].tool_calls)
    for tool_call in state["messages"][-1].tool_calls:
        print(f"Invoking tool: {tool_call.get('name')} with args: {tool_call.get('args')}")
        tool = tools_by_name[tool_call["name"]]
        content = tool.invoke(tool_call["args"])
        result.append(content)
        result.append(AIMessage(content="\n".join(result)))
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
    text_to_speech(state["messages"][-1].content)
    return state

def main():

    system_prompt = """
You are a precise assistant that controls a Roarm M2 robot arm.
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
        - Output tool calls when necessary to perform actions.
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
    graph.add_node("get_robot_state_node", get_robot_state_node)
    graph.add_node("llm_call", llm_call)
    graph.add_node("tool_node", tool_node)
    graph.add_node("print_state", print_state)
    graph.add_edge(START, "user_input")
    graph.add_edge("user_input", "get_robot_state_node")
    graph.add_edge("get_robot_state_node", "llm_call")
    graph.add_edge("llm_call", "print_state")
    graph.add_edge("print_state", "tool_node")
    graph.add_edge("tool_node", "user_input")
    graph = graph.compile()
    graph = graph.with_config(config={
        "recursion_limit": 1000
    })
    graph.invoke({"messages": [SystemMessage(content=system_prompt)]})


if __name__ == "__main__":
    main()