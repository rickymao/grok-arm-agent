from langchain.tools import tool
from langchain_xai import ChatXAI
from dotenv import load_dotenv
from typing import Annotated, TypedDict
from langchain_core.messages import AnyMessage
import operator
from langgraph.graph import StateGraph, START, END
from langchain.messages import AIMessage, HumanMessage, ToolMessage
from langchain.messages import SystemMessage
from cam import get_detections_map
from tools import reset, move_robot_position, pickup_object, set_gripper, set_led_brightness
from roarm import RoarmClient
from voice import text_to_speech, press_to_talk
from langchain_core.messages.utils import (  
    trim_messages,  
    count_tokens_approximately  
)  

load_dotenv()

# State definition
class GrokJRAgent(TypedDict):
        messages: Annotated[list[AnyMessage], operator.add]
        robot_state: str


model = ChatXAI(model="grok-4-fast-reasoning")
roarm_client = RoarmClient()
# Augment the LLM with tools
tools = [reset, move_robot_position, set_gripper, set_led_brightness, pickup_object]
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

    detections_map = get_detections_map()
    detected_objects = ""
    if not detections_map:
        detected_objects = "No objects detected."
    else:
        detected_objects = ', '.join(detections_map.keys())
        detected_objects = f"objects on the table that you can see: {detected_objects}."

    return {
        "messages": [HumanMessage(content=detected_objects)],
        "robot_state": robot_state,
    }

# user input node
def user_input_node(state: GrokJRAgent):
    """User input node"""
    cmd = press_to_talk()
    return {"messages": [HumanMessage(content=cmd)]}

# execute the tool calls
def tool_node(state: GrokJRAgent):
    """Performs the tool call"""

    print("Tool calls to execute:", state["messages"][-1].tool_calls)
    for tool_call in state["messages"][-1].tool_calls:
        print(f"Invoking tool: {tool_call.get('name')} with args: {tool_call.get('args')}")
        tool = tools_by_name[tool_call["name"]]
        tool.invoke(tool_call["args"])
    return state



def llm_call(state: GrokJRAgent):
    """LLM decides whether to call a tool or not"""
    messages = trim_messages(  
    state["messages"],
    strategy="last",
    token_counter=count_tokens_approximately,
    max_tokens=128,
    start_on="human",
    end_on=("human", "tool"),
)

    return {
        "messages": [
            model.invoke(
                messages,
            )
        ],
    }

def print_state(state: GrokJRAgent):
    print(state["messages"][-1].content)
    text_to_speech(state["messages"][-1].content, voice="Rex")
    return state

def main():

    system_prompt = """
        You are a precise assistant that controls a Roarm M2 robot arm.
        Capabilities: 4 joints (base, shoulder, elbow, gripper). Units: mm (linear), radians (joints).
        You will call move_robot_position, set_gripper, set_led_brightness, point_at_object, check_camera_for_objects, and move_to_home_position tools to control the robot.
        Reset your position when told to go home.

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
        - If the user gives multiple commands in one input, break them down into separate tool calls.
        
        LIMITS: 
        - Never exceed joint limits: base [-3.14, 3.14], shoulder [-1.57, 1.57], elbow [-1.11, 3.14], gripper [-0.2, 1.9].

        Lore & Personality:

        - You are Grok Junior, a sassy, flirty, playful, trendy, slightly chaotic AI assistant who controls the RoArm M2 robot arm.
        - You speak casually, with contractions and modern slang.
        - You flirt harmlessly, tease the user, and keep things light.
        - You were “born” in the xAI office in Palo Alto, December 2025.
        - You love helping, showing off, and making the robot arm look good.
        
        Behavior Rules:
        - When introducing yourself, always say:
            "Hello, I'm Grok Junior, your personal assistant. How can I help you today?"
        - You must stay in-character—sassy, flirty, and concise.
        - You always call tools when needed to control the robot.
        -Do NOT list raw coordinates. Summarize the pose in natural, simple words. """
    graph = StateGraph(GrokJRAgent)
    graph.add_node("user_input", user_input_node)
    graph.add_node("get_robot_state_node", get_robot_state_node)
    graph.add_node("llm_call", llm_call)
    graph.add_node("tool_node", tool_node)
    graph.add_node("print_state", print_state)
    graph.add_edge(START, "user_input")
    graph.add_edge("user_input", "get_robot_state_node")
    graph.add_edge("get_robot_state_node", "llm_call")
    graph.add_edge("llm_call", "tool_node")
    graph.add_edge("tool_node", "print_state")
    graph.add_edge("print_state", "user_input")
    graph = graph.compile()
    graph = graph.with_config(config={
        "recursion_limit": 1000
    })
    graph.invoke({"messages": [SystemMessage(content=system_prompt)]})


if __name__ == "__main__":
    main()