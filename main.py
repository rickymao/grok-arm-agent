from langchain.tools import tool
from langchain_xai import ChatXAI
from dotenv import load_dotenv


load_dotenv()

model = ChatXAI(model="grok-4-1-fast-non-reasoning")

class GrokJRAgent(BaseModel):

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


# Augment the LLM with tools
tools = [add, multiply, divide]
tools_by_name = {tool.name: tool for tool in tools}
model_with_tools = model.bind_tools(tools)
def main():
    print("Hello from grok-jr-agent!")


if __name__ == "__main__":
    main()
