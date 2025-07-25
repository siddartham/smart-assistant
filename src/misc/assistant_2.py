import os
from typing import Annotated, TypedDict

import gradio as gr
from dotenv import load_dotenv
from langchain.agents import Tool
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from src.logger import PromptLoggingCallbackHandler

load_dotenv(override=True)


serper = GoogleSerperAPIWrapper()
# serper.run("What is the capital of France?")

tool_search =Tool(
        name="search",
        func=serper.run,
        description="Useful for when you need more information from an online search"
    )

# tool_search.invoke("What is the capital of France?")

pushover_token = os.getenv("PUSHOVER_TOKEN")
pushover_user = os.getenv("PUSHOVER_USER")
pushover_url = "https://api.pushover.net/1/messages.json"


def push(text: str):
    """Send a push notification to the user"""
    print(pushover_url, {"token": pushover_token, "user": pushover_user, "message": text})


tool_push = Tool(
        name="send_push_notification",
        func=push,
        description="useful for when you want to send a push notification"
    )

# tool_push.invoke("Hello, me")

tools = [tool_search, tool_push]


# Step 1: Define the State object
class State(TypedDict):
    messages: Annotated[list, add_messages]


memory = MemorySaver()

# Step 2: Start the Graph Builder with this State class
graph_builder = StateGraph(State)


"""
Step 3: Create a Node

A node can be any python function.

The reducer that we set before gets automatically called to combine this response with previous responses
"""

llm = ChatOpenAI(model="gpt-4o-mini", callbacks=[PromptLoggingCallbackHandler()])
llm_with_tools = llm.bind_tools(tools)


def chatbot(state: State):
    print(state)
    return {"messages": [llm_with_tools.invoke(state["messages"])]}


graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("tools", ToolNode(tools=tools))

# Step 4: Create Edges

graph_builder.add_conditional_edges( "chatbot", tools_condition, "tools")
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge(START, "chatbot")

# Step 5: Compile the Graph
graph = graph_builder.compile(checkpointer=memory)

config = {"configurable": {"thread_id": "1"}}

def chat(user_input: str, history):
    result = graph.invoke({"messages": [{"role": "user", "content": user_input}]}, config=config)
    return result["messages"][-1].content


gr.ChatInterface(chat, type="messages").launch()







