from typing import Annotated, TypedDict
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode, tools_condition
from .tools import tools
from .logger import logger
from dotenv import load_dotenv
from langchain_core.messages import AIMessage
from .tools import document_qa_tool
from langchain_core.callbacks.stdout import StdOutCallbackHandler
from langchain_core.callbacks import CallbackManager

load_dotenv()


llm = ChatOpenAI(model="gpt-4o")
llm_with_tools = llm.bind_tools(tools)

class State(TypedDict):
    messages: Annotated[list, add_messages]


memory = MemorySaver()
graph_builder = StateGraph(State)

def assistant(state: State):
    user_msg = state["messages"][-1].content
    logger.info(f"[User Message] {user_msg}")

    try:
        response = llm_with_tools.invoke(state["messages"])

        if isinstance(response, AIMessage) and not getattr(response, "tool_calls", []):
            logger.warning("LLM responded without using a tool. Falling back to document_qa_tool.")
            fallback_answer = document_qa_tool(user_msg)
            return {"messages": [{"role": "assistant", "content": fallback_answer}]}

        logger.info(f"[Assistant Response] {response.content}")
        return {"messages": [response]}
    except Exception as e:
        logger.exception("Tool execution failed")
        return {"messages": [{"role": "assistant", "content": "Sorry, I didn’t quite understand that."}]}


graph_builder.add_node("assistant", assistant)
graph_builder.add_node("tools", ToolNode(tools=tools))
graph_builder.add_conditional_edges("assistant", tools_condition, "tools")
graph_builder.add_edge("tools", "assistant")
graph_builder.add_edge(START, "assistant")

graph = graph_builder.compile(checkpointer=memory)

# graph = graph_builder.compile(
#     checkpointer=memory,
#     callback_manager=CallbackManager([StdOutCallbackHandler()])
# )
