from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from .logger import PromptLoggingCallbackHandler, logger
from .tools import tools

load_dotenv()
llm = ChatOpenAI(model="gpt-4o-mini", callbacks=[PromptLoggingCallbackHandler()])
llm_with_tools = llm.bind_tools(tools)
memory = MemorySaver()


class State(TypedDict):
    messages: Annotated[list, add_messages]


graph_builder = StateGraph(State)


def assistant(state: State):
    user_msg = state["messages"][-1].content
    logger.info(f"[User Message] {user_msg}")

    try:
        response = llm_with_tools.invoke(state["messages"])
        logger.info(f"Response {response}")
        logger.info(f"[Tool Raw Response] {response.content}")
        # If the tool response is structured, generate reasoning and source
        if isinstance(response.content, dict) and "answer" in response.content:
            content = response.content
            tool_trace = content.get("tool_trace", [content.get("tool")])
            trace_msg = " → ".join(tool_trace)
            if "generated_code" in content:
                trace_msg += f"\n\n**Executed Query Code:**\n```python\n{content['generated_code']}\n```"

            reasoning_hint = (
                "The assistant escalated from one tool to another because the initial tool could not sufficiently "
                "answer the query."
                "This ensured the user received a more accurate and complete response."
                if len(set(tool_trace)) > 1 else
                "The assistant selected the most appropriate tool based on user query."
            )

            reflection_prompt = (
                f"The assistant just used a tool to respond to the user.\n\n"
                f"User query: {user_msg}\n"
                f"Tool trace: {trace_msg}\n"
                f"Tool output: {content.get('answer')}\n\n"
                f"Please write a summary that includes:\n{reasoning_hint}\n"
                "1. The final answer\n"
                "2. A breakdown of why these tool(s) were selected\n"
                "3. How the answer was derived\n"
                "4. What source(s) were used (e.g., PDF documents, structured tables)\n\n"
                "Format as markdown with the following headings: Answer, Tool Trace, Reasoning, Source."
            )
            reflection = llm.invoke(reflection_prompt).content
            response.content = reflection

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
