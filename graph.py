from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from openai import OpenAI
from typing import Literal
from dotenv import load_dotenv
from langsmith.wrappers import wrap_openai
from pydantic import BaseModel

# Response Schema
class DetectcallResponse(BaseModel):
    is_question_coding : bool

class CodingResponse(BaseModel):
    answer : str

client = wrap_openai(OpenAI())

load_dotenv()

class State(TypedDict):
    user_message: str
    ai_message: str
    is_coding_question: bool

def detect_query(state: State):
    user_message = state.get("user_message")
    system_prompt = """
    You are a helpful AI assiatent who tells the user if his query is related to coding or not
    Return the response in specified JSON boolean only.
    """
    result = client.beta.chat.completions.parse(
        response_format=DetectcallResponse,
        model="gpt-4o-mini",
        messages=[
            {"role":"system" , "content":system_prompt},
            {"role":"user", "content":user_message}
        ]
    )

    state["is_coding_question"] = result.choices[0].message.parsed.is_question_coding
    return state

def route_edge(state: State) -> Literal["solve_coding_question", "solve_simple_question"]:
    is_coding_question = state.get("is_coding_question")
    if is_coding_question:
        return "solve_coding_question"
    else:
        return "solve_simple_question"

def solve_coding_question(state: State):
    user_message = state.get("user_message")
    system_prompt = """
    You are a helpful AI assiatent. Your Job is to resolve the user query based on
    coding problem he is facing.
    """
    result = client.beta.chat.completions.parse(
        response_format=CodingResponse,
        model="gpt-4.1",
        messages=[
            {"role":"system" , "content":system_prompt},
            {"role":"user", "content":user_message}
        ]
    )

    state["ai_message"] = result.choices[0].message.parsed.answer

    return state

def solve_simple_question(state: State):
    user_message = state.get("user_message")

    system_prompt = """
    You are a helpful AI assiatent. Your Job is to resolve the user query.
    """
    result = client.beta.chat.completions.parse(
        response_format=CodingResponse,
        model="gpt-4o-mini",
        messages=[
            {"role":"system" , "content":system_prompt},
            {"role":"user", "content":user_message}
        ]
    )

    state["ai_message"] = result.choices[0].message.parsed.answer

    return state

# Build the graph
graph_builder = StateGraph(State)

graph_builder.add_node("detect_query", detect_query)
graph_builder.add_node("solve_coding_question", solve_coding_question)
graph_builder.add_node("solve_simple_question", solve_simple_question)
graph_builder.add_node("route_edge", route_edge)

graph_builder.add_edge(START, "detect_query")
graph_builder.add_conditional_edges("detect_query", route_edge)
graph_builder.add_edge("solve_simple_question", END)
graph_builder.add_edge("solve_coding_question", END)

graph = graph_builder.compile()

# Use the graph
def call_graph():
    inital_state = {
        "user_message": "What is the color of the sky a mi amore a mi amore",
        "ai_message": "",
        "is_coding_question": False       
    }
    result = graph.invoke(inital_state)
    print("Final Result: ", result )

call_graph()


