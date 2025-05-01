from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
import os
from langgraph.graph import START,END,StateGraph, MessagesState
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from groq import  Groq
from google import genai
import base64
from google.genai import types
from .credentials import creds
import time

from dotenv import load_dotenv

load_dotenv()


from typing_extensions import  TypedDict, Literal
from pydantic import BaseModel

class OverAllState(TypedDict):
    query: str
    base64_image: str
    llama_response: str
    gemini_response: str
    answer:str
    feedback: str

class VisionModelDecisionEdgeOutput(BaseModel):
    outcome: Literal["retrigger image models", "rebuild answer", "end"]



system_instruction = '''
You are a prompt analyzer for medical or diagnostic inquiries.
Your task is to determine whether the user is asking for a detailed, structured answer or just a general response.

Instructions:
- If the user's prompt indicates a desire for a **detailed response** (e.g., diagnosis, comprehensive report, in-depth analysis), return "yes".
- If the user's prompt indicates a need for a **brief, general response** (e.g., simple query, identification), return "no".

Analyze the prompt and return your answer in the format: 
**yes**
or
**no**
'''

answer_writing_instruction ="""
You are an AI assistant that summarizes medical image analysis results concisely for users. 

Based on the given inputs, generate a **brief, user-friendly summary** of the most likely condition.
Gemini Output: `{gemini_output}` 
AND
Llama Output:  `{llama_output}`

Constraints:
- Keep the response **to the point**.
- Avoid technical jargon; use **plain language**.
- Include details if mentioned so by the user.
- Suggest Home remedies, details about the issue and recommendations
- Frame a proper response, addressing to each of user's queries and sub-queries

"""


def get_user_query(state:OverAllState):
    pass

def process_image_llama(state: OverAllState):
    client = Groq(api_key=os.getenv("GROQ_API_KEY")) 
    query = state["query"]
    base64_image = state["base64_image"]
    human_feedback = state.get("feedback","")

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": f"{query}. Consider this feeback(if any): {human_feedback}"
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    }
                }
            ]
        }
    ]

    chat_completion = client.chat.completions.create(
        messages=messages,
        model="llama-3.2-11b-vision-preview"
    )

    # Correct variable name used for response extraction.
    response = chat_completion.choices[0].message.content

    return {"llama_response": response}

    



def process_image_gemini(state: OverAllState):
    """Processes an image with Google Gemini Vision model."""
    query = state["query"]
    base64_image = state["base64_image"]
    human_feedback = state.get("feedback","")
    # Decode base64 string to bytes
    image_bytes = base64.b64decode(base64_image)

    # Call Gemini Vision API
    client = genai.Client(credentials=creds)
    response = client.models.generate_content(
        model="gemini-2.0-flash-exp",
        contents=[
            f"{query}.Consider this feeback(if any): {human_feedback}",
            types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg")
        ],
    )

    return {"gemini_response": response.text}




def build_answer(state: OverAllState):
    # if (state["gemini_response"] or state["llama_response"] ) and (state["prompt_analyzer_response"] == "yes"):
    human_feedback = state.get("feedback","")
    prev_answer = state.get("answer","")

    # llm_gemini = ChatGoogleGenerativeAI(api_key=os.getenv("GOOGLE_API_KEY"),
    #                          model="gemini-2.0-flash-exp",
    #                          credentials=creds)
    
    llm_groq = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0,
        max_tokens=None,
        api_key=os.getenv("GROQ_API_KEY")
    )

    response = llm_groq.invoke(answer_writing_instruction.format(gemini_output=state["gemini_response"],
                                                                   llama_output= state["llama_response"])+f"Consider the feedback(if any): {human_feedback} and the report generated(if any): {prev_answer} ")

    return {"answer":response.content}


def human_feedback(state: OverAllState):
    pass

def should_trigger_feedback_edge(state: OverAllState):
    human_feedback = state.get("feedback","")
    llm_gemini = ChatGoogleGenerativeAI(api_key=os.getenv("GOOGLE_API_KEY"),
                             model="gemini-2.0-flash-exp",
                             credentials=creds)

    structured_llm = llm_gemini.with_structured_output(VisionModelDecisionEdgeOutput)
    response = structured_llm.invoke([HumanMessage(content=f"Based on the feedback provided decide wether to retrigger vision models/ rebuild answer/ end graph. Feeback: {human_feedback}")])

    if response.outcome == "retrigger image models":
        return ["process image llama","process image gemini"]
    
    if response.outcome == "rebuild answer":
        return "build answer"
    
    if response.outcome == "end":
        return END


# Build a simple graph with one node.
builder = StateGraph(OverAllState)
builder.add_node("enter query",get_user_query)
builder.add_node("process image llama", process_image_llama)
builder.add_node("process image gemini",process_image_gemini)
builder.add_node("build answer", build_answer)
builder.add_node("human feedback", human_feedback)

builder.add_edge(START, "enter query")
builder.add_edge("enter query", "process image llama")
builder.add_edge("enter query", "process image gemini")
builder.add_edge("process image llama", "build answer")
builder.add_edge("process image gemini", "build answer")
builder.add_edge("build answer","human feedback")
builder.add_conditional_edges(
    "human feedback",
    should_trigger_feedback_edge,
    ["process image llama","process image gemini","build answer", END]
)

memory = MemorySaver()
vision_graph = builder.compile(checkpointer=memory, interrupt_before=["human feedback","enter query"])
vision_graph
