# from langchain_openai import ChatOpenAI
from pydantic import BaseModel
from langchain_groq import ChatGroq


# class LLMs(BaseModel):
#     large_model: ChatOpenAI
#     mini_model: ChatOpenAI

class LLMs(BaseModel):
    large_model: ChatGroq
    mini_model: ChatGroq