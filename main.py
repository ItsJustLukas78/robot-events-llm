import os
import typing
from typing import Dict, List, TextIO

import streamlit as st
import torch
import yaml
from dotenv import load_dotenv
from langchain import OpenAI
from langchain.agents import AgentType, initialize_agent, load_tools
from langchain.agents.agent_toolkits import NLAToolkit, OpenAPIToolkit
from langchain.agents.agent_toolkits.openapi.spec import (
    ReducedOpenAPISpec,
    reduce_openapi_spec,
)
from langchain.chains import APIChain, LLMChain, OpenAPIEndpointChain
from langchain.chat_models import ChatOpenAI
from langchain.llms import GPT4All, OpenAI
from langchain.memory import (
    CombinedMemory,
    ConversationBufferMemory,
    ConversationKGMemory,
    ConversationTokenBufferMemory,
)
from langchain.requests import Requests, RequestsWrapper
from langchain.schema import AgentAction, AgentFinish
from langchain.tools import AIPluginTool, APIOperation, OpenAPISpec
from transformers import pipeline

load_dotenv()

from langchain.agents.agent_toolkits.openapi import planner

# hf_model = pipeline(
#     model="databricks/dolly-v2-12b",
#     torch_dtype=torch.bfloat16,
#     trust_remote_code=True,
#     device_map="auto",
#     return_full_text=True,
# )

openai_llm: ChatOpenAI = ChatOpenAI(
    temperature=0.2, model_name="gpt-3.5-turbo", verbose=True
)

# gpt4all_model = GPT4All(
#     model="./models/ggml-gpt4all-j-v1.3-groovy.bin",
#     n_ctx=512,
#     n_threads=8,
#     backend="gptj",
# )

current_llm = openai_llm

headers: Dict = {
    "Content-Type": "application/json",
    "Accept": "application/json",
    "Authorization": "Bearer " + os.getenv("ROBOT_EVENTS_API_KEY"),
}

requests_wrapper: RequestsWrapper = RequestsWrapper(headers=headers)

with open("assets/RE_documentation.yaml", "r") as file:
    raw_RE_api_apec: Dict = yaml.load(file, Loader=yaml.Loader)
    RE_api_spec: ReducedOpenAPISpec = reduce_openapi_spec(raw_RE_api_apec)

robot_events_agent = planner.create_openapi_agent(
    RE_api_spec,
    requests_wrapper,
    current_llm,
    shared_memory=ConversationBufferMemory(
        input_key="input", memory_key="buffer_memory"
    ),
    verbose=True,
    handle_parsing_errors=True,
)

while True:
    query: str = input("\nEnter a query: ")

    res: str = robot_events_agent.run(query)

    print("Question:")
    print(query)
    print("Answer:")
    print(res)
