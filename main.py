import os
import typing

import streamlit as st
from typing import Dict, List, TextIO

import yaml
import torch
from dotenv import load_dotenv
from langchain import OpenAI
from langchain.agents import load_tools, initialize_agent, AgentType
from langchain.agents.agent_toolkits import NLAToolkit, OpenAPIToolkit

from langchain.agents.agent_toolkits.openapi.spec import reduce_openapi_spec, ReducedOpenAPISpec
from langchain.chains import APIChain, LLMChain, OpenAPIEndpointChain
from langchain.chat_models import ChatOpenAI
from langchain.llms import GPT4All, OpenAI
from langchain.memory import (
    CombinedMemory,
    ConversationKGMemory,
    ConversationTokenBufferMemory,
    ConversationBufferMemory,
)
from langchain.requests import Requests, RequestsWrapper
from langchain.schema import AgentAction, AgentFinish
from langchain.tools import OpenAPISpec, APIOperation, AIPluginTool
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

openai_llm: ChatOpenAI = ChatOpenAI(temperature=0.2, model_name="gpt-3.5-turbo", verbose=True)
# gpt4all_model = GPT4All(
#     model="./models/ggml-gpt4all-j-v1.3-groovy.bin",
#     n_ctx=512,
#     n_threads=8,
#     backend="gptj",
# )

current_llm = openai_llm

# token_buffer_memory = ConversationTokenBufferMemory(
#     llm=current_llm, max_token_limit=200, memory_key="token_buffer", input_key="input"
# )
# buffer_memory = ConversationBufferMemory(
#     memory_key="buffer_memory", input_key="input"
# )
# kg_memory = ConversationKGMemory(
#     llm=current_llm, memory_key="kg_memory", input_key="input"
# )
#
# memory = CombinedMemory(memories=[buffer_memory, kg_memory])


headers: Dict = {
    "Content-Type": "application/json",
    "Accept": "application/json",
    "Authorization": "Bearer " + os.getenv("ROBOT_EVENTS_API_KEY"),
}

requests_wrapper: RequestsWrapper = RequestsWrapper(headers=headers)

with open("RE_documentation.yaml", "r") as file:
    raw_RE_api_apec: Dict = yaml.load(file, Loader=yaml.Loader)
    RE_api_spec: ReducedOpenAPISpec = reduce_openapi_spec(raw_RE_api_apec)

spec: OpenAPISpec = OpenAPISpec.from_file("RE_documentation.yaml")

robot_events_agent = planner.create_openapi_agent(
    RE_api_spec,
    requests_wrapper,
    current_llm,
    shared_memory=ConversationBufferMemory(input_key="input", memory_key="buffer_memory"),
    verbose=True,
    handle_parsing_errors=True,
)


# tool: AIPluginTool = AIPluginTool.from_plugin_url("https://roboteventsplugin-langchain.langdock.com/.well-known/ai-plugin.json")
#
# tools: List = load_tools(["requests_all"])
# tools += [tool]
#
# agent_chain = initialize_agent(tools, current_llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

#
# teams_operation = APIOperation.from_openapi_spec(spec, '/teams', "get")
#
# chain = OpenAPIEndpointChain.from_api_operation(
#     teams_operation,
#     current_llm,
#     requests=requests_wrapper,
#     verbose=True,
#     return_intermediate_steps=True,
# )

while True:
    query: str = input("\nEnter a query: ")

    # res: str = robot_events_agent.run(query)
    res: str = robot_events_agent.run(query)

    # Print the result
    print("Question:")
    print(query)
    print("Answer:")
    print(res)

# query = st.text_input("\nEnter a query: ")
#
# while True:
#     if query:
#         res = robot_events_agent.run(query)
#
#         # Print the result
#         st.text("Question:")
#         print(query)
#         st.text("Answer:")
#         print(res)
