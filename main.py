import os

import yaml
import torch
from dotenv import load_dotenv
from langchain import OpenAI
from langchain.agents.agent_toolkits import NLAToolkit, OpenAPIToolkit
from langchain.agents.agent_toolkits.openapi.spec import reduce_openapi_spec
from langchain.chains import APIChain, LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.llms import GPT4All, OpenAI
from langchain.memory import (
    CombinedMemory,
    ConversationKGMemory,
    ConversationTokenBufferMemory,
)
from langchain.requests import Requests, RequestsWrapper
from langchain.schema import AgentAction, AgentFinish
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

openai_llm = ChatOpenAI(temperature=0.2, model_name="gpt-3.5-turbo", verbose=True)
# gpt4all_model = GPT4All(
#     model="./models/ggml-gpt4all-j-v1.3-groovy.bin",
#     n_ctx=512,
#     n_threads=8,
#     backend="gptj",
# )

current_llm = openai_llm

token_buffer_memory = ConversationTokenBufferMemory(
    llm=current_llm, max_token_limit=200, memory_key="token_buffer", input_key="input"
)
kg_memory = ConversationKGMemory(
    llm=current_llm, memory_key="kg_memory", input_key="input"
)

memory = CombinedMemory(memories=[token_buffer_memory, kg_memory])

headers = {
    "Content-Type": "application/json",
    "Accept": "application/json",
    "Authorization": "Bearer " + os.getenv("ROBOT_EVENTS_API_KEY"),
}

requests_wrapper = RequestsWrapper(headers=headers)

with open("RE_documentation.yaml") as file:
    raw_RE_api_apec = yaml.load(file, Loader=yaml.Loader)
    RE_api_spec = reduce_openapi_spec(raw_RE_api_apec)

robot_events_agent = planner.create_openapi_agent(
    RE_api_spec,
    requests_wrapper,
    current_llm,
    shared_memory=memory,
    verbose=True,
)

while True:
    query = input("\nEnter a query: ")
    if query == "exit":
        break

    # Get the answer from the chain
    res = robot_events_agent.run(query)

    # Print the result
    print("\n\n> Question:")
    print(query)
    print("\n> Answer:")
    print(res)
