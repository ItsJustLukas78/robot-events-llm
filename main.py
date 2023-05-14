import os
from typing import Any, List, Optional, Tuple, Union

from dotenv import load_dotenv
from langchain import OpenAI, SerpAPIWrapper
from langchain.agents import (
    AgentExecutor,
    AgentType,
    BaseMultiActionAgent,
    Tool,
    initialize_agent,
    load_tools,
)
from langchain.agents.agent_toolkits import NLAToolkit, OpenAPIToolkit
from langchain.agents.agent_toolkits.openapi.spec import reduce_openapi_spec
from langchain.chains import LLMChain, APIChain
from langchain.llms import OpenAI, OpenAIChat, GPT4All
from langchain.prompts import PromptTemplate
from langchain.requests import Requests, RequestsWrapper
from langchain.schema import AgentAction, AgentFinish
from langchain.tools import APIOperation, OpenAPISpec
from transformers import pipeline
import yaml

load_dotenv()

from langchain.agents.agent_toolkits.openapi import planner

# hf_model = pipeline(
#     model="databricks/dolly-v2-12b",
#     torch_dtype=torch.bfloat16,
#     trust_remote_code=True,
#     device_map="auto",
#     return_full_text=True,
# )

# openai_llm = OpenAIChat(temperature=0.0, model_name="gpt-3.5-turbo", verbose=True)
gpt4all_model = GPT4All(
    model="./models/ggml-gpt4all-j-v1.3-groovy.bin",
    n_ctx=512,
    n_threads=8,
    backend='gptj',
)

# requests = Requests(headers={
#     "Content-Type": "application/json",
#     "Accept": "application/json",
#     "Authorization": "Bearer " + os.getenv("ROBOT_EVENTS_API_KEY")
# })
#

headers = {
    "Content-Type": "application/json",
    "Accept": "application/json",
    "Authorization": "Bearer " + os.getenv("ROBOT_EVENTS_API_KEY")
}

requests_wrapper = RequestsWrapper(headers=headers)

with open("RE_documentation.yaml") as file:
    raw_RE_api_apec = yaml.load(file, Loader=yaml.Loader)
    RE_api_spec = reduce_openapi_spec(raw_RE_api_apec)

robot_events_agent = planner.create_openapi_agent(
    RE_api_spec,
    requests_wrapper,
    gpt4all_model,
    verbose=True,
)

# re_nl_toolkit = NLAToolkit.from_llm_and_url(
#     openai_llm,
#     "https://www.robotevents.com/api/v2/swagger.yml",
#     requests=requests,
#     verbose=True,
# )
#
# openapi_format_instructions = """
# Use the following format:
#
# Question: the input question you must answer
# Thought: you should always think about what to do
# Action: the action to take, should be one of [{tool_names}]
# Action Input: what to instruct the AI Action representative.
# Observation: The Agent's response
# ... (this Thought/Action/Action Input/Observation can repeat N times)
# Thought: I now know the final answer. User can't see any of my observations, API responses, links, or tools.
# Final Answer: the final answer to the original input question with the right amount of detail
#
# When responding with your Final Answer, remember that the person you are responding to
# CANNOT see any of your Thought/Action/Action Input/Observations, so if there is any
# relevant information there you need to include it explicitly in your response.
# """
#
# natural_language_tools = re_nl_toolkit.get_tools()
#
# mrkl = initialize_agent(
#     natural_language_tools,
#     openai_llm,
#     agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
#     verbose=True,
#     agent_kwargs={"format_instructions": openapi_format_instructions},
# )
#
user_input = "My team ID is 93452, can you list the last 3 events we attended?"
robot_events_agent.run(user_input)
# mrkl.run(user_input)
