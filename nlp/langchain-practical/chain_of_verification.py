"""

https://arxiv.org/abs/2309.11495

Generate baseline response
Q: Tell me a bio of <person>
A: <bio of person>

Plan verifications
Context: Q: Tell me a bio of <person>.
A: <passage about person>
Response:
<fact in passage>, Verification Question
<fact in passage>, Verification Question


Execute verifications
Q: Verification Question
A: Answer

Generate final verified response
Context: <Original Passage>.
From another source,
<output of execute verification step: Q + A>
<output of execute verification step: Q + A>
Response: <revised and consistent Passage>

Factor + revise: Identify which facts are consistent
Context: <Original Fact>.
From another source,
<output of execute verification step: Q + A>
Response: CONSISTENT. <Consistent fact>
Context: <Original Fact>.
From another source,
<output of execute verification step: Q + A>
Response: INCONSISTENT.
Context: <Original Fact>.
From another source,
<output of execute verification step: Q + A>
Response: PARTIALLY CONSISTENT. <Consistent part>

"""

import os
import langchain

langchain.debug = True

from langchain import OpenAI, PromptTemplate
from langchain.chains import SequentialChain, LLMChain
from langchain.output_parsers import PydanticOutputParser
from langchain.pydantic_v1 import BaseModel, Field


os.environ["OPENAI_API_KEY"] = ""
llm = OpenAI(temperature=0, model_name="gpt-3.5-turbo-instruct")


query = "Give me 2 quotes from Naval Ravikant"

input_variables = ["query"]
base_response_output_key = "base_response"
base_response_template = """Question: {query} Answer:"""
base_repsonse_prompt_template = PromptTemplate(
    input_variables=input_variables, template=base_response_template
)

base_response_chain = LLMChain(
    llm=llm, prompt=base_repsonse_prompt_template, output_key=base_response_output_key
)
