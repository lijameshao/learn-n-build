"""
Custom chain:
    input_key:
        - query
    output:
        - base_response
Agent - search + verify:
    input_key:
        - base_response
    output_key:
        - output

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


"""Generate baseline response
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

Experience
- It is hard to generate expected verification questions
- There is a lot of room to interpret what is "common sense" and doesn't need verification (depending on the user's level)
- Final response it's difficult to use Factored model in GPT-3.5

Authors use Llama 65B model

"""


import os
import langchain

langchain.debug = True

from langchain import OpenAI, PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import SequentialChain, LLMChain
from langchain.output_parsers import PydanticOutputParser
from langchain.pydantic_v1 import BaseModel, Field


os.environ["OPENAI_API_KEY"] = ""
llm = OpenAI(temperature=0, model_name="gpt-3.5-turbo-instruct") # GPT-3.5 tends to work less well in generating a consistent final response
# llm = ChatOpenAI(temperature=0, model_name="gpt-4")


query = "List 5 US politicians born in New York"

input_variables = ["query"]
base_response_output_key = "base_response"
base_response_template = """Question: {query} Answer:"""
base_repsonse_prompt_template = PromptTemplate(
    input_variables=input_variables, template=base_response_template
)

base_response_chain = LLMChain(
    llm=llm, prompt=base_repsonse_prompt_template, output_key=base_response_output_key
)

plan_verifications_template = """
Given the below Question and answer, generate a series of verification questions that test the factual claims in the original baseline response.
For example if part of a longform model response contains the statement “The Mexican–American War
was an armed conflict between the United States and Mexico from 1846 to 1848”, then one possible
verification question to check those dates could be “When did the Mexican American war start and
end?”
Another example, if the question asks name some politicians born in London, UK; and if the answers are: Boris Johnson, Theresa May, Winston Churchill,
then the verification question to check could be "Where was Boris Johnson born"?

Question: {query}
Answer: {base_response}

<fact in passage>, <verification question, generated by combining the query and the fact>

{format_instructions}
"""


class PlanVerificationsOutput(BaseModel):
    query: str = Field(description="The user's query")
    base_response: str = Field(description="The response to the user's query")
    facts_and_verification_questions: dict[str, str] = Field(
        description="Facts (as the dictionary keys) extracted from the response and verification questions related to the query (as the dictionary values)"
    )


plan_verifications_output_parser = PydanticOutputParser(
    pydantic_object=PlanVerificationsOutput
)

plan_verifications_prompt_template = PromptTemplate(
    input_variables=input_variables + [base_response_output_key],
    template=plan_verifications_template,
    partial_variables={
        "format_instructions": plan_verifications_output_parser.get_format_instructions()
    },
)
plan_verifications_chain = LLMChain(
    llm=llm,
    prompt=plan_verifications_prompt_template,
    output_key="output",
    output_parser=plan_verifications_output_parser,
)


answer_and_plan_verification = SequentialChain(
    chains=[base_response_chain, plan_verifications_chain],
    input_variables=["query"],
    output_variables=["output"],
    verbose=True,
)


intermediate_result = answer_and_plan_verification.run(query)


# Execute verifications
claimed_facts = list(intermediate_result.facts_and_verification_questions.keys())
verification_questions = list(
    intermediate_result.facts_and_verification_questions.values()
)

# TODO: Chain with search agent


# =============================================================================
# Answer each question independently
# =============================================================================
verify_results = []
verify_results_str = ""
verify_input_variables = ["question"]
verify_output_key = "answer"
verify_template = """{question}"""

verify_prompt_template = PromptTemplate(
    input_variables=verify_input_variables, template=verify_template
)

verify_chain = LLMChain(
    llm=llm, prompt=verify_prompt_template, output_key=verify_output_key
)
for i in range(len(verification_questions)):
    claimed_fact = claimed_facts[i]
    question = verification_questions[i]
    answer = verify_chain.run(question)
    answer = answer.lstrip("\n")
    verify_results.append(answer)
    verify_results_str += f"Question: {question}\nAnswer: {answer}\n\n"

# =============================================================================
# Answer all in one go (less preferred)
# =============================================================================
# How to input list as an input variable?
verify_input_variables = ["questions"]
verify_output_key = "answers"
verify_template = """
Answer each question in the list, keeping the same order.
{questions}

{format_instructions}
"""


class VerifyOutput(BaseModel):
    questions_and_answers: dict[str, str] = Field(
        description="Question (as the dictionary keys) and its answer(as the dictionary values)"
    )


verify_output_parser = PydanticOutputParser(pydantic_object=VerifyOutput)

verify_prompt_template = PromptTemplate(
    input_variables=verify_input_variables,
    template=verify_template,
    partial_variables={
        "format_instructions": verify_output_parser.get_format_instructions()
    },
)

verify_chain = LLMChain(
    llm=llm,
    prompt=verify_prompt_template,
    output_key=verify_output_key,
    output_parser=verify_output_parser,
)
verify_results = verify_chain.run(str(verification_questions))


# Generate final verified response
# Input: Original question, original answer
# Verification questions & answers
# Outputs: revised passages (if needed)
# Given the discovered inconsistencies (if any), generate a revised response incorporating the verification results.


def concat_verify_results(verify_results: VerifyOutput) -> str:
    concatenated_string = ""
    for question, answer in verify_results.questions_and_answers.items():
        concatenated_string += f"Question: {question}\nAnswer: {answer}\n"
    return concatenated_string


verify_results_str = concat_verify_results(verify_results)


# =============================================================================
# Final response (Factor)
# =============================================================================
final_response_input_variables = ["query", "base_response", "verify_results"]
final_response_template = """Given the ORIGINAL_QUESTION and the ORIGINAL_RESPONSE,
revise the ORIGINAL_RESPONSE (if applicable) such that it is consistent with information in VERIFIED_SOURCE.
Only keep consistent information.

<ORIGINAL_QUESTION>
{query}

<ORIGINAL_RESPONSE>
{base_response}

<VERIFIED_SOURCE>
{verify_results}

Final response:
"""
final_response_prompt_template = PromptTemplate(
    input_variables=final_response_input_variables,
    template=final_response_template,
)

final_response_chain = LLMChain(llm=llm, prompt=final_response_prompt_template)

final_response = final_response_chain.run(
    query=intermediate_result.query,
    base_response=intermediate_result.base_response,
    verify_results=verify_results_str,
)


# =============================================================================
# TODO: Final response (Factor + revise)
# =============================================================================

