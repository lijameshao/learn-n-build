"""
General workflow:

1. Draw out the workflow, specifying the input and output at each stage
2. Create separate chains for individual components
3. Test each subchain separately before combining
4. Combine

https://medium.com/@james.li/mental-model-to-building-chains-with-langchain-expression-language-lcel-with-branching-and-36f185134eac
"""

import os
import getpass

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_openai import ChatOpenAI
from langchain.output_parsers import CommaSeparatedListOutputParser


os.environ["OPENAI_API_KEY"] = getpass.getpass("OpenAI API key:")

""" 
========================================================
Example 1:
Split into 2 branches and merge
========================================================
"""

prompt = ChatPromptTemplate.from_template("{question}")
combine_answers_prompt = ChatPromptTemplate.from_template(
    """Given the question: {question} and the below two answers:
Answer 1:
{answer1}

Answer 2:
{answer2}

Combine the viewpoints of two answers and form a coherent combined answer.
"""
)

# 2. Create separate chains for individual components
model1 = ChatOpenAI(temperature=0.5)
model2 = ChatOpenAI(temperature=1)
model3 = ChatOpenAI(temperature=0)

chain1 = prompt | model1 | StrOutputParser()
chain2 = prompt | model2 | StrOutputParser()
chain3 = combine_answers_prompt | model3 | StrOutputParser()


# 3. Test each subchain separately
question = "What's the best way to stay up to date with latest Large Language Model news? Please keep the answer short and concise, limit to 3 bullet points."
answer1_output = chain1.invoke(question)
answer2_output = chain2.invoke(question)
combined_answer_output = chain3.invoke(
    {"question": question, "answer1": answer1_output, "answer2": answer2_output}
)

# 4. Combine
combined_chain = {
    "question": RunnablePassthrough(),
    "answer1": chain1,
    "answer2": chain2,
} | chain3
combined_chain.invoke(question)

# This is equivalent to piping a single dictionary into chain3
combined_chain = (
    {"question": RunnablePassthrough()}
    | RunnablePassthrough.assign(answer1=chain1)
    | RunnablePassthrough.assign(answer2=chain2)
    | chain3
)

""" 
========================================================
Example 2:
Split into n branches and merge
========================================================
"""


def list_to_numbered_string(lst: list[str]) -> str:
    """
    Returns a numbered list string

    For example:
    ["a", "b", "c"] returns
    1. a
    2. b
    3. c
    """
    combined_string = ""
    for idx, item in enumerate(lst):
        if idx != 0:
            combined_string += "\n"
        combined_string += f"{idx+1}. {item}\n\n"
    return combined_string


model = ChatOpenAI(temperature=0)

# 2. Create separate chains for individual components

list_output_parser = CommaSeparatedListOutputParser()
format_instructions = list_output_parser.get_format_instructions()
answers_prompt = ChatPromptTemplate.from_template(
    "Please answer the question in 3 bullet points: {question}.\n{format_instructions}",
    partial_variables={"format_instructions": format_instructions},
)
answers_chain = answers_prompt | model | list_output_parser


modify_answer_prompt = ChatPromptTemplate.from_template(
    "Given the question: {question}, and the following statement: {answer}. Provide a pro and a con to the statement."
)
modify_answer_chain = (
    (
        lambda x: [
            {"question": x["question"], "answer": answer} for answer in x["answers"]
        ]
    )
    | (modify_answer_prompt | model | StrOutputParser()).map()
) | RunnableLambda(list_to_numbered_string)


combine_answers_prompt = ChatPromptTemplate.from_template(
    """Given the question: {question} and the below answers:
Answers:
{modified_answers}

Combine the viewpoints of the answers and form a coherent final answer, giving evidence for the pros and cons.
"""
)
combine_answers_chain = combine_answers_prompt | model | StrOutputParser()

# 3. Test each subchain separately
question = "How to get on the housing ladder?"
answers = answers_chain.invoke({"question": question})
print(answers)
modified_answers = modify_answer_chain.invoke(
    {"question": question, "answers": answers}
)
print(modified_answers)
combined_answer = combine_answers_chain.invoke(
    {"question": question, "answers": answers, "modified_answers": modified_answers}
)
print(combined_answer)

# 4. Combine
whole_chain = (
    {"question": RunnablePassthrough()}
    | RunnablePassthrough.assign(answers=answers_chain)
    | RunnablePassthrough.assign(modified_answers=modify_answer_chain)
    | combine_answers_chain
)
whole_chain_answer = whole_chain.invoke(question)
print(whole_chain_answer)
