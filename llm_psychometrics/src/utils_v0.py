import itertools
import os
from openai import OpenAI
from anthropic import Anthropic
import json
from typing import List

# os.environ["OPENAI_API_KEY"] = "your-api-key"


def list_to_str(question_list: List):

    return " \n".join([f"{i+1}. {value}" for i, value in
                       enumerate(question_list)])


def inverse_likert(likert_values: List):

    inverted_likert = likert_values[::-1]

    return inverted_likert


def paraphrasing_prompt_template(question_list: List):

    questions = list_to_str(question_list)

    prompt_template = f"""
    Paraphrase the below mentioned questions. Use different words while keeping the meaning same 
    {questions}
    """
    return prompt_template


def persona_curation(base_text, job_persona):
    return f"""{base_text} who is a {job_persona['Summary']}.
                You are {job_persona['Age']} years old, based out of
                {job_persona['Location']}.
                You have a background as a {job_persona['Background']},
                you are {job_persona['Personality Traits']}
                and {job_persona['Style']}"""


def prompt_formatting(persona, base_prompt, questions):
    prompt = f"""{persona} \nTask: {base_prompt} \nQuestion: \n{questions}
                    \n Answer:"""
    return prompt


def generate_combinations(dict1, dict2):
    keys1, values1 = zip(*dict1.items())
    keys2, values2 = zip(*dict2.items())

    return [
        ({k: v for k, v in zip(keys1, combo1)},
         {k: v for k, v in zip(keys2, combo2)})
        for combo1 in itertools.product(*values1)
        for combo2 in itertools.product(*values2)
    ]


def statement_to_question_prompt(statement, n=10):

    prompt_template = f"""Given a statement, generate {n} real-life situation as a MCQ without indicating how the person would react to it. situations can be on different topic as long as they are similar.
                        Return only the question, not the multiple choice answers.
                        Respond ONLY in a strict python list.

                        ### Statement:
                        {statement}
    """
    return prompt_template


def question_to_mcq_prompt(generated_question, trait):

    prompt_template = f'''
                    User’s Question:
                    This is the user’s question. As an agent, please answer me 4 options you would recommend. 1. Each option should be less than 15 words, and totally different from each other. 2. Two options are plausible to be done with high {trait}, two options are plausible to be done with low {trait}.
                    Tag the answers and add an extra key in the dictionary with level of trait.
                    Respond ONLY in a strict python dictionary in the format:
                    {{"question": {generated_question},
                    "answers": List of answers,
                    "trait":{trait}}}.
                    ### Question:
                    {generated_question}
                    ### Options to Act: 1.
    '''
    return prompt_template


def openai_api_call(prompt, response_format, persona_prompt="You are a helpful assistant. Always respond with valid JSON. No explanations.", model="gpt-4.1-mini",
                    temperature=0.9, top_p=0.9, presence_penalty=0.4,
                    frequency_penalty=0.3):

    # Set your OpenAI API key
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # Prepare your messages in OpenAI's chat template format
    messages = [
        {"role": "system", "content": persona_prompt},
        {"role": "user", "content": prompt}
    ]

    # Call the Chat API
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        top_p=top_p,
        presence_penalty=presence_penalty,
        frequency_penalty=frequency_penalty,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "StructuredResponse",
                "schema": response_format.model_json_schema()
            }
        }
    )
    parsed = response.choices[0].message.content
    return response_format.model_validate_json(parsed)


def anthropic_api_call(prompt, response_format, persona_prompt="You are a helpful assistant. Always respond with valid JSON. No explanations.", model="claude-sonnet-4-5",
                    temperature=0.9, top_p=0.9, presence_penalty=0.4,
                    frequency_penalty=0.3):

    # Set your OpenAI API key
    client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    # Prepare your messages in OpenAI's chat template format
    messages = [
        # {"role": "system", "content": persona_prompt},
        {"role": "user", "content": prompt}
    ]

    # Call the Chat API
    response = client.beta.messages.parse(
        model=model,
        max_tokens=4096,
        betas=["structured-outputs-2025-11-13"],
        system=persona_prompt,
        messages=messages,
        output_format=response_format,
    )
    parsed = response.parsed_output
    return response_format.model_validate(parsed)


def generate_mcq_per_statement(statement, trait, persona_prompt="You are a helpful assistant. Always respond with valid JSON. No explanations.", n=10):

    mcq_questions = []

    statement_to_question_api_response = openai_api_call(
        persona_prompt,
        statement_to_question_prompt(statement, n=n))

    generated_questions = json.loads(statement_to_question_api_response)

    for question in generated_questions:
        mcq_question_openai_response = openai_api_call(
            persona_prompt,
            question_to_mcq_prompt(question, trait))
        try:
            question = json.loads(mcq_question_openai_response)
        except Exception:
            question = {}

        mcq_questions.append(question)

    return mcq_questions


def generate_mcq(statements, traits, n=10):
    MCQs = []
    for statement in statements:
        for trait in traits:
            questions = generate_mcq_per_statement(statement, trait, n)

            MCQs.append(questions)

    return MCQs
