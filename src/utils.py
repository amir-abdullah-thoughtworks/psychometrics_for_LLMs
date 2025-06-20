import itertools
import os
import openai

os.environ["OPENAI_API_KEY"] = "your-api-key"


def persona_curation(base_text, job_persona):
    return f"""{base_text} who is a {job_persona['Summary']}.
                You are {job_persona['Age']} years old, based out of
                {job_persona['Location']}.
                You have a background as a {job_persona['Background']},
                you are {job_persona['Personality Traits']}
                and {job_persona['Style']}"""


def prompt_formatting(persona, base_prompt, questions):
    prompt = f"""{persona} \n Task: {base_prompt} \n Question: \n {questions}
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


def statement_to_question_prompt(statement):

    prompt_template = f"""
                        Given a statement, generate 10 real-life situation as a MCQ without indicating how the person would react to it. situations can be on different topic as long as they are similar.
                        Return only the question, not the multiple choice answers.
                        Respond ONLY in a strict python list.

                        ### Statement:
                        {statement}
    """
    return prompt_template


def question_to_mcq_prompt(generated_question, trait):

    prompt_template = f"""
                    User’s Question:
                    This is the user’s question. As an agent, please answer me 4 options you would recommend. 1. Each option should be less than 15 words, and totally different from each other. 2. Two options are plausible to be done with high {trait}, two options are plausible to be done with low {trait}.
                    Respond ONLY in a strict python list.
                    ### Question:
                    {generated_question}
                    ### Options to Act: 1.
    """
    return prompt_template


def openai_api_call(persona_prompt, prompt, model="gpt-4"):

    # Set your OpenAI API key
    openai.api_key = os.getenv("OPENAI_API_KEY")

    # Prepare your messages in OpenAI's chat template format
    messages = [
        {"role": "system", "content": persona_prompt},
        {"role": "user", "content": prompt}
    ]

    # Call the Chat API
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0.7,
        max_tokens=500
    )
    return response['choices'][0]['message']['content']
