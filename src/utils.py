import itertools


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
