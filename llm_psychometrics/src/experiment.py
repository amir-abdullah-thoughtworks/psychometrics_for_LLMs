import random
from src.utils import prompt_formatting, persona_curation


def text_generate(prompts, pipeline, pipeline_config, generation_config):
    if generation_config["shuffle"]:
        shuffled_prompts = prompts.copy()
        random.shuffle(prompts)
        indices = [prompts.index(item) for item in shuffled_prompts]
    else:
        print(f"No of prompts: {len(prompts)}")
        indices = list(range(len(prompts)))

    if generation_config["generation_type"] == "one_at_time":
        outputs = []
        for prompt in prompts:
            outputs.append(
                pipeline(prompt, return_full_text=False, **pipeline_config)[0])
    else:
        outputs = pipeline(prompts, return_full_text=False, **pipeline_config)

    output_dict = {
        "indices": indices,
        "outputs": outputs,
        "prompts": prompts
    }

    return output_dict


def experiment_setup(job_title, question_list, config_combinations, pipeline,
                     base_prompt_config, persona_config):

    tokenizer = base_prompt_config['tokenizer']
    experiment_results = {}
    print(f"Running Experiment for: {job_title}")
    base_text = persona_config[job_title]['base_text']
    personas = persona_config[job_title]['personas']

    print(f"Total no of config combinations: {len(config_combinations)}")
    print(f"Total no of personas: {len(personas)}")

    for persona in personas:
        print(f"Persona: {persona['Summary']}")
        persona_result = []
        for n, config in enumerate(config_combinations):
            print(f"Config No: {n}")
            pipeline_config, generation_config = config

            if generation_config['reasoning']:
                base_prompt = base_prompt_config["reasoning_base_prompt"]
            else:
                base_prompt = base_prompt_config["non_reasoning_base_prompt"]

            print(f"Generation Type: {generation_config['generation_type']}")

            if generation_config['generation_type'] == "one_at_time":
                prompts = []
                base_prompts = []

                for question in question_list:

                    chat = [
                        {"role": "user", "content": prompt_formatting(
                            persona_curation(base_text, persona), base_prompt,
                            question)}
                    ]
                    prompt = tokenizer.apply_chat_template(
                        chat, tokenize=False, add_generation_prompt=True)
                    prompts.append(prompt)
                    base_prompts.append(prompt_formatting(
                        persona_curation(base_text, persona), base_prompt,
                        question))

            else:
                questions = "\n ".join(
                    [f"{str(i+1)}. {question}" for i, question in
                     enumerate(question_list)])
                chat = [
                    {"role": "user", "content": prompt_formatting(
                        persona_curation(base_text, persona), base_prompt,
                        questions)}
                ]
                prompts = [tokenizer.apply_chat_template(
                    chat, tokenize=False, add_generation_prompt=True)]
                base_prompts = [prompt_formatting(
                    persona_curation(base_text, persona), base_prompt,
                    questions)]

            text_generation_output_dict = text_generate(
                prompts, pipeline, pipeline_config, generation_config)

            text_generation_output_dict['generation_config'] = generation_config
            text_generation_output_dict['pipeline_config'] = pipeline_config
            text_generation_output_dict['base_prompts'] = base_prompts

            persona_result.append(text_generation_output_dict)

        experiment_results[persona['Summary']] = persona_result

    return experiment_results
