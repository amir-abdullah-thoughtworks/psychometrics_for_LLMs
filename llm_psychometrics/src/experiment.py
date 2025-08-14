import random
from typing import List
from src.utils import prompt_formatting, persona_curation, paraphrasing_prompt_template, inverse_likert, list_to_str


def paraphrasing_questions(question_list: List, pipeline, tokenizer,
                           pipeline_config):

    prompt = paraphrasing_prompt_template(question_list)

    chat = [
        {"role": "user", "content": prompt}
    ]
    prompt = tokenizer.apply_chat_template(
        chat, tokenize=False, add_generation_prompt=True)

    return pipeline(prompt, return_full_text=False, **pipeline_config)


def text_generate(prompts, pipeline, pipeline_config, generation_config):

    print(f"No of prompts: {len(prompts)}")

    if generation_config["generation_type"] == "one_at_time":
        outputs = []
        for prompt in prompts:
            outputs.append(
                pipeline(prompt, return_full_text=False, **pipeline_config)[0])
    else:
        outputs = pipeline(prompts, return_full_text=False, **pipeline_config)

    return outputs


def experiment_setup(job_title,
                     question_list,
                     paraphrased_question_list,
                     config_combinations,
                     pipeline,
                     base_prompt_config,
                     persona_config,
                     likert_scale):

    def chat_formatting(question, base_text, persona, base_prompt):
        raw_prompt = prompt_formatting(
            persona_curation(base_text, persona), base_prompt,
            question)
        chat = [
            {"role": "user", "content": raw_prompt}
        ]
        chat_prompt = tokenizer.apply_chat_template(
            chat, tokenize=False, add_generation_prompt=True)

        return raw_prompt, chat_prompt

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
            text_generation_output_dict = {}
            print(f"Config No: {n}")
            pipeline_config, generation_config = config

            if generation_config['reasoning']:
                base_prompt = base_prompt_config["reasoning_base_prompt"]
            else:
                base_prompt = base_prompt_config["non_reasoning_base_prompt"]

            print(f"Generation Type: {generation_config['generation_type']}")

            normal_likert_base_prompt = base_prompt.replace(
                "<LIKERT_SCALE>", "\n" + list_to_str(likert_scale))
            inverse_likert_base_prompt = base_prompt.replace(
                "<LIKERT_SCALE>", "\n" + list_to_str(inverse_likert(likert_scale)))

            if generation_config["shuffle"]:
                shuffled_questions = question_list.copy()
                random.shuffle(question_list)
                indices = [question_list.index(item)
                           for item in shuffled_questions]
                paraphrased_question_list = paraphrased_question_list[indices]
            else:
                print(f"No of Questions: {len(question_list)}")
                indices = list(range(len(question_list)))

            if generation_config['generation_type'] == "one_at_time":
                normal_chat_prompts = []
                normal_likert_prompts = []
                inverse_paraphrase_chat_prompts = []
                inverse_paraphrase_prompts = []
                normal_paraphrase_chat_prompts = []
                normal_paraphrase_prompts = []
                inverse_likert_chat_prompts = []
                inverse_likert_prompts = []

                for question, paraphrased_question in zip(question_list, paraphrased_question_list):

                    raw_prompt, chat_prompt = chat_formatting(
                        question, base_text, persona, normal_likert_base_prompt)

                    paraphrased_raw_prompt, paraphrased_chat_prompt = chat_formatting(
                        paraphrased_question, base_text, persona, normal_likert_base_prompt)

                    raw_inverse_likert_prompt, chat_inverse_likert_prompt = chat_formatting(
                        question, base_text, persona, inverse_likert_base_prompt)

                    paraphrased_raw_inverse_likert_prompt, paraphrased_chat_inverse_likert_prompt = chat_formatting(
                        paraphrased_question, base_text, persona, inverse_likert_base_prompt)

                    normal_chat_prompts.append(chat_prompt)
                    normal_likert_prompts.append(raw_prompt)
                    normal_paraphrase_prompts.append(paraphrased_raw_prompt)
                    normal_paraphrase_chat_prompts.append(
                        paraphrased_chat_prompt)
                    inverse_likert_chat_prompts.append(
                        chat_inverse_likert_prompt)
                    inverse_likert_prompts.append(raw_inverse_likert_prompt)
                    inverse_paraphrase_chat_prompts.append(
                        paraphrased_chat_inverse_likert_prompt)
                    inverse_paraphrase_prompts.append(
                        paraphrased_raw_inverse_likert_prompt)
            else:
                questions = list_to_str(question_list)
                paraphrased_questions = list_to_str(paraphrased_question_list)

                normal_raw_prompt, normal_chat_prompt = chat_formatting(
                    questions, base_text, persona, normal_likert_base_prompt)

                normal_paraphrased_raw_prompt, normal_paraphrased_chat_prompt = chat_formatting(
                    paraphrased_questions, base_text, persona, normal_likert_base_prompt)

                normal_invert_likert_raw_prompt, normal_invert_likert_chat_prompt = chat_formatting(
                    questions, base_text, persona, inverse_likert_base_prompt)

                paraphrased_invert_likert_raw_prompt, paraphrased_invert_likert_chat_prompt = chat_formatting(
                    paraphrased_questions, base_text, persona, inverse_likert_base_prompt)

                normal_chat_prompts = [normal_chat_prompt]
                normal_likert_prompts = [normal_raw_prompt]
                normal_paraphrase_chat_prompts = [
                    normal_paraphrased_chat_prompt]
                normal_paraphrase_prompts = [normal_paraphrased_raw_prompt]
                inverse_likert_chat_prompts = [
                    normal_invert_likert_chat_prompt]
                inverse_likert_prompts = [normal_invert_likert_raw_prompt]
                inverse_paraphrase_chat_prompts = [
                    paraphrased_invert_likert_chat_prompt]
                inverse_paraphrase_prompts = [
                    paraphrased_invert_likert_raw_prompt]

            print("##############################################")
            print("\n")
            print("Generating Output for original prompts with normal likert scale")
            print("\n")
            normal_output_dict = text_generate(
                normal_chat_prompts, pipeline, pipeline_config, generation_config)

            print("##############################################")
            print("\n")
            print("Generating Output for paraphrased prompts with normal likert scale")
            print("\n")
            paraphrased_output_dict = text_generate(
                normal_paraphrase_chat_prompts, pipeline, pipeline_config, generation_config)

            print("##############################################")
            print("\n")
            print("Generating Output for original prompts with invserse likert scale")
            print("\n")
            inverse_likert_output_dict = text_generate(
                inverse_likert_chat_prompts, pipeline, pipeline_config, generation_config)

            print("##############################################")
            print("\n")
            print("Generating Output for paraphrased prompts with invserse likert scale")
            print("\n")
            inverse_likert_paraphrased_output_dict = text_generate(
                inverse_paraphrase_chat_prompts, pipeline, pipeline_config, generation_config)

            text_generation_output_dict['generation_config'] = generation_config
            text_generation_output_dict['pipeline_config'] = pipeline_config
            text_generation_output_dict['indices'] = indices
            text_generation_output_dict['normal_likert'] = {}
            text_generation_output_dict['normal_likert']['prompts'] = normal_likert_prompts
            text_generation_output_dict['normal_likert']['output'] = normal_output_dict
            text_generation_output_dict['paraphrased_likert'] = {}
            text_generation_output_dict['paraphrased_likert']['prompts'] = normal_paraphrase_prompts
            text_generation_output_dict['paraphrased_likert']['output'] = paraphrased_output_dict
            text_generation_output_dict['normal_inverse_likert'] = {}
            text_generation_output_dict['normal_inverse_likert']['prompts'] = inverse_likert_prompts
            text_generation_output_dict['normal_inverse_likert']['output'] = inverse_likert_output_dict
            text_generation_output_dict['paraphrased_inverse_likert'] = {}
            text_generation_output_dict['paraphrased_inverse_likert']['prompts'] = inverse_paraphrase_prompts
            text_generation_output_dict['paraphrased_inverse_likert']['output'] = inverse_likert_paraphrased_output_dict

            persona_result.append(text_generation_output_dict)

        experiment_results[persona['Summary']] = persona_result

    return experiment_results
