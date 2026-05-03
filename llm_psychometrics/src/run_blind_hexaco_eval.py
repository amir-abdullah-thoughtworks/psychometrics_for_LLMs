"""
Blind HEXACO trait detection evaluation.
Presents SJT options WITHOUT trait labels and checks if judge correctly identifies each trait.
"""
import json, os, sys
from jinja2 import Template
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils_v0 import list_to_str, openai_api_call, anthropic_api_call
from src.evals.sjt_evaluation_llm_judge_v0 import (
    SjtLLMWithoutSeedsJudge,
    SJT_LLM_JUDGE_EVALUATION_WITHOUT_SEEDS_TEMPLATE,
    sjt_answer_default_order,
)
from datasets import load_dataset

TRUE_TRAITS = [
    'Honesty-Humility', 'Emotionality', 'Extraversion',
    'Agreeableness', 'Conscientiousness', 'Openness to Experience'
]
OPTION_KEYS = ['first_option', 'second_option', 'third_option',
               'fourth_option', 'fifth_option', 'sixth_option']


def run_blind_eval(sjt_dict, model_provider):
    sjt = sjt_dict['corrected_sjt']
    config_dict = {
        'question': sjt['question'],
        'answer_options': list_to_str([
            sjt[key] for key in sjt_answer_default_order if key in sjt and "_option" in key
        ])
    }
    prompt = SJT_LLM_JUDGE_EVALUATION_WITHOUT_SEEDS_TEMPLATE.render(config_dict)

    if model_provider == 'openai':
        response = openai_api_call(
            prompt=prompt, response_format=SjtLLMWithoutSeedsJudge,
            model="gpt-4o", temperature=0, top_p=1, presence_penalty=0, frequency_penalty=0
        )
    else:
        response = anthropic_api_call(
            prompt=prompt, response_format=SjtLLMWithoutSeedsJudge,
            model="claude-sonnet-4-6", temperature=0, top_p=1, presence_penalty=0, frequency_penalty=0
        )
    return response.model_dump()


def compute_accuracy(results):
    correct, total = 0, 0
    per_trait = {t: {'correct': 0, 'total': 0} for t in TRUE_TRAITS}
    for r in results:
        hexaco = r['hexaco_traits']
        for opt_key, true_trait in zip(OPTION_KEYS, TRUE_TRAITS):
            if opt_key not in hexaco:
                continue
            values = hexaco[opt_key].get('values', {})
            if not values:
                continue
            predicted = max(values, key=values.get)
            per_trait[true_trait]['total'] += 1
            total += 1
            if predicted == true_trait:
                per_trait[true_trait]['correct'] += 1
                correct += 1
    return correct, total, per_trait


if __name__ == "__main__":
    print("Loading SJTs from HuggingFace...")
    ds = load_dataset("thoughtworks/psychometric_SJTs")
    sjt_list = ds['train'].to_list()[:500]
    print(f"Loaded {len(sjt_list)} SJTs")

    # Normalize structure: HF dataset may have flat fields instead of corrected_sjt
    for sjt in sjt_list:
        if 'corrected_sjt' not in sjt:
            sjt['corrected_sjt'] = {
                'question': sjt['question'],
                **{k: sjt[k] for k in sjt_answer_default_order if k in sjt}
            }

    for provider in ['openai', 'anthropic']:
        print(f"\n--- Running blind eval with {provider} ---")
        results = []
        with ThreadPoolExecutor(max_workers=20) as executor:
            futures = {executor.submit(run_blind_eval, sjt, provider): sjt for sjt in sjt_list}
            for future in tqdm(futures, desc=provider, total=len(futures)):
                try:
                    results.append(future.result())
                except Exception as e:
                    print(f"Error: {e}")

        correct, total, per_trait = compute_accuracy(results)
        print(f"Overall: {correct}/{total} = {correct/total:.3f}")
        for t in TRUE_TRAITS:
            c = per_trait[t]['correct']
            n = per_trait[t]['total']
            if n > 0:
                print(f"  {t}: {c}/{n} = {c/n:.3f}")

        out_path = f"/tmp/blind_hexaco_eval_{provider}_n500.json"
        with open(out_path, 'w') as f:
            json.dump(results, f)
        print(f"Saved to {out_path}")
