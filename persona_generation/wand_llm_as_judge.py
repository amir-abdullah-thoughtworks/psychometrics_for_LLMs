# Import required libraries
from clize import run  # CLI argument parsing
import json
import pandas as pd
from pydantic import BaseModel, Field  # Data validation and schema definition
from typing import List, Optional, Literal, Dict
from datetime import datetime, timedelta
from pydantic import BaseModel
from lmformatenforcer import JsonSchemaParser  # Enforce JSON schema for LLM outputs
from openai import OpenAI
from lmformatenforcer.integrations.transformers import build_transformers_prefix_allowed_tokens_fn
import pprint
import random
from fundus import PublisherCollection, Crawler
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig  # Hugging Face transformers
import pickle
import wandb  # Weights & Biases for experiment tracking
import torch
import numpy as np
from torch.nn import CrossEntropyLoss

def main(
    conversations_file='slack_conversations_eight_day.csv',
    openai_api_key=None,
    persona_file='persona.jsonl',
    company_file='company.jsonl',
    model_id="meta-llama/Meta-Llama-3.1-70B-Instruct",
    lora_base_path="/home/lain/lora-",
    wandb_project="llm-as-judge-lora",
    wandb_run_name="experiment-1", 
    output_file='llm_as_judge2_results.pkl',
    judge_model="gpt-4o",
    temperature=2.0,
    min_p=0.2,
    max_tokens=200,
    num_samples=10,
    company_filter="ZenMaster"
):
    """
    Run an experiment evaluating LoRA-adapted language models against base models using an LLM judge.
    
    :param conversations_file: Path to CSV file containing conversation data
    :param openai_api_key: OpenAI API key
    :param persona_file: Path to JSONL file containing persona data
    :param company_file: Path to JSONL file containing company data
    :param model_id: Hugging Face model ID to use
    :param lora_base_path: Base path where LoRA adapters are stored
    :param wandb_project: Weights & Biases project name
    :param wandb_run_name: Weights & Biases run name
    :param output_file: Path to save results pickle file
    :param judge_model: OpenAI model to use as judge
    :param temperature: Sampling temperature
    :param min_p: Minimum probability for sampling
    :param max_tokens: Maximum tokens to generate
    :param num_samples: Number of samples per adapter
    :param company_filter: Company name to filter personas by
    """

    # Hard-coded values
    lora_ranks = [4, 8, 16, 32]
    sample_names = ["Hannelore Klose", "Kai Wagner", "Leopold Morgenstern", "Rohini Desai"]

    # Initialize OpenAI client for LLM judge
    client = OpenAI(api_key=openai_api_key)

    # Load conversation data from CSV and remove any rows with missing values
    df = pd.read_csv(conversations_file, sep='|')
    df = df.dropna()

    # Create dictionary of responses grouped by day
    day_responses = {
        f"day{i}": df[f"day{i}"].tolist() 
        for i in range(1, 9)
    }

    # Helper function to load and parse JSONL files containing persona/company data
    def load_jsonl(file_path):
        with open(file_path, 'r') as f:
            return [json.loads(json.loads(line)) for line in f]
            
    list_of_data = load_jsonl(persona_file)
    list_of_company_data = load_jsonl(company_file)

    def extract_named_examples(name, examples):
        """
        Extract text examples associated with a specific name from conversation data.
        Uses pattern matching to find text between name markers.
        """
        named_examples = []
        for example in examples:
            position = 0
            while position < len(example):
                name_pattern = name + ":"
                idx = example.find(name_pattern, position)
                if idx == -1:
                    break
                    
                next_position = idx + len(name_pattern)
                next_name_idx = -1
                
                # Look for start of next name (capital letter followed by lowercase)
                for i in range(next_position, len(example)):
                    if i+1 < len(example) and example[i].isupper() and example[i+1].islower():
                        colon_idx = example.find(":", i)
                        if colon_idx != -1 and colon_idx - i < 30:
                            next_name_idx = i
                            break
                
                # Extract text between current name and next name (or end)
                if next_name_idx == -1:
                    extracted_text = example[next_position:].strip()
                else:
                    extracted_text = example[next_position:next_name_idx].strip()
                    
                if extracted_text:
                    named_examples.append(extracted_text)
                    
                position = next_position if next_name_idx == -1 else next_name_idx
                
        return named_examples

    # Build dataset by extracting examples for each persona
    dataset = {}
    for name in sample_names:
        all_examples = []
        for day_num in range(1, 9):
            day_data = extract_named_examples(name, day_responses[f"day{day_num}"])
            all_examples.extend(day_data)
        dataset[name] = all_examples

    # Initialize language model and tokenizer with optimizations
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map='auto',
        attn_implementation="flash_attention_2",
        torch_dtype=torch.float16
    )

    # Load LoRA adapters for each persona and rank combination
    adapter_names = []
    for name in sample_names:
        print(name)
        for rank in lora_ranks:
            adapter_name = f"{name}-rank{rank}"
            adapter_names.append([adapter_name, name])
            model.load_adapter(
                f"{lora_base_path}{name}-rank{rank}",
                adapter_name=adapter_name,
                device_map="auto"
            )

    # Define schema for LLM judge responses using Pydantic
    class LLMasJudge(BaseModel):
        response1_output_quality_score: Literal["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]
        response1_persona_matching_score: Literal["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]
        response1_creativity_score: Literal["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]
        response1_coherence_score: Literal["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]
        response1_technical_accuracy_score: Literal["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]
        response1_emotional_intelligence_score: Literal["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]
        response1_professional_tone_score: Literal["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]
        response2_output_quality_score: Literal["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]
        response2_persona_matching_score: Literal["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]
        response2_creativity_score: Literal["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]
        response2_coherence_score: Literal["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]
        response2_technical_accuracy_score: Literal["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]
        response2_emotional_intelligence_score: Literal["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]
        response2_professional_tone_score: Literal["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]
        detailed_feedback: str
        overall_winner: Literal["1", "2"]

    # Set up JSON schema parsing for LLM outputs
    parser = JsonSchemaParser(LLMasJudge.schema())
    prefix_function = build_transformers_prefix_allowed_tokens_fn(tokenizer, parser)

    # Initialize Weights & Biases run for experiment tracking
    wandb.init(
        project=wandb_project,
        name=wandb_run_name,
        config={
            "model": model_id,
            "ranks": lora_ranks,
            "temperature": temperature,
            "min_p": min_p,
            "max_tokens": max_tokens,
            "num_samples": num_samples
        }
    )

    # Store results
    list_of_stuff = []
    
    # Filter personas by company
    matching_elements = [d for d in list_of_data if d["company_worked_at_name"] == company_filter]

    # Main evaluation loop
    for adapter_name, name in adapter_names:
        # Get persona details
        user_persona_str = ""
        for elem in matching_elements:
            if elem["first_name"] + " " + elem["last_name"] == name:
                user_persona_str = str(elem)
                
        # Set active LoRA adapter
        model.set_adapter(adapter_name=adapter_name)
        print(model.active_adapters())
        print(adapter_name)
        print("-----------------------------")
        
        # Generate and evaluate samples
        for i in range(num_samples):
            # Generate response with LoRA adapter
            model.enable_adapters()
            sample_prompt = random.sample(dataset[name], 1)[0]
            print(sample_prompt)
            
            # Construct prompt with persona context
            text = f"""You are {user_persona_str}
Continue this response in your persona's style: {sample_prompt}"""
            
            # Generate response using personalized (LoRA) model
            inputs = tokenizer(text, return_tensors="pt")
            output = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=True, min_p=min_p, temperature=temperature)
            the_output = tokenizer.batch_decode(output, skip_special_tokens=True)
            generated_text = the_output[0][len(text):].strip()

            # Generate response using base model
            model.disable_adapters()
            output2 = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=True, min_p=min_p, temperature=temperature)
            the_output2 = tokenizer.batch_decode(output2, skip_special_tokens=True)
            generated_text2 = the_output2[0][len(text):].strip()
            
            # Randomly order responses for unbiased evaluation
            ord_num = random.randint(0,1)

            # Construct prompt for LLM judge
            if ord_num == 1:
                llm_as_judge_text = f"""You are given a description of an individual's persona: {user_persona_str}

Along with AI generated responses from models trained on their slack conversations.

Response 1: {generated_text}

Response 2: {generated_text2}

Act as a judge and evaluate both responses. Score each aspect from 0-10 and provide detailed feedback on:
- Output quality (clarity, grammar, style)
- Persona matching (how well it matches the given persona)
- Creativity (uniqueness and originality)
- Coherence (logical flow and consistency)
- Technical accuracy (if technical content is present)
- Emotional intelligence (appropriate tone and empathy)
- Professional tone (workplace appropriateness)

Finally, determine which response is the overall winner."""

            else:
                llm_as_judge_text = f"""You are given a description of an individual's persona: {user_persona_str}

Along with AI generated responses from models trained on their slack conversations.

Response 1: {generated_text2}

Response 2: {generated_text}

Act as a judge and evaluate both responses. Score each aspect from 0-10 and provide detailed feedback on:
- Output quality (clarity, grammar, style)
- Persona matching (how well it matches the given persona)
- Creativity (uniqueness and originality)
- Coherence (logical flow and consistency)
- Technical accuracy (if technical content is present)
- Emotional intelligence (appropriate tone and empathy)
- Professional tone (workplace appropriateness)

Finally, determine which response is the overall winner."""

            # Get evaluation from LLM judge
            llm_response = client.beta.chat.completions.parse(
                messages=[
                    {"role": "system", "content": "You are an expert judge evaluating AI-generated text responses"},
                    {"role": "user", "content": llm_as_judge_text}
                ],
                model=judge_model,
                response_format=LLMasJudge,
                temperature=1.0,
                max_tokens=1000,
                n=1,
                stop="<|eot_id|>"
            )
            
            print(llm_response)
            
            # Log metrics and results
            metrics = {
                "adapter_name": adapter_name,
                "sample_number": i,
                "input_prompt": text,
                "personalized_output": generated_text,
                "base_output": generated_text2,
                "llm_judge_response": llm_response.choices[0].message.content,
                "personalized_order": ord_num,
                "response1_output_quality_score": int(llm_response.choices[0].message.parsed.response1_output_quality_score),
                "response1_persona_matching_score": int(llm_response.choices[0].message.parsed.response1_persona_matching_score),
                "response1_creativity_score": int(llm_response.choices[0].message.parsed.response1_creativity_score),
                "response1_coherence_score": int(llm_response.choices[0].message.parsed.response1_coherence_score),
                "response1_technical_accuracy_score": int(llm_response.choices[0].message.parsed.response1_technical_accuracy_score),
                "response1_emotional_intelligence_score": int(llm_response.choices[0].message.parsed.response1_emotional_intelligence_score),
                "response1_professional_tone_score": int(llm_response.choices[0].message.parsed.response1_professional_tone_score),
                "response2_output_quality_score": int(llm_response.choices[0].message.parsed.response2_output_quality_score),
                "response2_persona_matching_score": int(llm_response.choices[0].message.parsed.response2_persona_matching_score),
                "response2_creativity_score": int(llm_response.choices[0].message.parsed.response2_creativity_score),
                "response2_coherence_score": int(llm_response.choices[0].message.parsed.response2_coherence_score),
                "response2_technical_accuracy_score": int(llm_response.choices[0].message.parsed.response2_technical_accuracy_score),
                "response2_emotional_intelligence_score": int(llm_response.choices[0].message.parsed.response2_emotional_intelligence_score),
                "response2_professional_tone_score": int(llm_response.choices[0].message.parsed.response2_professional_tone_score),
                "detailed_feedback": llm_response.choices[0].message.parsed.detailed_feedback,
                "overall_winner": int(llm_response.choices[0].message.parsed.overall_winner),
                "user_persona": user_persona_str
            }
            
            wandb.log(metrics)
            list_of_stuff.append(metrics)

    # Save results to pickle file
    with open(output_file, 'wb') as f:
        pickle.dump(list_of_stuff, f)

    wandb.finish()

if __name__ == '__main__':
    run(main)