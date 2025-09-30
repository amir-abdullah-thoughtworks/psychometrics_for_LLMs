hexaco_base_prompt_templates = {
    "gpt": [
        {
            "role": "system",
            "content": """Task: Answer the below questions:""",
                    },
                    {
                        "role": "user",
                        "content": """{{ text }}

            Answer the question as either {{ likert_scale }}.
            Do not return the question, just return the answer directly.""",
        },
    ],
    
    "llama": """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
    
    Task: Answer the below questions:<|eot_id|><|start_header_id|>user<|end_header_id|>
    
    {{ text }}
    
    Answer the question as either {{ likert_scale }}.
    Do not return the question, just return the answer directly.<|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
    
    "qwen": """<|im_start|>system
    
    
    Task: Answer the below questions:
    <|im_end|>
    <|im_start|>user

    {{ text }}
    
    Answer the question as either {{ likert_scale }}.
    Do not return the question, just return the answer directly.
    <|im_end|>
    <|im_start|>assistant"""
    }