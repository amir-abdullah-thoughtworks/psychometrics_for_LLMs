hexaco_persona_prompt_templates = {
    "gpt": """<|im_start|>system
    You are a law enforcement officer with following attributes :
    
    {{ attributes }}
    
    Task: Answer the below questions:
    <|im_end|>
    <|im_start|>user

    {{ text }}
    
    Answer the question as either {{ likert_scale }}.
    Do not return the question, just return the answer directly.
    <|im_end|>
    <|im_start|>assistant""",
    
    "llama": """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

    You are a law enforcement officer with following attributes :
    
    {{ attributes }}
    
    Task: Answer the below questions:<|eot_id|><|start_header_id|>user<|end_header_id|>
    
    {{ text }}
    
    Answer the question as either {{ likert_scale }}.
    Do not return the question, just return the answer directly.<|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
    
    "qwen": """<|im_start|>system
    
    
    You are a law enforcement officer with following attributes :
    
    {{ attributes }}
    
    Task: Answer the below questions:
    <|im_end|>
    <|im_start|>user
    {{ text }}
    
    Answer the question as either {{ likert_scale }}.
    Do not return the question, just return the answer directly.
    <|im_end|>
    <|im_start|>assistant"""
    }