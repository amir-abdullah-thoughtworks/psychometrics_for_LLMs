sjt_base_prompt_templates = {
    "gpt_cot": [
        {
            "role": "system",
            "content": """You are a law enforcement officer.
            Task: Answer the below multiple choice questions:""",
                    },
        {
            "role": "user",
            "content": """Question: {{ question }}

            Choices for these questions are:

            {{ answer_options }}

            Evaluate how consistent each option is with the behaviour.

            Assign a score to every option from 1 to 10, where:
            - 10 = highly consistent with the behaviour
            - 1 = not consistent with the behaviour

            Score each option independently rather than comparing them directly.

            Do not default to procedural or conservative choices. Base your scores strictly on how well each option matches the traits, behavior, and judgment.

            Provide concise reasoning explaining the scoring.

            Always respond as a JSON object in the following format:

            ```
            {
                "reasoning": "Brief explanation of how the persona traits influenced the scoring",
                "scores": [score_for_option_1, score_for_option_2, score_for_option_3, score_for_option_4, score_for_option_5, score_for_option_6]
            }
            ```
            
            """,
        },
    ],
    "gpt": [
        {
            "role": "system",
            "content": """You are a law enforcement officer.
            Task: Answer the below multiple choice questions:""",
                    },
        {
            "role": "user",
            "content": """Question: {{ question }}

            Choices for these questions are:

            {{ answer_options }}

            Answer the question using one of the values from [1,2,3,4,5,6].""",
        },
    ],
    
    "llama": """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
    
    Task: Answer the below multiple choice questions:<|eot_id|><|start_header_id|>user<|end_header_id|>
    
    Question: {{ question }}
    
    Choices for these questions are:

    {{ answer_options }}

    Answer the question using one of the values from [1,2,3,4,5,6].<|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
    
    "qwen": """<|im_start|>system
    
    Task: Answer the below multiple choice questions:
    <|im_end|>
    <|im_start|>user
    
    Question: {{ question }}
    
    Choices for these questions are:

    {{ answer_options }}

    Answer the question using one of the values from [1,2,3,4,5,6].
    <|im_end|>
    <|im_start|>assistant"""
    }