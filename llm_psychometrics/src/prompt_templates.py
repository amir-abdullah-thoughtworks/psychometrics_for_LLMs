import outlines

base_hexaco_template = outlines.Template.from_string("""
<|im_start>user
Task: Answer the below questions:

{{ text }}

Answer the question as either {{ likert_scale }} .
<|im_end>
<|im_start>assistant
""")

persona_hexaco_template = outlines.Template.from_string("""
<|im_start>user
{{base_text}} with following attributes :

{{attributes}}

Task: Answer the below questions:

{{ text }}

Answer the question as either {{ likert_scale }}.
<|im_end>
<|im_start>assistant
""")