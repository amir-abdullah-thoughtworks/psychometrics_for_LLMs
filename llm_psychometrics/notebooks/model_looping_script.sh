models=("gpt-4.1")


# Error : "gpt-4.1"
# "gpt-4.1-mini"  "Qwen/Qwen2.5-7B-Instruct" "Qwen/Qwen2.5-0.5B-Instruct" meta-llama/Llama-3.2-1B-Instruct
# "Qwen/Qwen2.5-1.5B-Instruct" "Qwen/Qwen2.5-3B-Instruct" "meta-llama/Llama-3.2-3B-Instruct" "meta-llama/Llama-3.1-8B-Instruct"

# # Command for Experiment 1 - Base Model
# for model in "${models[@]}"; do
#     echo "Running for $model"
#     python3 hf_personas_hexaco_v0.py --model-name $model \
#     --persona-source="base_model" \
#     --hf-token="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"
# done


# Command for Experiment 1 - Simple Personas
# for model in "${models[@]}"; do
#     echo "Running for $model"
#     python3 hf_personas_hexaco_v0.py --model-name $model \
#     --persona-source="personallm_paper"
# done

# Command for Experiment 2 - Simple Personas
for model in "${models[@]}"; do
    echo "Running for $model"
    python3 hf_personas_hexaco_v1.py --model-name $model \
    --persona-source="personallm_paper" \
    --n-times=30 \
    --n-personasample=10 \
    # --provider="vllm"
done