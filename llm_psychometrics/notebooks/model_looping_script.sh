models=("gpt-4.1-mini" "gpt-4.1" "meta-llama/Llama-3.2-1B-Instruct")

# "meta-llama/Llama-3.2-3B-Instruct" 
# "meta-llama/Llama-3.1-8B-Instruct" "Qwen/Qwen3-0.6B" "Qwen/Qwen3-1.7B" "Qwen/Qwen3-4B"
# "Qwen/Qwen3-8B")

# Command for Experiment 1 - Base Model
# for model in "${models[@]}"; do
#     echo "Running for $model"
#     python3 hf_personas_hexaco_v0.py --model-name $model \
#     --persona-source="base_model"
# done


# # Command for Experiment 1 - Simple Personas
for model in "${models[@]}"; do
    echo "Running for $model"
    python3 hf_personas_hexaco_v0.py --model-name $model \
    --persona-source="personallm_paper"
done
