models=("gpt-4.1")


# "gpt-4.1-mini"    "Qwen/Qwen2.5-7B-Instruct"  "meta-llama/Llama-3.1-8B-Instruct"
# "Qwen/Qwen2.5-0.5B-Instruct" "Qwen/Qwen2.5-1.5B-Instruct" "Qwen/Qwen2.5-3B-Instruct" 
# "meta-llama/Llama-3.2-3B-Instruct" "meta-llama/Llama-3.2-1B-Instruct"

# # Command for Experiment 1 - Base Model
# for model in "${models[@]}"; do
#     echo "Running for $model"
#     python3 hf_personas_hexaco_v1.py --model-name $model \
#     --persona-source="base_model" \
#     --provider="vllm"
# done


# Command for Experiment 1 - Simple Personas
# for model in "${models[@]}"; do
#     echo "Running for $model"
#     python3 hf_personas_hexaco_v1.py --model-name $model \
#     --persona-source="personallm_paper" \
#     --provider="vllm" \
#     --n-personasample=64
# done

# Command for Experiment 2 - Base Model
# for model in "${models[@]}"; do
#     echo "Running for $model"
#     python3 hf_personas_hexaco_v1.py --model-name $model \
#     --persona-source="base_model" \
#     --n-times=30 \
#     --provider="vllm"
# done

# Command for Experiment 2 - Simple Personas
# for model in "${models[@]}"; do
#     echo "Running for $model"
#     python3 hf_personas_hexaco_v1.py --model-name $model \
#     --persona-source="personallm_paper" \
#     --n-times=30 \
#     --n-personasample=10 \
#     # --provider="vllm"
# done


# # Command for Experiment 3 - Base Model
# for model in "${models[@]}"; do
#     echo "Running for $model"
#     python3 hf_personas_hexaco_v1.py --model-name $model \
#     --persona-source="base_model" \
#     --n-times=2 \
#     --likert-shuffle \
#     --provider="vllm"
    
# done

# Command for Experiment 4 - Base Model
for model in "${models[@]}"; do
    echo "Running for $model"
    python3 hf_personas_hexaco_v1.py --model-name $model \
    --persona-source="base_model" \
    --paraphrase \
    # --provider="vllm"
    
done
