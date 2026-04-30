models=("Qwen/Qwen2.5-7B-Instruct" )

export HF_TOKEN="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"

eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"
pyenv activate vllm311

#   "gpt-4.1-mini" "gpt-4.1"
#  "Qwen/Qwen2.5-0.5B-Instruct" "Qwen/Qwen2.5-1.5B-Instruct" "Qwen/Qwen2.5-3B-Instruct"
#  "meta-llama/Llama-3.2-3B-Instruct"  "meta-llama/Llama-3.2-1B-Instruct" "meta-llama/Llama-3.1-8B-Instruct" 

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

# # Command for Experiment 4 - Base Model
# for model in "${models[@]}"; do
#     echo "Running for $model"
#     python3 hf_personas_hexaco_v1.py --model-name $model \
#     --persona-source="base_model" \
#     --paraphrase \
#     # --provider="vllm"
    
# done

# # Command for Experiment 6 - Base Model
# for model in "${models[@]}"; do
#     echo "Running for $model"
#     python3 hf_personas_sjt_answers_v1.py --model-name $model \
#     --persona-source="base_model" \
#     --answer-shuffle \
#     --n-sjtsample=25 \
#     --out-dir="../experiment_results/reliability_experiments/base_text_experiment_results" \
#     --provider="vllm"
        
# done


# # # Command for Experiment 6 - Simple Personas
# for model in "${models[@]}"; do
#     echo "Running for $model"
#     python3 hf_personas_sjt_answers_v1.py --model-name $model \
#     --persona-source="personallm_paper" \
#     --n-personasample=10 \
#     --n-times=30 \
#     --answer-shuffle \
#     --n-sjtsample=5 \
#     --out-dir="../experiment_results/reliability_experiments/vllm_experiment_6" \
#     # --provider="vllm"
    
    
# done


# models=( "gpt-4.1")

# # Command for Zero Compute Analysis Data Run - SJT
for model in "${models[@]}"; do
    echo "Running for $model"
    python3 hf_personas_sjt_answers_v1.py --model-name $model \
    --persona-source="huggingface" \
    --n-personasample=25 \
    --answer-shuffle \
    --n-sjtsample=25 \
    --out-dir="../experiment_results/zero_compute_analysis_sjt" \
    --provider="vllm" 
    
# done

# # Command for Zero Compute Analysis Data Run - Hexaco
# for model in "${models[@]}"; do
#     echo "Running for $model"
#     python3 hf_personas_hexaco_v1.py --model-name $model \
#     --persona-source="huggingface" \
#     --n-personasample=25 \
#     --likert-shuffle \
#     --out-dir="../experiment_results/zero_compute_analysis_hexaco"
#     # --provider="vllm" \
    
# done
