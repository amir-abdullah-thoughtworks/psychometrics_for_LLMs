python -m pip install --upgrade pip

# Make sure you run on 3.11.9
# 1) Install PyTorch built for CUDA 12.6
pip install torch==2.7.0 --index-url https://download.pytorch.org/whl/cu126

# 2) Install vLLM (0.9.2 is a safe match with Torch 2.6; 0.10.2 also works on many setups)
pip install "vllm==0.9.2" --no-deps
pip install "transformers==4.53.2"
pip install --upgrade outlines --no-deps
# 
# Make sure flash attention not installed. Make sure enforce_eager runs.
