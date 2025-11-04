import argparse
import json
import os
import psutil
import requests
import sys
from typing import List, Literal
from urllib.parse import urlparse
import transformers

from jinja2 import Template
from pydantic import BaseModel
from tqdm import tqdm
from datasets import load_dataset, Dataset
from openai import OpenAI
import yaml
from concurrent.futures import ThreadPoolExecutor
import random

# Custom imports
sys.path.append("../")
from src.utils.vllm_utils import VLLMServerManager

vllm_client = OpenAI(
        base_url="http://127.0.0.1:8000/v1",
        api_key="-",
)

# ----------------------------
# Setup
# ----------------------------
transformers.logging.set_verbosity_error()

NO_ANSWER = "Do not wish to answer"


# ----------------------------
# Argument Parsing
# ----------------------------
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="meta-llama/Llama-3.2-1B-Instruct",
                        help="Model Name")
    parser.add_argument("--hf-token", type=str, default=None,
                        help="Huggingface token")
    parser.add_argument("--vllm-base-url", type=str, default="http://127.0.0.1:8000/v1",
                        help="Base URL for vLLM OpenAI-compatible server, e.g., http://127.0.0.1:8000/v1")
    parser.add_argument("--vllm-api-key", type=str, default="-",
                        help="API key to send to vLLM (usually ignored but required by the OpenAI client).")
    parser.add_argument("--out-dir", type=str, default=".",
                        help="Directory for Storing output")
    return parser.parse_args()


args = parse_args()
u = urlparse(args.vllm_base_url)
host = u.hostname or "127.0.0.1"
port = u.port or 8000

  
server_extra_args = os.environ.get("VLLM_SERVER_ARGS", "").strip().split()
vllm_mgr = VLLMServerManager(
    model=args.model_name,
    host=host,
    port=port,
    timeout_s=180,
    kill_existing=True,
    server_extra_args=server_extra_args,
)
vllm_mgr.ensure_fresh_server()

