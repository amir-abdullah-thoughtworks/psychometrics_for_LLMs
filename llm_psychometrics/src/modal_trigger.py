import os
import tempfile
import zipfile
from pathlib import Path

import modal

app = modal.App("run-bench-local")

vllm_image = (
    modal.Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .uv_pip_install(
        "vllm==0.10.2",
        "huggingface_hub[hf_transfer]==0.35.0",
        "flashinfer-python==0.3.1",
        "torch==2.8.0",
        "datasets==3.6.0",
        "anthropic==0.75.0"
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})  # faster model transfers
    .add_local_dir(".", "/root/my_project")
)

image = (
    modal.Image.debian_slim()
    .pip_install(
        "openai==1.109.1",
        "transformers==4.57.3",
        "pandas==2.3.2",
        "python-dotenv==1.1.1",
        "huggingface-hub==0.35.1",
        "PyYAML==6.0.3",
        "tiktoken==0.11.0",
        "plotly==6.3.0",
        "sentencepiece==0.2.1",
        "accelerate",
        "sentence-transformers",
        "torch",
        "kaleido",
        "ipdb",
    )
    .add_local_dir(".", "/root/my_project")
)
# add_local_file("persona_experiments_v2.py", "/root/persona_experiments_v2.py")


volume = modal.Volume.from_name("my-outputs", create_if_missing=True)


# @app.function(image=modal.Image.debian_slim(), volumes={"/outputs": volume})
# def download_all_files():
#     """Download all files from volume as zip"""
#     if not os.path.exists("/outputs") or not os.listdir("/outputs"):
#         return None

#     with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as tmp:
#         with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zipf:
#             for root, dirs, files in os.walk("/outputs"):
#                 for file in files:
#                     file_path = os.path.join(root, file)
#                     arcname = os.path.relpath(file_path, "/outputs")
#                     zipf.write(file_path, arcname)

#         with open(tmp.name, "rb") as f:
#             return f.read()



@app.function(image=vllm_image, gpu="A100", timeout=86400, volumes={"/outputs": volume},secrets=[modal.Secret.from_name("LLMPsychometrics")])
def run_my_script():
    import os
    import sys

    # Add the project directory to Python path
    sys.path.append("/root/my_project")

    # Change to the project directory (important for relative file paths)
    print(os.getcwd())
    print(os.listdir())
    os.chdir("/root/my_project")
    sys.path.append(str(Path(__file__).parent))
    from external_response_generation.sjt_response_generator_merged import main

    main()

    # import glob

    # print("Glob: ", glob.glob("results/**"))

    # import glob
    # import shutil

    # # Copy all generated files to the volume
    # src = "results"
    # dst = "/outputs"

    # os.makedirs(dst, exist_ok=True)

    # for filename in os.listdir(src):
    #     print(filename)
    #     src_path = os.path.join(src, filename)
    #     dst_path = os.path.join(dst, filename)

    #     if os.path.isfile(src_path):
    #         shutil.copy2(src_path, dst_path)
    #     else:
    #         shutil.copytree(src_path, dst_path, dirs_exist_ok=True)


@app.local_entrypoint()
def main():
    run_my_script.remote()


if __name__ == "__main__":
    main()
    # zip_data = download_all_files.remote()
    # with open("my_files.zip", "wb") as f:
    #     f.write(zip_data)
    # print("Downloaded to my_files.zip")
