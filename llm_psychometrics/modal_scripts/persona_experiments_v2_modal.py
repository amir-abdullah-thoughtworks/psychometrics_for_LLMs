import modal
import os
import zipfile
import tempfile

app = modal.App("run-local-script")
image = modal.Image.debian_slim().pip_install("torch", "transformers","outlines==1.1.1","tqdm","huggingface_hub","accelerate","datasets").add_local_dir(".", "/root/my_project")
    # add_local_file("persona_experiments_v2.py", "/root/persona_experiments_v2.py")
    

volume = modal.Volume.from_name("my-outputs", create_if_missing=True)

@app.function(
    image=modal.Image.debian_slim(),
    volumes={"/outputs": volume}
)
def download_all_files():
    """Download all files from volume as zip"""
    if not os.path.exists("/outputs") or not os.listdir("/outputs"):
        return None
    
    with tempfile.NamedTemporaryFile(delete=False, suffix='.zip') as tmp:
        with zipfile.ZipFile(tmp, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk("/outputs"):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, "/outputs")
                    zipf.write(file_path, arcname)
        
        with open(tmp.name, "rb") as f:
            return f.read()


@app.function(
    image=image, 
    gpu="T4", 
    timeout=10000,
    volumes={"/outputs": volume})
def run_my_script():
    import sys
    import os
    
    # Add the project directory to Python path
    sys.path.append("/root/my_project")
    
    # Change to the project directory (important for relative file paths)
    print(os.getcwd())
    print(os.listdir())
    os.chdir("/root/my_project")
    import persona_experiments_v2
    persona_experiments_v2.main()
    
    import glob
    print("Glob: ",glob.glob("persona_experiment_results_v2/*.json"))
    
    import shutil
    import glob
    
    # Copy all generated files to the volume
    output_patterns = ["persona_experiment_results_v2/*.json"]
    for pattern in output_patterns:
        for file in glob.glob(pattern):
            shutil.copy(file, f"/outputs/{file.split('/')[-1]}")
            print(f"Saved {file} to outputs")
    
if __name__ == "__main__":
    
   with app.run():
    run_my_script.remote()
    zip_data = download_all_files.remote()
    with open("my_files.zip", "wb") as f:
        f.write(zip_data)
    print("Downloaded to my_files.zip")
