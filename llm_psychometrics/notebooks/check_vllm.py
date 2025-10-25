from hf_personas_hexaco_v1 import VLLMServerManager

vm = VLLMServerManager()
vm.ensure_fresh_server(run_benchmark=True)