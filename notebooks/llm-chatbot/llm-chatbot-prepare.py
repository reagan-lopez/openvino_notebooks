import shutil
from pathlib import Path
import requests
import subprocess
from llm_config import SUPPORTED_LLM_MODELS
from huggingface_hub import login, whoami


# Choose the model precision
prepare_int4_model = True
prepare_int8_model = False
prepare_fp16_model = False

# fetch model configuration
config_shared_path = Path("../../utils/llm_config.py")
config_dst_path = Path("llm_config.py")

if not config_dst_path.exists():
    if config_shared_path.exists():
        shutil.copy(config_shared_path, config_dst_path)
    else:
        r = requests.get(url="https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/utils/llm_config.py")
        with open("llm_config.py", "w") as f:
            f.write(r.text)

model_languages = list(SUPPORTED_LLM_MODELS)
model_language = model_languages[0]
model_ids = list(SUPPORTED_LLM_MODELS[model_language])
model_id = model_ids[5]
print(f"Selected model {model_id}")
model_configuration = SUPPORTED_LLM_MODELS[model_language][model_id]



pt_model_id = model_configuration["model_id"]
print(model_id)
pt_model_name = model_id.split("-")[0]
fp16_model_dir = Path(model_id) / "FP16"
int8_model_dir = Path(model_id) / "INT8_compressed_weights"
int4_model_dir = Path(model_id) / "INT4_compressed_weights"

def convert_to_fp16():
    if (fp16_model_dir / "openvino_model.xml").exists():
        return
    remote_code = model_configuration.get("remote_code", False)
    export_command_base = "optimum-cli export openvino --model {} --task text-generation-with-past --weight-format fp16".format(pt_model_id)
    if remote_code:
        export_command_base += " --trust-remote-code"
    export_command = export_command_base + " " + str(fp16_model_dir)
    print(f"Export Command: `{export_command}`")
    subprocess.run(export_command, shell=True)

def convert_to_int8():
    if (int8_model_dir / "openvino_model.xml").exists():
        return
    int8_model_dir.mkdir(parents=True, exist_ok=True)
    remote_code = model_configuration.get("remote_code", False)
    export_command_base = "optimum-cli export openvino --model {} --task text-generation-with-past --weight-format int8".format(pt_model_id)
    if remote_code:
        export_command_base += " --trust-remote-code"
    export_command = export_command_base + " " + str(int8_model_dir)
    print(f"Export Command: `{export_command}`")
    subprocess.run(export_command, shell=True)

def convert_to_int4():
    compression_configs = {
        "llama-2-chat-7b": {
            "sym": True,
            "group_size": 128,
            "ratio": 0.8,
        },
        "default": {
            "sym": False,
            "group_size": 128,
            "ratio": 0.8,
        },
    }



    try:
        whoami()
        print('Authorization token already provided')
    except OSError:
        login()    

    model_compression_params = compression_configs.get(model_id, compression_configs["default"])
    if (int4_model_dir / "openvino_model.xml").exists():
        return
    remote_code = model_configuration.get("remote_code", False)
    export_command_base = "optimum-cli export openvino --model {} --task text-generation-with-past --weight-format int4".format(pt_model_id)
    int4_compression_args = " --group-size {} --ratio {}".format(model_compression_params["group_size"], model_compression_params["ratio"])
    if model_compression_params["sym"]:
        int4_compression_args += " --sym"
    export_command_base += int4_compression_args
    if remote_code:
        export_command_base += " --trust-remote-code"
    export_command = export_command_base + " " + str(int4_model_dir)
    print(f"Export Command: `{export_command}`")
    subprocess.run(export_command, shell=True)

if prepare_fp16_model:
    convert_to_fp16()
if prepare_int8_model:
    convert_to_int8()
if prepare_int4_model:
    convert_to_int4()

fp16_weights = fp16_model_dir / "openvino_model.bin"
int8_weights = int8_model_dir / "openvino_model.bin"
int4_weights = int4_model_dir / "openvino_model.bin"

if fp16_weights.exists():
    print(f"Size of FP16 model is {fp16_weights.stat().st_size / 1024 / 1024:.2f} MB")
for precision, compressed_weights in zip([8, 4], [int8_weights, int4_weights]):
    if compressed_weights.exists():
        print(f"Size of model with INT{precision} compressed weights is {compressed_weights.stat().st_size / 1024 / 1024:.2f} MB")
    if compressed_weights.exists() and fp16_weights.exists():
        print(f"Compression rate for INT{precision} model: {fp16_weights.stat().st_size / compressed_weights.stat().st_size:.3f}")