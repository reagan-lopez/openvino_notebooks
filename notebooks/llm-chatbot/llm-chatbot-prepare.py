import sys
import shutil
from pathlib import Path
import requests
import subprocess
from llm_config import SUPPORTED_LLM_MODELS
from huggingface_hub import login, whoami
import argparse


def convert_to_int4(model_configuration, int4_model_dir, model_id):
    compression_configs = {
        "zephyr-7b-beta": {
            "sym": True,
            "group_size": 64,
            "ratio": 0.6,
        },
        "mistral-7b": {
            "sym": True,
            "group_size": 64,
            "ratio": 0.6,
        },
        "minicpm-2b-dpo": {
            "sym": True,
            "group_size": 64,
            "ratio": 0.6,
        },
        "gemma-2b-it": {
            "sym": True,
            "group_size": 64,
            "ratio": 0.6,
        },
        "notus-7b-v1": {
            "sym": True,
            "group_size": 64,
            "ratio": 0.6,
        },
        "neural-chat-7b-v3-1": {
            "sym": True,
            "group_size": 64,
            "ratio": 0.6,
        },
        "llama-2-chat-7b": {
            "sym": True,
            "group_size": 128,
            "ratio": 0.8,
        },
        "llama-3-8b-instruct": {
            "sym": True,
            "group_size": 128,
            "ratio": 0.8,
        },
        "gemma-7b-it": {
            "sym": True,
            "group_size": 128,
            "ratio": 0.8,
        },
        "chatglm2-6b": {
            "sym": True,
            "group_size": 128,
            "ratio": 0.72,
        },
        "qwen-7b-chat": {"sym": True, "group_size": 128, "ratio": 0.6},
        "red-pajama-3b-chat": {
            "sym": False,
            "group_size": 128,
            "ratio": 0.5,
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
    export_command_base = "optimum-cli export openvino --model {} --task text-generation-with-past --weight-format int4".format(model_configuration["model_id"])
    int4_compression_args = " --group-size {} --ratio {}".format(model_compression_params["group_size"], model_compression_params["ratio"])
    if model_compression_params["sym"]:
        int4_compression_args += " --sym"
    export_command_base += int4_compression_args
    if remote_code:
        export_command_base += " --trust-remote-code"
    export_command = export_command_base + " " + str(int4_model_dir)
    print(f"Export Command: `{export_command}`")
    subprocess.run(export_command, shell=True)

def convert_to_int8(model_configuration, int8_model_dir):
    if (int8_model_dir / "openvino_model.xml").exists():
        return
    int8_model_dir.mkdir(parents=True, exist_ok=True)
    remote_code = model_configuration.get("remote_code", False)
    export_command_base = "optimum-cli export openvino --model {} --task text-generation-with-past --weight-format int8".format(model_configuration["model_id"])
    if remote_code:
        export_command_base += " --trust-remote-code"
    export_command = export_command_base + " " + str(int8_model_dir)
    print(f"Export Command: `{export_command}`")
    subprocess.run(export_command, shell=True)

def convert_to_fp16(model_configuration, fp16_model_dir):
    if (fp16_model_dir / "openvino_model.xml").exists():
        return
    remote_code = model_configuration.get("remote_code", False)
    export_command_base = "optimum-cli export openvino --model {} --task text-generation-with-past --weight-format fp16".format(model_configuration["model_id"])
    if remote_code:
        export_command_base += " --trust-remote-code"
    export_command = export_command_base + " " + str(fp16_model_dir)
    print(f"Export Command: `{export_command}`")
    subprocess.run(export_command, shell=True)

def main(model_index, precisions):

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
    model_id = model_ids[model_index]
    print(f"Selected model {model_id}")
    model_configuration = SUPPORTED_LLM_MODELS[model_language][model_id]

    int4_model_dir = Path(model_id) / "INT4_compressed_weights"
    int8_model_dir = Path(model_id) / "INT8_compressed_weights"
    fp16_model_dir = Path(model_id) / "FP16"

    for precision in precisions:
        if precision == "INT4":
            convert_to_int4(model_configuration, int4_model_dir, model_id)
        if precision == "INT8":
            convert_to_int8(model_configuration, int8_model_dir)
        if precision == "FP16":
            convert_to_fp16(model_configuration, fp16_model_dir)

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OpenVINO Model Preparation")
    parser.add_argument("--model_idx", type=int, default=2, help="Index of the model to use from SUPPORTED_LLM_MODELS")
    parser.add_argument("--precisions", type=str, default="INT4", help="Model precisions (INT4, INT8, FP16)")
    args = parser.parse_args()
    main(args.model_idx, args.precisions.split(','))
