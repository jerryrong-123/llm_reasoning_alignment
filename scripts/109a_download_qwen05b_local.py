import os
import socket
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

LOCAL_MODEL_DIR = PROJECT_ROOT / "models" / "qwen2.5-0.5b-instruct"
HF_CACHE_DIR = PROJECT_ROOT / ".hf_cache"

LOCAL_MODEL_DIR.mkdir(parents=True, exist_ok=True)
HF_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def force_ipv4_dns():
    original_getaddrinfo = socket.getaddrinfo

    def getaddrinfo_ipv4(host, port, family=0, type=0, proto=0, flags=0):
        return original_getaddrinfo(
            host,
            port,
            socket.AF_INET,
            type,
            proto,
            flags,
        )

    socket.getaddrinfo = getaddrinfo_ipv4


def main():
    print("====== Qwen local download debug ======", flush=True)
    print("python:", sys.executable, flush=True)
    print("project_root:", PROJECT_ROOT, flush=True)
    print("local_model_dir:", LOCAL_MODEL_DIR, flush=True)
    print("hf_cache_dir:", HF_CACHE_DIR, flush=True)

    os.environ["HF_HUB_DISABLE_XET"] = "1"
    os.environ["HF_HOME"] = str(HF_CACHE_DIR)

    if "HF_ENDPOINT" in os.environ:
        del os.environ["HF_ENDPOINT"]

    force_ipv4_dns()

    print("HF_HUB_DISABLE_XET:", os.environ.get("HF_HUB_DISABLE_XET"), flush=True)
    print("HF_HOME:", os.environ.get("HF_HOME"), flush=True)
    print("HF_ENDPOINT:", os.environ.get("HF_ENDPOINT"), flush=True)

    print("====== import huggingface_hub ======", flush=True)
    from huggingface_hub import snapshot_download

    print("====== start downloading Qwen/Qwen2.5-0.5B-Instruct ======", flush=True)

    snapshot_download(
        repo_id="Qwen/Qwen2.5-0.5B-Instruct",
        local_dir=str(LOCAL_MODEL_DIR),
        cache_dir=str(HF_CACHE_DIR),
        max_workers=1,
    )

    print("====== download finished ======", flush=True)

    check_files = [
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "generation_config.json",
    ]

    for name in check_files:
        path = LOCAL_MODEL_DIR / name
        print(f"check {name}: {path.exists()} -> {path}", flush=True)

    print("====== test local transformers load ======", flush=True)
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(LOCAL_MODEL_DIR),
        local_files_only=True,
        trust_remote_code=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        str(LOCAL_MODEL_DIR),
        local_files_only=True,
        torch_dtype=torch.float32,
        trust_remote_code=True,
    )

    model.eval()

    text = "Hello"
    inputs = tokenizer(text, return_tensors="pt")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=5,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)

    print("local qwen loaded ok", flush=True)
    print("test generation:", decoded, flush=True)


if __name__ == "__main__":
    main()