import os
import socket
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

LOCAL_MODEL_DIR = PROJECT_ROOT / "models" / "bge-small-en-v1.5"
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
    print("====== BGE official download debug ======", flush=True)
    print("python:", sys.executable, flush=True)
    print("project_root:", PROJECT_ROOT, flush=True)
    print("local_model_dir:", LOCAL_MODEL_DIR, flush=True)
    print("hf_cache_dir:", HF_CACHE_DIR, flush=True)

    os.environ["HF_HUB_DISABLE_XET"] = "1"
    os.environ["HF_HOME"] = str(HF_CACHE_DIR)

    # 关键修复：不要使用 hf-mirror。
    # 上一次失败就是因为 HF_ENDPOINT=https://hf-mirror.com 返回的 metadata 不被 huggingface_hub 接受。
    if "HF_ENDPOINT" in os.environ:
        del os.environ["HF_ENDPOINT"]

    force_ipv4_dns()

    print("HF_HUB_DISABLE_XET:", os.environ.get("HF_HUB_DISABLE_XET"), flush=True)
    print("HF_HOME:", os.environ.get("HF_HOME"), flush=True)
    print("HF_ENDPOINT:", os.environ.get("HF_ENDPOINT"), flush=True)

    print("====== import huggingface_hub ======", flush=True)
    from huggingface_hub import snapshot_download

    print("====== start downloading BAAI/bge-small-en-v1.5 from official Hugging Face ======", flush=True)

    snapshot_download(
        repo_id="BAAI/bge-small-en-v1.5",
        local_dir=str(LOCAL_MODEL_DIR),
        cache_dir=str(HF_CACHE_DIR),
        max_workers=1,
    )

    print("====== download finished ======", flush=True)

    check_files = [
        "modules.json",
        "config_sentence_transformers.json",
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
    ]

    for name in check_files:
        path = LOCAL_MODEL_DIR / name
        print(f"check {name}: {path.exists()} -> {path}", flush=True)

    print("====== test local SentenceTransformer load ======", flush=True)
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(
        str(LOCAL_MODEL_DIR),
        local_files_only=True,
    )

    emb = model.encode(["test"], normalize_embeddings=True)
    print("local bge loaded ok", flush=True)
    print("embedding shape:", emb.shape, flush=True)


if __name__ == "__main__":
    main()