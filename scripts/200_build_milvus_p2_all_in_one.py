from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

MILVUS_DIR = PROJECT_ROOT / "outputs" / "hierarchical_rag" / "milvus"
REPORT_DIR = PROJECT_ROOT / "outputs" / "reports"
PACKAGE_DIR = PROJECT_ROOT / "outputs" / "packages"
MILVUS_DB_PATH = MILVUS_DIR / "milvus_child_store.db"
COLLECTION_NAME = "hierarchical_rag_child_chunks"

FAISS_INDEX_PATH = PROJECT_ROOT / "outputs" / "hierarchical_rag" / "index" / "faiss_child.index"
FAISS_META_PATH = PROJECT_ROOT / "outputs" / "hierarchical_rag" / "index" / "faiss_child_meta.json"

MILVUS_STORE_PATH = PROJECT_ROOT / "industrial_rag_service" / "milvus_store.py"
FACTORY_PATH = PROJECT_ROOT / "industrial_rag_service" / "vector_store_factory.py"
CONFIG_PATH = PROJECT_ROOT / "industrial_rag_service" / "config.yaml"
REQ_PATH = PROJECT_ROOT / "requirements_rag_service.txt"

BUILD_RESULTS_PATH = MILVUS_DIR / "milvus_index_build_results.json"
BUILD_REPORT_PATH = MILVUS_DIR / "milvus_index_build_report.md"

SEARCH_RESULTS_PATH = MILVUS_DIR / "milvus_vector_search_test_results.json"
SEARCH_REPORT_PATH = MILVUS_DIR / "milvus_vector_search_test_report.md"

STORE_RESULTS_PATH = MILVUS_DIR / "milvus_store_backend_test_results.json"
STORE_REPORT_PATH = MILVUS_DIR / "milvus_store_backend_test_report.md"

CHECK_RESULTS_PATH = REPORT_DIR / "milvus_backend_file_check_results.json"
CHECK_REPORT_PATH = REPORT_DIR / "milvus_backend_file_check_report.md"

PACKAGE_PATH = PACKAGE_DIR / "milvus_p2_backend_update_package.tar.gz"


MILVUS_STORE_CODE = r'''
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from industrial_rag_service.vector_store import VectorSearchResult, VectorStore


class MilvusVectorStore(VectorStore):
    """
    Milvus-compatible backend for the industrial Hierarchical RAG service.

    This implementation uses pymilvus MilvusClient. In this project stage, the
    URI can point to a Milvus Lite local .db file. In a production deployment,
    the same class can point to a remote Milvus Standalone / Distributed service,
    for example http://host:19530.
    """

    def __init__(
        self,
        uri: str,
        collection_name: str = "hierarchical_rag_child_chunks",
        model_name: str = "BAAI/bge-small-en-v1.5",
        device: str = "cpu",
        normalize_embeddings: bool = True,
        load_embedding_model: bool = True,
        vector_field_name: str = "vector",
        metric_type: str = "IP",
    ) -> None:
        self.uri = uri
        self.collection_name = collection_name
        self.model_name = model_name
        self.device = device
        self.normalize_embeddings = normalize_embeddings
        self.load_embedding_model = load_embedding_model
        self.vector_field_name = vector_field_name
        self.metric_type = metric_type

        self.client: Optional[Any] = None
        self.model: Optional[Any] = None
        self.loaded = False

    def load(self) -> None:
        os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
        os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

        from pymilvus import MilvusClient

        if self.uri.endswith(".db"):
            Path(self.uri).parent.mkdir(parents=True, exist_ok=True)

        self.client = MilvusClient(self.uri)

        if self.load_embedding_model:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.model_name, device=self.device)
        else:
            self.model = None

        self.loaded = True

    def search(
        self,
        query: str,
        top_k: int = 10,
        source_query: Optional[str] = None,
    ) -> List[VectorSearchResult]:
        if not self.loaded or self.client is None:
            raise RuntimeError("MilvusVectorStore is not loaded. Call load() first.")

        if self.model is None:
            raise RuntimeError(
                "MilvusVectorStore was loaded without an embedding model. "
                "Use search_by_embedding(...) or initialize with load_embedding_model=True."
            )

        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string.")

        query = query.strip()
        top_k = max(1, int(top_k))

        query_embedding = self.model.encode(
            [query],
            batch_size=1,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize_embeddings,
        )

        query_embedding = np.asarray(query_embedding, dtype="float32")[0]

        return self.search_by_embedding(
            query_embedding=query_embedding,
            top_k=top_k,
            source_query=source_query or query,
        )

    def search_by_embedding(
        self,
        query_embedding: Sequence[float],
        top_k: int = 10,
        source_query: Optional[str] = None,
    ) -> List[VectorSearchResult]:
        if not self.loaded or self.client is None:
            raise RuntimeError("MilvusVectorStore is not loaded. Call load() first.")

        top_k = max(1, int(top_k))
        vector = np.asarray(query_embedding, dtype="float32").reshape(-1)

        raw = self.client.search(
            collection_name=self.collection_name,
            data=[vector.tolist()],
            anns_field=self.vector_field_name,
            limit=top_k,
            output_fields=["child_id", "parent_id", "title", "text", "index_id"],
        )

        return self._convert_milvus_search_result(raw, source_query=source_query)

    def _convert_milvus_search_result(
        self,
        raw: Any,
        source_query: Optional[str] = None,
    ) -> List[VectorSearchResult]:
        hits = raw[0] if raw else []
        results: List[VectorSearchResult] = []

        for rank, hit in enumerate(hits, start=1):
            if isinstance(hit, dict):
                entity = hit.get("entity", {}) or {}
                hit_id = hit.get("id")
                distance = hit.get("distance", hit.get("score", 0.0))
            else:
                entity = getattr(hit, "entity", {}) or {}
                hit_id = getattr(hit, "id", None)
                distance = getattr(hit, "distance", getattr(hit, "score", 0.0))

            if not isinstance(entity, dict):
                try:
                    entity = dict(entity)
                except Exception:
                    entity = {}

            score = float(distance or 0.0)

            child_id = str(entity.get("child_id") or hit_id)
            parent_id = entity.get("parent_id")
            title = entity.get("title")
            text = str(entity.get("text", ""))

            index_id_raw = entity.get("index_id", rank - 1)
            try:
                index_id = int(index_id_raw)
            except (TypeError, ValueError):
                index_id = rank - 1

            metadata = dict(entity)
            metadata["milvus_id"] = hit_id

            results.append(
                VectorSearchResult(
                    rank=rank,
                    score=score,
                    index_id=index_id,
                    child_id=child_id,
                    parent_id=parent_id,
                    title=title,
                    text=text,
                    source_query=source_query,
                    metadata=metadata,
                )
            )

        return results

    def add_documents(self, docs: List[Dict[str, Any]]) -> None:
        raise NotImplementedError(
            "MilvusVectorStore.add_documents is not implemented in this stage. "
            "Use the FAISS-to-Milvus build script for bulk import."
        )

    def delete_documents(self, ids: List[str]) -> None:
        if not self.loaded or self.client is None:
            raise RuntimeError("MilvusVectorStore is not loaded. Call load() first.")

        clean_ids = [int(item) for item in ids if str(item).strip()]
        if not clean_ids:
            return

        expr = "id in [" + ",".join(str(item) for item in clean_ids) + "]"
        self.client.delete(collection_name=self.collection_name, filter=expr)

    def count(self) -> int:
        if not self.loaded or self.client is None:
            raise RuntimeError("MilvusVectorStore is not loaded. Call load() first.")

        try:
            stats = self.client.get_collection_stats(collection_name=self.collection_name)
            if isinstance(stats, dict):
                for key in ("row_count", "num_rows"):
                    if key in stats:
                        return int(stats[key])
        except Exception:
            pass

        try:
            rows = self.client.query(
                collection_name=self.collection_name,
                filter="id >= 0",
                output_fields=["id"],
                limit=16384,
            )
            return len(rows)
        except Exception:
            return 0

    def close(self) -> None:
        self.client = None
        self.model = None
        self.loaded = False
'''


FACTORY_CODE = r'''
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from industrial_rag_service.vector_store import VectorStore


def resolve_project_path(project_root: Path, path_value: str) -> Path:
    path = Path(path_value)

    if path.is_absolute():
        return path

    return project_root / path


def create_vector_store(
    project_root: Path,
    config: Dict[str, Any],
) -> VectorStore:
    """
    Create a vector store backend from service config.

    Supported backends:
    - faiss
    - chroma
    - milvus
    """

    paths_config = config.get("paths", {})
    retrieval_config = config.get("retrieval", {})
    embedding_config = config.get("embedding", {})

    backend = str(retrieval_config.get("backend", "faiss")).strip().lower()

    model_name = str(
        embedding_config.get("model_name", "BAAI/bge-small-en-v1.5")
    )
    device = str(embedding_config.get("device", "cpu"))
    normalize_embeddings = bool(embedding_config.get("normalize_embeddings", True))

    if backend == "faiss":
        from industrial_rag_service.faiss_store import FAISSVectorStore

        index_path = resolve_project_path(
            project_root=project_root,
            path_value=str(
                paths_config.get(
                    "faiss_index_path",
                    "outputs/hierarchical_rag/index/faiss_child.index",
                )
            ),
        )
        meta_path = resolve_project_path(
            project_root=project_root,
            path_value=str(
                paths_config.get(
                    "faiss_meta_path",
                    "outputs/hierarchical_rag/index/faiss_child_meta.json",
                )
            ),
        )

        return FAISSVectorStore(
            index_path=str(index_path),
            meta_path=str(meta_path),
            model_name=model_name,
            device=device,
            normalize_embeddings=normalize_embeddings,
        )

    if backend == "chroma":
        from industrial_rag_service.chroma_store import ChromaVectorStore

        persist_dir = resolve_project_path(
            project_root=project_root,
            path_value=str(
                paths_config.get(
                    "chroma_persist_dir",
                    "outputs/hierarchical_rag/chroma/chroma_child_store",
                )
            ),
        )

        collection_name = str(
            retrieval_config.get(
                "chroma_collection_name",
                "hierarchical_rag_child_chunks",
            )
        )

        return ChromaVectorStore(
            persist_dir=str(persist_dir),
            collection_name=collection_name,
            model_name=model_name,
            device=device,
            normalize_embeddings=normalize_embeddings,
            load_embedding_model=True,
        )

    if backend == "milvus":
        from industrial_rag_service.milvus_store import MilvusVectorStore

        uri = str(
            paths_config.get(
                "milvus_uri",
                "outputs/hierarchical_rag/milvus/milvus_child_store.db",
            )
        )

        if uri.endswith(".db") and not Path(uri).is_absolute():
            uri = str(resolve_project_path(project_root=project_root, path_value=uri))

        collection_name = str(
            retrieval_config.get(
                "milvus_collection_name",
                "hierarchical_rag_child_chunks",
            )
        )

        vector_field_name = str(retrieval_config.get("milvus_vector_field_name", "vector"))
        metric_type = str(retrieval_config.get("milvus_metric_type", "IP"))

        return MilvusVectorStore(
            uri=uri,
            collection_name=collection_name,
            model_name=model_name,
            device=device,
            normalize_embeddings=normalize_embeddings,
            load_embedding_model=True,
            vector_field_name=vector_field_name,
            metric_type=metric_type,
        )

    raise ValueError(
        f"Unsupported vector store backend: {backend}. "
        "Supported backends are: faiss, chroma, milvus."
    )
'''


def run(cmd: List[str]) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, check=True)


def install_deps() -> None:
    run([sys.executable, "-m", "pip", "install", "-U", "pymilvus[milvus_lite]"])


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def write_md(path: Path, title: str, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# {title}",
        "",
        "```json",
        json.dumps(data, ensure_ascii=False, indent=2),
        "```",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_code_files() -> None:
    print("[1/8] Writing Milvus backend code")
    MILVUS_STORE_PATH.write_text(MILVUS_STORE_CODE.strip() + "\n", encoding="utf-8")
    FACTORY_PATH.write_text(FACTORY_CODE.strip() + "\n", encoding="utf-8")

    import yaml

    cfg = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    cfg.setdefault("paths", {})
    cfg.setdefault("retrieval", {})

    cfg["paths"]["milvus_dir"] = "outputs/hierarchical_rag/milvus"
    cfg["paths"]["milvus_uri"] = "outputs/hierarchical_rag/milvus/milvus_child_store.db"

    cfg["retrieval"]["milvus_collection_name"] = COLLECTION_NAME
    cfg["retrieval"]["milvus_vector_field_name"] = "vector"
    cfg["retrieval"]["milvus_metric_type"] = "IP"

    CONFIG_PATH.write_text(
        yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )

    req = REQ_PATH.read_text(encoding="utf-8")
    if "pymilvus" not in req:
        req = req.rstrip() + "\npymilvus[milvus_lite]\n"
    REQ_PATH.write_text(req, encoding="utf-8")


def load_faiss_vectors_and_meta() -> tuple[Any, List[Dict[str, Any]]]:
    import faiss

    if not FAISS_INDEX_PATH.exists():
        raise FileNotFoundError(f"FAISS index not found: {FAISS_INDEX_PATH}")
    if not FAISS_META_PATH.exists():
        raise FileNotFoundError(f"FAISS meta not found: {FAISS_META_PATH}")

    meta = load_json(FAISS_META_PATH)
    chunks = meta.get("chunks", [])
    if not chunks:
        raise ValueError("FAISS metadata chunks missing.")

    index = faiss.read_index(str(FAISS_INDEX_PATH))
    vectors = index.reconstruct_n(0, int(index.ntotal))

    if len(chunks) != int(index.ntotal):
        raise ValueError(f"chunk/vector mismatch: chunks={len(chunks)}, vectors={index.ntotal}")

    return vectors, chunks


def truncate(text: str, max_len: int = 6000) -> str:
    text = str(text or "")
    if len(text) <= max_len:
        return text
    return text[:max_len]


def build_milvus_index() -> Dict[str, Any]:
    print("[2/8] Building Milvus Lite index from FAISS vectors")
    from pymilvus import MilvusClient

    MILVUS_DIR.mkdir(parents=True, exist_ok=True)

    if MILVUS_DB_PATH.exists():
        if MILVUS_DB_PATH.is_dir():
            shutil.rmtree(MILVUS_DB_PATH)
        else:
            MILVUS_DB_PATH.unlink()

    vectors, chunks = load_faiss_vectors_and_meta()
    vector_dim = int(vectors.shape[1])

    client = MilvusClient(str(MILVUS_DB_PATH))

    if client.has_collection(COLLECTION_NAME):
        client.drop_collection(COLLECTION_NAME)

    client.create_collection(
        collection_name=COLLECTION_NAME,
        dimension=vector_dim,
        metric_type="IP",
        auto_id=False,
    )

    batch_size = 128
    inserted = 0
    started = time.time()

    for start in range(0, len(chunks), batch_size):
        end = min(start + batch_size, len(chunks))
        records = []
        for i in range(start, end):
            chunk = chunks[i]
            records.append(
                {
                    "id": int(i),
                    "vector": vectors[i].astype("float32").tolist(),
                    "index_id": int(i),
                    "child_id": str(chunk.get("child_id", f"chunk_{i}")),
                    "parent_id": str(chunk.get("parent_id", "")),
                    "title": str(chunk.get("title", "")),
                    "text": truncate(chunk.get("text", "")),
                }
            )
        client.insert(collection_name=COLLECTION_NAME, data=records)
        inserted += len(records)
        print(f"Inserted {inserted}/{len(chunks)}")

    try:
        client.flush(collection_name=COLLECTION_NAME)
    except Exception:
        pass

    results = {
        "milvus_uri": str(MILVUS_DB_PATH),
        "collection_name": COLLECTION_NAME,
        "source_faiss_index": str(FAISS_INDEX_PATH),
        "source_faiss_meta": str(FAISS_META_PATH),
        "loaded_child_chunks": len(chunks),
        "vector_count": int(vectors.shape[0]),
        "vector_dim": vector_dim,
        "inserted_count": inserted,
        "metric_type": "IP",
        "elapsed_seconds": time.time() - started,
        "first_child": {
            "child_id": chunks[0].get("child_id"),
            "parent_id": chunks[0].get("parent_id"),
            "title": chunks[0].get("title"),
        },
        "overall_pass": bool(inserted == len(chunks) == int(vectors.shape[0]) and vector_dim > 0),
    }

    write_json(BUILD_RESULTS_PATH, results)
    write_md(BUILD_REPORT_PATH, "Milvus Lite Index Build Report", results)

    return results


def test_milvus_search() -> Dict[str, Any]:
    print("[3/8] Testing direct Milvus vector search")
    from pymilvus import MilvusClient

    vectors, chunks = load_faiss_vectors_and_meta()
    query_index_id = 0
    expected_child_id = str(chunks[query_index_id].get("child_id"))

    client = MilvusClient(str(MILVUS_DB_PATH))
    raw = client.search(
        collection_name=COLLECTION_NAME,
        data=[vectors[query_index_id].astype("float32").tolist()],
        anns_field="vector",
        limit=5,
        output_fields=["child_id", "parent_id", "title", "text", "index_id"],
    )

    top_results = []
    for rank, hit in enumerate(raw[0], start=1):
        entity = hit.get("entity", {}) if isinstance(hit, dict) else {}
        top_results.append(
            {
                "rank": rank,
                "id": hit.get("id") if isinstance(hit, dict) else None,
                "score": hit.get("distance") if isinstance(hit, dict) else None,
                "child_id": entity.get("child_id"),
                "parent_id": entity.get("parent_id"),
                "title": entity.get("title"),
                "index_id": entity.get("index_id"),
            }
        )
       

    top1_child_id = top_results[0]["child_id"] if top_results else None

    results = {
        "milvus_uri": str(MILVUS_DB_PATH),
        "collection_name": COLLECTION_NAME,
        "query_index_id": query_index_id,
        "expected_child_id": expected_child_id,
        "top1_child_id": top1_child_id,
        "top1_matches_expected": top1_child_id == expected_child_id,
        "returned_count": len(top_results),
        "top_results": top_results,
        "overall_pass": bool(top1_child_id == expected_child_id),
    }

    write_json(SEARCH_RESULTS_PATH, results)
    write_md(SEARCH_REPORT_PATH, "Milvus Lite Vector Search Test Report", results)

    return results


def test_milvus_store_backend() -> Dict[str, Any]:
    print("[4/8] Testing MilvusVectorStore backend wrapper")
    from industrial_rag_service.milvus_store import MilvusVectorStore
    from industrial_rag_service.vector_store import VectorSearchResult

    vectors, chunks = load_faiss_vectors_and_meta()
    query_index_id = 0
    expected_child_id = str(chunks[query_index_id].get("child_id"))

    store = MilvusVectorStore(
        uri=str(MILVUS_DB_PATH),
        collection_name=COLLECTION_NAME,
        load_embedding_model=False,
    )
    store.load()
    results = store.search_by_embedding(
        query_embedding=vectors[query_index_id].astype("float32").tolist(),
        top_k=5,
        source_query=f"faiss_vector_index_{query_index_id}",
    )
    store.close()

    top1_child_id = results[0].child_id if results else None
    result_type_check = all(isinstance(item, VectorSearchResult) for item in results)

    output = {
        "milvus_uri": str(MILVUS_DB_PATH),
        "collection_name": COLLECTION_NAME,
        "expected_child_id": expected_child_id,
        "top1_child_id": top1_child_id,
        "top1_matches_expected": top1_child_id == expected_child_id,
        "result_type_check": result_type_check,
        "returned_count": len(results),
        "top_results": [
            {
                "rank": item.rank,
                "score": item.score,
                "child_id": item.child_id,
                "parent_id": item.parent_id,
                "title": item.title,
                "index_id": item.index_id,
            }
            for item in results
        ],
        "overall_pass": bool(top1_child_id == expected_child_id and result_type_check),
    }

    write_json(STORE_RESULTS_PATH, output)
    write_md(STORE_REPORT_PATH, "MilvusVectorStore Backend Test Report", output)

    return output


def compile_python_files() -> Dict[str, Any]:
    print("[5/8] Compiling Python files")
    import py_compile

    files = [
        "industrial_rag_service/milvus_store.py",
        "industrial_rag_service/vector_store_factory.py",
        "industrial_rag_service/app.py",
        "industrial_rag_service/retriever.py",
        "scripts/200_build_milvus_p2_all_in_one.py",
    ]

    rows = []
    for item in files:
        path = PROJECT_ROOT / item
        try:
            py_compile.compile(str(path), doraise=True)
            rows.append({"path": item, "compile_ok": True, "error": None})
        except Exception as exc:
            rows.append({"path": item, "compile_ok": False, "error": repr(exc)})

    return {
        "python_compile_total": len(rows),
        "python_compile_ok": sum(1 for row in rows if row["compile_ok"]),
        "python_compile_errors": sum(1 for row in rows if not row["compile_ok"]),
        "items": rows,
    }


def check_factory_milvus() -> Dict[str, Any]:
    print("[6/8] Checking factory backend=milvus construction")
    import yaml
    from industrial_rag_service.vector_store_factory import create_vector_store

    cfg = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    cfg["retrieval"]["backend"] = "milvus"

    store = create_vector_store(PROJECT_ROOT, cfg)
    class_name = store.__class__.__name__
    store.close()

    return {
        "backend": "milvus",
        "expected_class": "MilvusVectorStore",
        "actual_class": class_name,
        "overall_pass": class_name == "MilvusVectorStore",
    }


def final_check(build: Dict[str, Any], search: Dict[str, Any], store: Dict[str, Any], compile_result: Dict[str, Any], factory: Dict[str, Any]) -> Dict[str, Any]:
    print("[7/8] Writing final Milvus backend check report")

    expected_files = [
        "industrial_rag_service/milvus_store.py",
        "industrial_rag_service/vector_store_factory.py",
        "industrial_rag_service/config.yaml",
        "requirements_rag_service.txt",
        "scripts/200_build_milvus_p2_all_in_one.py",
        "outputs/hierarchical_rag/milvus/milvus_child_store.db",
        "outputs/hierarchical_rag/milvus/milvus_index_build_results.json",
        "outputs/hierarchical_rag/milvus/milvus_vector_search_test_results.json",
        "outputs/hierarchical_rag/milvus/milvus_store_backend_test_results.json",
    ]

    file_rows = []
    for item in expected_files:
        path = PROJECT_ROOT / item
        file_rows.append(
            {
                "path": item,
                "exists": path.exists(),
                "size_bytes": path.stat().st_size if path.exists() and path.is_file() else None,
            }
        )

    output = {
        "summary": {
            "total_expected_files": len(file_rows),
            "total_existing_files": sum(1 for row in file_rows if row["exists"]),
            "total_missing_files": sum(1 for row in file_rows if not row["exists"]),
            "python_compile_total": compile_result["python_compile_total"],
            "python_compile_ok": compile_result["python_compile_ok"],
            "python_compile_errors": compile_result["python_compile_errors"],
            "build_pass": build["overall_pass"],
            "search_pass": search["overall_pass"],
            "store_backend_pass": store["overall_pass"],
            "factory_milvus_pass": factory["overall_pass"],
        },
        "expected_files": file_rows,
        "build_results": build,
        "search_results": search,
        "store_backend_results": store,
        "python_compile": compile_result,
        "factory_milvus": factory,
    }

    output["summary"]["overall_pass"] = bool(
        output["summary"]["total_missing_files"] == 0
        and output["summary"]["python_compile_errors"] == 0
        and output["summary"]["build_pass"]
        and output["summary"]["search_pass"]
        and output["summary"]["store_backend_pass"]
        and output["summary"]["factory_milvus_pass"]
    )

    write_json(CHECK_RESULTS_PATH, output)
    write_md(CHECK_REPORT_PATH, "Milvus Backend File Check Report", output)

    return output


def make_package() -> None:
    print("[8/8] Packaging P2 Milvus backend update")
    PACKAGE_DIR.mkdir(parents=True, exist_ok=True)

    include_paths = [
        "industrial_rag_service/milvus_store.py",
        "industrial_rag_service/vector_store_factory.py",
        "industrial_rag_service/config.yaml",
        "requirements_rag_service.txt",
        "scripts/200_build_milvus_p2_all_in_one.py",
        "outputs/hierarchical_rag/milvus",
        "outputs/reports/milvus_backend_file_check_results.json",
        "outputs/reports/milvus_backend_file_check_report.md",
    ]

    if PACKAGE_PATH.exists():
        PACKAGE_PATH.unlink()

    with tarfile.open(PACKAGE_PATH, "w:gz") as tar:
        for item in include_paths:
            path = PROJECT_ROOT / item
            if path.exists():
                tar.add(path, arcname=item)

    print(f"Package saved: {PACKAGE_PATH}")


def main() -> None:
    started = time.time()

    MILVUS_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    install_deps()
    write_code_files()
    build = build_milvus_index()
    search = test_milvus_search()
    store = test_milvus_store_backend()
    compile_result = compile_python_files()
    factory = check_factory_milvus()
    check = final_check(build, search, store, compile_result, factory)
    make_package()

    summary = {
        "elapsed_seconds": time.time() - started,
        "package_path": str(PACKAGE_PATH),
        "overall_pass": check["summary"]["overall_pass"],
        "build_pass": build["overall_pass"],
        "search_pass": search["overall_pass"],
        "store_backend_pass": store["overall_pass"],
        "factory_milvus_pass": factory["overall_pass"],
    }

    print("=" * 80)
    print("P2 Milvus Lite backend build summary")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("=" * 80)

    if not summary["overall_pass"]:
        raise RuntimeError("P2 Milvus backend build failed.")

    print("P2 Milvus Lite backend build passed.")


if __name__ == "__main__":
    main()
