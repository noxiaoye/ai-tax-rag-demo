# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import List, Dict, Any, Tuple
import os
import numpy as np
import faiss
from openai import OpenAI

def _client() -> OpenAI:
    return OpenAI(
        api_key=os.getenv("OPENAI_API_KEY", ""),
        base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
    )

def embed_texts(texts: List[str], model: str) -> np.ndarray:
    c = _client()
    resp = c.embeddings.create(model=model, input=texts)
    vecs = [d.embedding for d in resp.data]
    return np.asarray(vecs, dtype="float32")

def load_index(vdb_dir: str) -> Tuple[faiss.Index, List[Dict[str, Any]]]:
    index_path = os.path.join(vdb_dir, "index.faiss")
    meta_path = os.path.join(vdb_dir, "meta.json")
    if not (os.path.exists(index_path) and os.path.exists(meta_path)):
        raise FileNotFoundError("未找到向量库，请先运行：python ingest.py")
    index = faiss.read_index(index_path)
    import json
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    return index, meta

def retrieve(question: str, index: faiss.Index, meta: List[Dict[str, Any]], embed_model: str, top_k: int) -> List[Dict[str, Any]]:
    qv = embed_texts([question], model=embed_model)
    faiss.normalize_L2(qv)
    _, I = index.search(qv, top_k)
    hits = []
    for idx in I[0]:
        if idx == -1:
            continue
        hits.append(meta[idx])
    return hits

def chat_answer(question: str, context_blocks: List[Dict[str, Any]], model: str) -> str:
    ctx_lines = []
    for i, b in enumerate(context_blocks, start=1):
        ctx_lines.append(f"[{i}] 来源：{b['source']}（chunk {b['chunk_id']}）\n{b['text']}")
    ctx = "\n\n".join(ctx_lines)

    system = (
        "你是一名严谨的财税知识库问答助手。"
        "你必须只依据【资料片段】回答，不要编造。"
        "资料不足就说明需要补充哪些资料。"
        "回答末尾给出引用编号，例如：……[1][2]。"
        "并提示：不构成税务建议，具体以最新政策与主管部门口径为准。"
    )
    user = f"问题：{question}\n\n【资料片段】\n{ctx}\n\n请输出：1) 简洁结论 2) 关键依据 3) 风险/注意事项"
    c = _client()
    resp = c.chat.completions.create(
        model=model,
        messages=[{"role":"system","content":system},{"role":"user","content":user}],
        temperature=0.2,
    )
    return resp.choices[0].message.content
