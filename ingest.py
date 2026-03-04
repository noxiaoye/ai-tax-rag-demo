# -*- coding: utf-8 -*-
from __future__ import annotations
import os, json
import faiss
from dotenv import load_dotenv
from utils import read_text_files, split_into_chunks
from rag import embed_texts

def main():
    load_dotenv()
    root = os.path.dirname(__file__)
    data_dir = os.path.join(root, "data")
    vdb_dir = os.path.join(root, "vectordb")
    os.makedirs(vdb_dir, exist_ok=True)

    chunk_size = int(os.getenv("CHUNK_SIZE", "900"))
    overlap = int(os.getenv("CHUNK_OVERLAP", "180"))
    embed_model = os.getenv("EMBED_MODEL", "text-embedding-3-small")

    raw_docs = read_text_files(data_dir)
    chunks = []
    for text, fname in raw_docs:
        chunks.extend(split_into_chunks(text, fname, chunk_size, overlap))
    if not chunks:
        raise ValueError("data/ 下没有可用文本，请放入 txt/md 文件后重试")

    texts = [c.text for c in chunks]
    meta = [{"text": c.text, "source": c.source, "chunk_id": c.chunk_id} for c in chunks]

    vecs = embed_texts(texts, model=embed_model)
    faiss.normalize_L2(vecs)
    dim = vecs.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vecs)

    faiss.write_index(index, os.path.join(vdb_dir, "index.faiss"))
    with open(os.path.join(vdb_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"✅ 已入库：{len(chunks)} 个片段，向量维度：{dim}")
    print("下一步：streamlit run app.py")

if __name__ == "__main__":
    main()
