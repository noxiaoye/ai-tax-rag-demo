# -*- coding: utf-8 -*-
from __future__ import annotations
import os
import streamlit as st
from dotenv import load_dotenv
from rag import load_index, retrieve, chat_answer

def main():
    load_dotenv()
    st.set_page_config(page_title="AI 财税知识库问答（RAG）", layout="wide")
    st.title("AI 财税知识库问答（RAG）Demo")
    st.caption("先运行：python ingest.py 构建向量库；再运行：streamlit run app.py")

    root = os.path.dirname(__file__)
    vdb_dir = os.path.join(root, "vectordb")
    embed_model = os.getenv("EMBED_MODEL", "text-embedding-3-small")
    chat_model = os.getenv("CHAT_MODEL", "gpt-4o-mini")
    top_k = int(os.getenv("TOP_K", "5"))

    col1, col2 = st.columns([2, 1], gap="large")
    with col1:
        q = st.text_input("请输入财税问题（示例：什么是进项税额抵扣？）")
        if st.button("检索并回答", type="primary"):
            if not q.strip():
                st.warning("请输入问题")
                return
            try:
                index, meta = load_index(vdb_dir)
            except Exception as e:
                st.error(str(e))
                return
            hits = retrieve(q, index, meta, embed_model, top_k)
            if not hits:
                st.warning("未检索到相关片段，请补充知识库资料。")
                return
            with st.spinner("生成中…"):
                ans = chat_answer(q, hits, chat_model)
            st.subheader("回答")
            st.write(ans)
            st.session_state["last_hits"] = hits

    with col2:
        st.subheader("引用片段（TopK）")
        hits = st.session_state.get("last_hits", [])
        if not hits:
            st.info("点击左侧“检索并回答”后展示引用片段。")
        else:
            for i, h in enumerate(hits, start=1):
                with st.expander(f"[{i}] {h['source']} (chunk {h['chunk_id']})", expanded=(i <= 2)):
                    st.write(h["text"])

if __name__ == "__main__":
    main()
