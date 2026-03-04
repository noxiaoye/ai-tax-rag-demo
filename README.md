# AI 财税知识库问答（RAG）Demo（可直接发 HR）

这是一个 **RAG（检索增强生成）知识库问答**小项目：把财税资料做成可检索、可追溯引用的问答系统。
特点：先检索 TopK 资料片段，再让大模型基于片段回答，并输出引用来源，减少“幻觉”。
备注：仅用于演示学习与产品能力，不构成税务/法律建议，实际以当地最新政策与主管部门口径为准。

## 功能亮点
- 文档切分 → 向量化（Embedding）→ 向量库检索（FAISS）
- RAG：检索 TopK 片段 + 大模型基于片段生成答案
- 引用可追溯：答案末尾标注引用编号，右侧展示引用片段（文件名+chunk）
- 模型可替换：OpenAI / 其他 OpenAI 兼容 API（改 .env 即可）

## 快速开始（3 步）
1) 安装依赖（Python 3.10+）
```bash
pip install -r requirements.txt
```

2) 配置 OpenAI
复制 `.env.example` 为 `.env`，填写 OPENAI_API_KEY：
```bash
OPENAI_API_KEY=你的OpenAI_API_KEY
OPENAI_BASE_URL=https://api.openai.com/v1
CHAT_MODEL=gpt-4o-mini
EMBED_MODEL=text-embedding-3-small
TOP_K=5
```

3) 入库并启动
```bash
python ingest.py
streamlit run app.py
```

## Demo 建议问题
- 什么是进项税额抵扣？常见风险点有哪些？
- 一般纳税人与小规模纳税人主要区别是什么？
- 企业所得税税前扣除常见注意事项有哪些？
- 发票合规性风险有哪些典型场景？

## 面试 60 秒讲解
痛点：资料分散，检索慢；纯大模型问答容易幻觉、不可追溯。
方案：RAG：切分→Embedding→向量检索→把 TopK 片段喂给 LLM。
关键：强制只依据资料片段回答 + 引用；资料不足则提示补充资料。
评估：Recall@K、引用准确率、满意度、延迟与成本。
