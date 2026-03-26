# SmartKnowledge-Agent 🤖
**基于 RAG 与自定义 ReAct 架构的智能知识库 Agent**

## 🚀 项目简介
这是一个基于 LangChain 1.x LTS 架构实现的个人知识库智能体。它不仅能检索本地 PDF 文档，还能通过联网搜索、系统工具调用来回答复杂问题。

项目核心亮点在于**不依赖高层封装**，而是手动实现了 ReAct 思考循环，并在代码层集成了**自省拦截器**，有效解决了轻量级大模型（如 GLM-4-Flash）常见的“幻觉”与“任务偷懒”问题。

## ✨ 核心特性
- **模块化架构**：核心逻辑（Core）、工具箱（Tools）、数据层（Data）完全解耦。
- **自定义 ReAct 引擎**：手动实现 `Thought -> Action -> Observation -> Final Answer` 闭环。
- **智能防御拦截**：自动检测模型生成的“占位符”和“敷衍式回答”，强制模型进行深度检索。
- **跨语言支持**：支持英文文档检索与中文总结输出。

## 🛠️ 技术栈
- **LLM**: ZhipuAI (GLM-4-Flash)
- **Vector DB**: ChromaDB (langchain-chroma)
- **Framework**: LangChain 1.x LTS, Python 3.13

## 📦 快速开始
1. **安装依赖**:
   ```bash
   pip install -r requirements.txt