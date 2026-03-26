# tools/tool_definitions.py
import os
import datetime
from langchain_core.tools import Tool
from langchain_community.tools import DuckDuckGoSearchRun


def get_tools_map(vector_db):
    """
    初始化并返回 Agent 可调用的工具字典。
    集成了增强型 RAG 检索逻辑、联网搜索、文件清单及系统时间。
    """

    # --- 1. 增强型知识库检索工具 (RAG) ---
    # 默认检索 k=5 个片段
    retriever = vector_db.as_retriever(search_kwargs={"k": 5})

    def knowledge_base_func(query: str) -> str:
        """
        内部逻辑：针对总结性问题自动扩充检索深度，并按文档物理顺序重排片段。
        """
        # 语义识别：判断是否为全局总结性问题
        summary_keywords = ["summary", "abstract", "overview", "总结", "摘要", "全文", "讲了什么", "主要内容"]
        is_global_query = any(kw in query.lower() for kw in summary_keywords)

        # 如果是总结意图，动态增加检索块数量至 10 个，以覆盖更多上下文
        k_value = 10 if is_global_query else 5

        # 执行检索 (LangChain 1.x 标准使用 invoke)
        docs = retriever.invoke(query, k=k_value)

        if not docs:
            return "知识库中未找到相关内容。请尝试更换更具体的关键词。"

        # 【核心优化】：按文档页码(Page)和索引(Index)进行物理顺序重排
        # 这样交给 LLM 的片段是“顺着读”的，不会逻辑颠倒
        try:
            docs.sort(key=lambda x: (x.metadata.get('page', 0), x.metadata.get('index', 0)))
        except Exception:
            pass  # 如果没有 metadata 则保持原样

        # 格式化输出，标注出处
        formatted_results = []
        for i, doc in enumerate(docs):
            page_num = doc.metadata.get('page', '未知')
            content = f"[来源片段 {i + 1} | 第 {page_num} 页]:\n{doc.page_content}"
            formatted_results.append(content)

        return "\n\n".join(formatted_results)

    kb_tool = Tool(
        name="knowledge_base",
        func=knowledge_base_func,
        description=(
            "深度语义检索工具。用于查询本地PDF文档或个人笔记。支持跨语言检索。"
            "当你需要回答事实细节或总结全文时必用。如果是总结全文，请输入英文关键词如 'abstract and main contributions'。"
        )
    )

    # --- 2. 互联网搜索工具 ---
    search_tool = DuckDuckGoSearchRun()

    # --- 3. 查看本地文件列表工具 ---
    def list_files_func(query: str = None) -> str:
        data_dir = './data'
        if not os.path.exists(data_dir):
            return "错误：data 文件夹不存在。请创建该文件夹并放入 PDF 文件。"

        files = [f for f in os.listdir(data_dir) if not f.startswith('.')]
        if not files:
            return "知识库 data 文件夹目前是空的，请先上传 PDF 文件。"

        return f"当前知识库中的原始文件清单：\n- " + "\n- ".join(files)

    list_tool = Tool(
        name="list_data_files",
        func=list_files_func,
        description="元数据查询工具。用于查看本地知识库中有哪些文件名。当你不知道有哪些文件可供参考，或需要确认特定文件名时使用。"
    )

    # --- 4. 获取系统时间工具 ---
    def get_time_func(query: str = None) -> str:
        now = datetime.datetime.now()
        # 返回包含日期、时间和星期的详细字符串
        return f"当前系统精确时间：{now.strftime('%Y-%m-%d %H:%M:%S')} {now.strftime('%A')}"

    time_tool = Tool(
        name="get_current_time",
        func=get_time_func,
        description="时间工具。获取当前真实的日期和时间。当涉及‘今天’、‘现在’或具体日期计算时，必须使用此工具获取基准时间。"
    )

    # 返回工具字典映射
    return {
        "knowledge_base": kb_tool,
        "web_search": search_tool,
        "list_data_files": list_tool,
        "get_current_time": time_tool
    }