# main.py
import os
from dotenv import load_dotenv
from langchain_community.chat_models import ChatZhipuAI
from langchain_community.embeddings import ZhipuAIEmbeddings

# 导入所有解耦的模块
from core.knowledge_base import build_or_load_vector_db
from core.agent_logic import simple_agent
from tools.tool_definitions import get_tools_map


def main():
    load_dotenv()
    api_key = os.getenv("ZHIPUAI_API_KEY")

    embeddings = ZhipuAIEmbeddings(model="embedding-2", api_key=api_key)
    llm = ChatZhipuAI(model="glm-4-flash", temperature=0.1, api_key=api_key)

    # 调用 core 里的逻辑
    vector_db = build_or_load_vector_db(
        data_path="./data",
        persist_path="./vector_storage",
        embeddings=embeddings
    )

    tools = get_tools_map(vector_db)

    print("\n✅ 通用 Agent 已就绪。")
    while True:
        q = input("\n👤 用户提问: ")
        if q.lower() == 'quit': break
        if q.strip():
            result = simple_agent(q, tools, llm)
            print(f"\n✨ 最终回答: {result}")


if __name__ == "__main__":
    main()