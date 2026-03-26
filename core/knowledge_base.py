# core/knowledge_base.py
import os
from langchain_community.document_loaders import PyPDFDirectoryLoader # 替换原来的 DirectoryLoader 和 PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma


def build_or_load_vector_db(data_path, persist_path, embeddings):
    """
    通用逻辑：如果本地有持久化数据库则加载，否则读取 data 文件夹下的所有 PDF 构建。
    """
    # 1. 检查持久化目录是否存在且不为空
    if os.path.exists(persist_path) and os.listdir(persist_path):
        print(f"📦 发现现有数据库，正在从 {persist_path} 加载...")
        return Chroma(persist_directory=persist_path, embedding_function=embeddings)

    # 2. 如果没有现有数据库，则开始构建
    print("📚 正在构建新知识库...")
    if not os.path.exists(data_path):
        os.makedirs(data_path)
        print(f"警告：{data_path} 文件夹已创建，请放入 PDF 文件后重新运行。")
        return None

    # 替换原来的 loader = DirectoryLoader(...)
    loader = PyPDFDirectoryLoader(data_path)
    documents = loader.load()

    if not documents:
        print("未发现 PDF 文档，请在 data 文件夹中放入 PDF 文件。")
        return None

    # 3. 切分文档 (增加重叠度 Overlap 确保上下文不丢失)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=150
    )
    split_docs = text_splitter.split_documents(documents)

    # 4. 创建向量数据库并存入磁盘
    db = Chroma.from_documents(
        documents=split_docs,
        embedding=embeddings,
        persist_directory=persist_path
    )
    print(f"✅ 知识库构建完成，共处理 {len(split_docs)} 个片段。")
    return db