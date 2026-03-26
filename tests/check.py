import langchain
import langchain.agents
print(f"LangChain 版本: {langchain.__version__}")
print(f"Agents 目录下的可用内容: {dir(langchain.agents)}")