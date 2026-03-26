# core/agent_logic.py
import re
from core.prompts import SYSTEM_PROMPT

def parse_llm_output(text):
    """精确对应源代码的正则解析逻辑"""
    action_match = re.search(r"Action\s*[:：]\s*(.*)", text)
    action_input_match = re.search(r"Action Input\s*[:：]\s*(.*)", text)
    action = action_match.group(1).strip().strip('[]"\'') if action_match else None
    action_input = action_input_match.group(1).strip().strip('[]"\'') if action_input_match else None
    return action, action_input

def simple_agent(user_query, tools_map, llm):
    """集成自省与模糊匹配的 Agent 核心，精确对应源代码逻辑"""
    tool_descs = "\n".join([f"- {name}: {t.description}" for name, t in tools_map.items()])
    current_system_prompt = SYSTEM_PROMPT.replace("{tool_descriptions}", tool_descs)

    print(f"\n🚀 Agent 正在深度解析中...")
    history = f"User Question: {user_query}"
    last_action_input = None

    for i in range(15):
        full_input = f"{current_system_prompt}\n\n{history}\nThought:"
        res_obj = llm.invoke(full_input)
        response = res_obj.content
        print(f"\n🤔 思考进程 [{i + 1}]:\n{response}")

        # 1. 模糊匹配最终回答
        final_keywords = ["Final Answer:", "最终回答:", "最终答案:", "结论是:", "结论："]
        found_final = any(k in response for k in final_keywords)
        if "最终答案" in response and "Action" not in response:
            found_final = True

        if found_final:
            answer = response
            for k in final_keywords:
                if k in answer:
                    answer = answer.split(k)[-1]
            return answer.strip()

        # 2. 解析并执行 Action
        action, action_input = parse_llm_output(response)

        if action and action in tools_map:
            if action_input == last_action_input:
                print("⚠️ 检测到重复操作...")
                observation = "【系统提示】：你已经在重复执行相同操作。请直接以 'Final Answer: [内容]' 给出结论。"
                history += f"\nThought: {response}\nObservation: {observation}"
                continue

            print(f"🛠️ 执行工具: {action}")
            try:
                observation = tools_map[action].run(action_input)
                last_action_input = action_input

                # 废话检测逻辑 (精确对应源码位置)
                if action == "knowledge_base":
                    useless_words = ["next version", "will be reported", "to provide insights", "is refining"]
                    if any(w in str(observation).lower() for w in useless_words):
                        observation = "【系统警告】：你刚才搜到的是占位符内容！请改用更专业的英文术语重新执行 knowledge_base。"
                        history += f"\nThought: {response}\nObservation: {observation}"
                        continue

                clean_thought = response.split("Action")[0].strip()
                history += f"\nThought: {clean_thought}\nAction: {action}\nAction Input: {action_input}\nObservation: {observation}"
                continue
            except Exception as e:
                history += f"\nThought: {response}\nObservation: 工具运行出错: {str(e)}"
                continue

        # 3. 兜底逻辑
        history += f"\nThought: {response}\nObservation: 我没看到你的 Action 指令或 Final Answer。请务必带上 'Final Answer:' 标签。"

    return "任务处理超时。"