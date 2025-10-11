# Optional with LangChain: mostly handled internally
class PromptManager:
    @staticmethod
    def build_final_prompt(question: str, tool_result_summary: str) -> str:
        return f"Use ONLY the information below.\nTool output:\n{tool_result_summary}\nQuestion: {question}\nAnswer:"
