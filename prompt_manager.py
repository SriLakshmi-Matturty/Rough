# prompt_manager.py

class PromptManager:
    def __init__(self):
        pass

    def build_tool_prompt(self, question: str) -> str:
        return f"""
You are a tool planner.

Available tools:
- calculator
- wikipedia

Respond ONLY with a JSON list of steps.
Each step must have exactly this format:
[
  {{
    "tool": "tool_name",
    "query": "string query here"
  }}
]

Example:
[
  {{"tool": "wikipedia", "query": "Albert Einstein"}}
]

Question: {question}
JSON:
"""

    def build_final_prompt(self, question: str, results: list) -> str:
        return f"""
You are an answer composer.

Question: {question}

Tool results:
{results}

Write the final short answer using the tool results only.
Do NOT invent facts.
Answer:
"""
