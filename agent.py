import re, json

class Agent:
    def __init__(self, llm, prompt_manager, tools):
        self.llm = llm
        self.prompt_manager = prompt_manager
        self.tools = tools

    def run(self, question: str) -> str:
        # Step 1: Tool planning
        tool_prompt = self.prompt_manager.build_tool_prompt(question)
        raw_plan = self.llm.generate(tool_prompt)

        # Step 2: Extract tool plan
        match = re.search(r"\[[\s\S]*?\]", raw_plan)
        if not match:
            return f"❌ Tool planner failed:\n{raw_plan}"
        try:
            plan = json.loads(match.group(0))
        except Exception as e:
            return f"❌ Invalid JSON:\n{raw_plan}\nError: {e}"

        # Step 3: Run tools
        results = []
        for step in plan:
            tool_name = step.get("tool")
            query = step.get("query", "")
            tool = self.tools.get(tool_name)
            if tool:
                results.append({
                    "tool": tool_name,
                    "query": query,
                    "result": tool.run(query)
                })
            else:
                results.append({"tool": tool_name, "query": query, "result": "Unknown tool"})

        # Step 4: Final answer (short prompt!)
        final_prompt = f"Q: {question}\nResults: {results}\nA:"
        raw_answer = self.llm.generate(final_prompt).strip()

        # Step 5: Clean output
        cleaned = re.sub(r"(you are|tool results).*", "", raw_answer, flags=re.I).strip()
        return cleaned or raw_answer
