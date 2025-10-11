from langchain.tools import BaseTool

class CalculatorTool(BaseTool):
    name = "calculator"
    description = "Use this tool to perform arithmetic operations. Input should be a valid numeric expression."

    def _run(self, expr: str):
        try:
            allowed_chars = "0123456789+-*/.() "
            if not all(c in allowed_chars for c in expr):
                return "Calculator Error: Invalid characters"
            return str(eval(expr))
        except Exception as e:
            return f"Calculator Error: {e}"

    async def _arun(self, expr: str):
        raise NotImplementedError("CalculatorTool does not support async")

class SearchTool(BaseTool):
    name = "search"
    description = "Use this tool to search factual information online."

    def __init__(self, api_key=None):
        self.api_key = api_key
        super().__init__()

    def _run(self, query: str):
        if not query:
            return "No query provided"
        # Replace with real API call or Wikipedia API
        return f"[SearchTool] Simulated search result for: {query}"

    async def _arun(self, query: str):
        raise NotImplementedError("SearchTool does not support async")
