from langchain.tools import BaseTool
from pydantic import Field
from typing import Optional

class CalculatorTool(BaseTool):
    name: str = "calculator"
    description: str = "Use this tool to perform arithmetic operations. Input should be a valid numeric expression."

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
    name: str = "search"
    description: str = "Use this tool to search factual information online."
    api_key: Optional[str] = Field(default=None)

    def _run(self, query: str):
        if not query:
            return "No query provided"
        # Replace with real API call if needed
        return f"[SearchTool] Simulated search result for: {query}"

    async def _arun(self, query: str):
        raise NotImplementedError("SearchTool does not support async")
