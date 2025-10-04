import wikipedia

class CalculatorTool:
    def run(self, query: str) -> str:
        try:
            # Extract just numbers/operators
            expr = "".join(ch for ch in query if ch.isdigit() or ch in "+-*/().")
            return str(eval(expr))
        except Exception:
            return "Calculation error"

class WikipediaTool:
    def run(self, query: str) -> str:
        try:
            return wikipedia.summary(query, sentences=2)
        except Exception as e:
            return f"Wikipedia error: {e}"

class Toolset:
    def __init__(self):
        self.tools = {
            "calculator": CalculatorTool(),
            "search": WikipediaTool(),
        }

    def get(self, name: str):
        return self.tools.get(name)
