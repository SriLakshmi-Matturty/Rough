from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent, Tool
from tools import CalculatorTool, SearchTool
from langchain.prompts import PromptTemplate

class Agent:
    def __init__(self, llm_model="mistralai/Mistral-7B-Instruct-v0.2", serpapi_key=None):
        # Initialize tools
        self.tools = [
            CalculatorTool(),
            SearchTool(api_key=serpapi_key)
        ]

        # Wrap tools for LangChain
        self.lc_tools = [Tool(name=t.name, func=t.run, description=t.description) for t in self.tools]

        # Use HuggingFace model as LangChain chat model
        from langchain.llms import HuggingFacePipeline
        from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
        tokenizer = AutoTokenizer.from_pretrained(llm_model)
        model = AutoModelForCausalLM.from_pretrained(llm_model, torch_dtype="auto", device_map="auto")
        pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=128)
        self.llm = HuggingFacePipeline(pipeline=pipe)

        # Initialize LangChain agent with tools
        self.agent_executor = initialize_agent(
            tools=self.lc_tools,
            llm=self.llm,
            agent="zero-shot-react-description",
            verbose=True
        )

    def run(self, question: str):
        return self.agent_executor.run(question)
