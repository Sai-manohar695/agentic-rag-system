"""
Agentic RAG System with Multi-Tool Orchestration
and Adaptive Query Routing

Agent decides which tool(s) to use based on query type:
- RAG Search      → domain knowledge questions
- Wikipedia       → factual/encyclopedic questions  
- ArXiv           → research paper questions
- Web Search      → current events / general web
- Calculator      → math expressions
"""

import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import Tool
from langchain import hub
from langchain.prompts import PromptTemplate
from tools.rag_tool import rag_search, ingest_documents
from tools.web_tools import (
    wikipedia_search,
    arxiv_search,
    web_search,
    calculator
)

load_dotenv()

# ── LLM ──────────────────────────────────────────────────
llm = ChatGroq(
    api_key     = os.getenv("GROQ_API_KEY"),
    model_name  = "llama-3.3-70b-versatile",
    temperature = 0.1,
    max_tokens  = 2048
)

# ── Tools ─────────────────────────────────────────────────
tools = [
    Tool(
        name        = "RAG_Search",
        func        = rag_search,
        description = (
            "Search the internal knowledge base using "
            "semantic similarity. Use this FIRST for questions "
            "about documents you have ingested, domain-specific "
            "knowledge, or when the user asks about uploaded content."
        )
    ),
    Tool(
        name        = "Wikipedia_Search",
        func        = wikipedia_search,
        description = (
            "Search Wikipedia for factual, encyclopedic information. "
            "Use for definitions, historical facts, well-known concepts, "
            "people, places or events."
        )
    ),
    Tool(
        name        = "ArXiv_Search",
        func        = arxiv_search,
        description = (
            "Search ArXiv for academic research papers. "
            "Use when the user asks about recent research, "
            "papers, studies, scientific findings or academic topics."
        )
    ),
    Tool(
        name        = "Web_Search",
        func        = web_search,
        description = (
            "Search the web using DuckDuckGo for current information, "
            "news, tutorials, recent events or anything not covered "
            "by other tools."
        )
    ),
    Tool(
        name        = "Calculator",
        func        = calculator,
        description = (
            "Perform mathematical calculations. "
            "Use for arithmetic, percentages, or any numeric computation. "
            "Input should be a math expression like '2 ** 10 + 100'."
        )
    )
]

# ── Custom Prompt ─────────────────────────────────────────
SYSTEM_PROMPT = """You are an intelligent AI assistant with access 
to multiple tools. You adaptively route queries to the most 
appropriate tool based on the question type.

ROUTING STRATEGY:
- Domain/document questions    → RAG_Search first
- Factual/encyclopedic         → Wikipedia_Search
- Research/academic papers     → ArXiv_Search  
- Current events/general web   → Web_Search
- Math/calculations            → Calculator
- Complex questions            → combine multiple tools

Always think step by step about which tool is most appropriate.
After getting tool results, synthesize a clear, helpful response.

{tools}

Use the following format:
Question: the input question you must answer
Thought: think about which tool to use and why
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought: {agent_scratchpad}"""

prompt = PromptTemplate.from_template(SYSTEM_PROMPT)

# ── Agent ─────────────────────────────────────────────────
agent = create_react_agent(
    llm     = llm,
    tools   = tools,
    prompt  = prompt
)

agent_executor = AgentExecutor(
    agent          = agent,
    tools          = tools,
    verbose        = True,
    max_iterations = 5,
    handle_parsing_errors = True,
    return_intermediate_steps = True
)

def run_agent(query: str) -> dict:
    """Run the agent and return response + tool trace."""
    try:
        result = agent_executor.invoke({"input": query})
        
        # Extract tool trace
        steps = result.get("intermediate_steps", [])
        tool_trace = []
        for action, observation in steps:
            tool_trace.append({
                "tool"       : action.tool,
                "input"      : action.tool_input,
                "output"     : str(observation)[:300]
            })
        
        return {
            "query"      : query,
            "answer"     : result["output"],
            "tool_trace" : tool_trace,
            "tools_used" : [t["tool"] for t in tool_trace]
        }
    except Exception as e:
        return {
            "query"      : query,
            "answer"     : f"Error: {str(e)}",
            "tool_trace" : [],
            "tools_used" : []
        }

if __name__ == "__main__":
    test_queries = [
        "What is Retrieval Augmented Generation?",
        "Find recent research papers on RAG systems",
        "What is 15% of 85000?",
    ]

    for query in test_queries:
        print("\n" + "="*60)
        print(f"QUERY: {query}")
        print("="*60)
        result = run_agent(query)
        print(f"\nFINAL ANSWER:\n{result['answer']}")
        print(f"\nTOOLS USED: {result['tools_used']}")