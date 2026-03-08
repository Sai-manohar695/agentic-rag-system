"""
Tools 2-4: Wikipedia, ArXiv and Web Search Tools
"""

import arxiv
import wikipedia
from ddgs import DDGS
import requests
from bs4 import BeautifulSoup

def wikipedia_search(query: str,
                     sentences: int = 5) -> str:
    """Search Wikipedia and return summary."""
    try:
        wikipedia.set_lang("en")
        result = wikipedia.summary(
            query,
            sentences   = sentences,
            auto_suggest = True
        )
        page = wikipedia.page(query, auto_suggest=True)
        return (f"**Wikipedia: {page.title}**\n"
                f"URL: {page.url}\n\n{result}")
    except wikipedia.exceptions.DisambiguationError as e:
        try:
            result = wikipedia.summary(
                e.options[0], sentences=sentences)
            return f"**Wikipedia (disambiguated):**\n{result}"
        except:
            return f"Wikipedia disambiguation: {e.options[:5]}"
    except Exception as e:
        return f"Wikipedia search failed: {str(e)}"


def arxiv_search(query: str,
                 max_results: int = 3) -> str:
    """Search ArXiv for recent papers."""
    try:
        client = arxiv.Client()
        search = arxiv.Search(
            query      = query,
            max_results = max_results,
            sort_by    = arxiv.SortCriterion.Relevance
        )
        results = list(client.results(search))

        if not results:
            return "No ArXiv papers found."

        papers = []
        for r in results:
            authors = ", ".join(
                str(a) for a in r.authors[:3])
            papers.append(
                f"**{r.title}**\n"
                f"Authors : {authors}\n"
                f"Published: {r.published.strftime('%Y-%m-%d')}\n"
                f"URL     : {r.entry_id}\n"
                f"Abstract: {r.summary[:300]}..."
            )
        return "\n\n---\n\n".join(papers)
    except Exception as e:
        return f"ArXiv search failed: {str(e)}"


def web_search(query: str,
               max_results: int = 4) -> str:
    """DuckDuckGo web search."""
    try:
        results = []
        with DDGS() as ddgs:
            for r in ddgs.text(
                query,
                max_results = max_results
            ):
                results.append(
                    f"**{r['title']}**\n"
                    f"URL    : {r['href']}\n"
                    f"Snippet: {r['body'][:200]}..."
                )
        if not results:
            return "No web results found."
        return "\n\n---\n\n".join(results)
    except Exception as e:
        return f"Web search failed: {str(e)}"


def calculator(expression: str) -> str:
    """Safe calculator for math expressions."""
    try:
        allowed = set(
            '0123456789+-*/()., '
            'abcdefghijklmnopqrstuvwxyz'
        )
        safe_expr = ''.join(
            c for c in expression.lower()
            if c in allowed
        )
        result = eval(safe_expr, {"__builtins__": {}}, {
            "abs": abs, "round": round,
            "min": min,  "max": max,
            "sum": sum,  "pow": pow
        })
        return f"Result: {result}"
    except Exception as e:
        return f"Calculator error: {str(e)}"


if __name__ == "__main__":
    print("Testing Wikipedia...")
    print(wikipedia_search("Retrieval Augmented Generation"))
    print("\nTesting ArXiv...")
    print(arxiv_search("RAG retrieval augmented generation"))
    print("\nTesting Web Search...")
    print(web_search("LangChain agents 2024"))
    print("\nTesting Calculator...")
    print(calculator("2 ** 10 + 100"))