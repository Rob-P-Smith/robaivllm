"""
System prompt templates for web search injection into vLLM context.

Duplicated from robaiagents/agents/web_search_20250305/prompt_templates.py
to avoid cross-project imports.
"""


def build_system_prompt(query: str, search_results: list, crawled_content: list) -> str:
    """
    Build system prompt with web search results and crawled content injected.

    Args:
        query: The search query that was executed
        search_results: List of search results from Serper
        crawled_content: List of crawled URL content

    Returns:
        Formatted system prompt for vLLM
    """
    # Format search results
    results_lines = []
    for i, result in enumerate(search_results, 1):
        title = result.get("title", "No title")
        link = result.get("link", "")
        snippet = result.get("snippet", "")
        results_lines.append(f"{i}. {title}")
        results_lines.append(f"   URL: {link}")
        if snippet:
            results_lines.append(f"   {snippet}")
        results_lines.append("")

    formatted_results = "\n".join(results_lines)

    # Format crawled content
    crawled_lines = []
    for content in crawled_content:
        if isinstance(content, Exception):
            crawled_lines.append(f"[Error crawling]: {str(content)}")
            continue

        title = content.get("title", "No title")
        url = content.get("url", "")
        markdown = content.get("markdown", "")

        crawled_lines.append(f"\n### {title}")
        crawled_lines.append(f"URL: {url}")
        crawled_lines.append("")
        crawled_lines.append(markdown[:2000])  # Limit to 2000 chars per URL
        crawled_lines.append("")

    formatted_crawled = "\n".join(crawled_lines)

    system_prompt = f"""You are responding to a user query with web search results.

I performed a web search for: "{query}"

Here are the search results:

{formatted_results}

Here is detailed content from the top URLs:

{formatted_crawled}

Now respond naturally to the user's original question using the information above. Base your answer on the search results and crawled content provided."""

    return system_prompt
