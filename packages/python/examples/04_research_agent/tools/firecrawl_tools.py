"""Raw Firecrawl tools — used by sub-agents, not the orchestrator."""

from __future__ import annotations

import os

from firecrawl import Firecrawl

from dendrux import tool


def _get_client() -> Firecrawl:
    return Firecrawl(api_key=os.environ["FIRECRAWL_API_KEY"])


@tool()
async def firecrawl_search(query: str) -> str:
    """Search the web for a query. Returns titles, URLs, and snippets of top results."""
    client = _get_client()
    results = client.search(query, limit=5)

    # SDK v4 returns SearchData with .web list of SearchResult objects
    items = results.web if results.web else []
    if not items:
        return f"No results found for: {query}"

    formatted = []
    for i, item in enumerate(items, 1):
        title = item.title or "No title"
        url = item.url or ""
        snippet = (item.description or "")[:300]
        formatted.append(f"[{i}] {title}\n    URL: {url}\n    {snippet}")

    return "\n\n".join(formatted)


@tool()
async def firecrawl_scrape(url: str) -> str:
    """Scrape a URL and return its content as clean markdown. Use this to read full articles."""
    client = _get_client()
    result = client.scrape(url)

    # SDK v4 returns ScrapeData with .markdown attribute
    markdown = result.markdown or ""
    if not markdown:
        return f"Could not extract content from: {url}"

    # Truncate to ~8000 chars to stay within reasonable tool result size
    if len(markdown) > 8000:
        markdown = markdown[:8000] + "\n\n[...content truncated...]"

    return markdown
