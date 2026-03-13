import asyncio
import os

from vision_deepresearch_async_workflow.tools.shared import (
    DeepResearchTool,
    get_cache_async,
    get_cache_key,
    log_search,
    log_tool_event,
    run_with_retries_async,
    set_cache_async,
)


class SearchTool(DeepResearchTool):
    """Web search tool using Serper API (google.serper.dev)."""

    MAX_URLS = 10

    def __init__(self):
        super().__init__(
            name="search",
            description="Performs batched web searches: supply an array 'query'; the tool retrieves the top 10 results for each query in one call.",
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Array of query strings. Include multiple complementary search queries in a single call.",
                    },
                },
                "required": ["query"],
            },
        )
        self.serper_api_key = os.getenv("SERPER_API_KEY", "")
        self.serper_search_url = "https://google.serper.dev/search"

    def contains_chinese(self, text: str) -> bool:
        """Check if text contains Chinese characters."""
        return any("\u4e00" <= char <= "\u9fff" for char in text)

    async def _serper_search(self, query: str | list) -> str:
        """Use google.serper.dev web search API."""
        import requests

        queries = [query] if isinstance(query, str) else query

        async def search_single_query(q: str) -> str:
            cache_key = get_cache_key(q)
            cached_result = await get_cache_async(
                "text_search", cache_key, executor=self.executor
            )
            if cached_result:
                return cached_result

            headers = {
                "X-API-KEY": self.serper_api_key,
                "Content-Type": "application/json",
            }

            if self.contains_chinese(q):
                payload = {"q": q, "location": "China", "gl": "cn", "hl": "zh-cn"}
            else:
                payload = {"q": q, "location": "United States", "gl": "us", "hl": "en"}

            def send_request():
                return requests.request(
                    "POST",
                    self.serper_search_url,
                    headers=headers,
                    json=payload,
                    timeout=30,
                )

            try:
                resp = await run_with_retries_async(
                    send_request, executor=self.executor
                )
            except Exception as exc:  # noqa: BLE001
                error_message = f"Search request failed for '{q}': {exc}"
                log_search("Serper", "Exception", q, error=error_message)
                return error_message

            if resp.status_code != 200:
                error_message = f"HTTP {resp.status_code}: {resp.text}"
                log_search("Serper", "HTTPError", q, error=error_message)
                return f"Search returned HTTP {resp.status_code} for '{q}'\n{resp.text}"

            try:
                data_obj = resp.json()
            except Exception:
                data_obj = {}

            if isinstance(data_obj, dict) and "error" in data_obj:
                error_message = f"Serper error for '{q}': {data_obj['error']}"
                log_search("Serper", "APIError", q, error=error_message)
                return error_message

            items = []
            if isinstance(data_obj, dict):
                items = data_obj.get("organic") or []

            web_snippets: list[str] = []
            for idx, item in enumerate(items[: self.MAX_URLS], 1):
                title = (
                    item.get("title", "Untitled")
                    if isinstance(item, dict)
                    else "Untitled"
                )
                url = item.get("link", "") if isinstance(item, dict) else ""
                snippet = item.get("snippet", "") if isinstance(item, dict) else ""
                date = item.get("date") if isinstance(item, dict) else None
                source = item.get("source") if isinstance(item, dict) else None

                snippet = (snippet or "").strip()

                entry = f"{idx}. [{title}]({url})"
                if date:
                    entry += f"\n   Date published: {date}"
                if source:
                    entry += f"\n   Source: {source}"
                if snippet:
                    entry += f"\n   {snippet}"
                web_snippets.append(entry)

            content = (
                f"Search for '{q}' returned {len(web_snippets)} results:\n\n"
                + "\n\n".join(web_snippets)
                if web_snippets
                else f"No search results found for '{q}'"
            )
            if web_snippets:
                log_search("Serper", "Success", q, result=content)
                await set_cache_async(
                    "text_search", cache_key, q, content, executor=self.executor
                )

            return content

        tasks = [search_single_query(q) for q in queries]
        all_results: list[str] = await asyncio.gather(*tasks) if tasks else []

        final_result = (
            "\n=======\n".join(all_results)
            if len(all_results) > 1
            else (all_results[0] if all_results else "")
        )

        return final_result

    async def call(self, query: str | list, **kwargs) -> str:
        """
        Search the web using Serper API (google.serper.dev).

        Args:
            query: Search query string or list of queries

        Returns:
            Formatted search results
        """
        return await self._serper_search(query)
