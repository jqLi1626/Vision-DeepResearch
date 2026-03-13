import os

import requests

from rllm.tools.tool_base import Tool, ToolOutput

REFERENCE_COUNT = 8
DEFAULT_SEARCH_ENGINE_TIMEOUT = 30
SERPER_SEARCH_ENDPOINT = "https://google.serper.dev/search"
class GoogleSearchTool(Tool):
    """A tool for searching Google via the Serper API."""

    NAME = "google_search"
    DESCRIPTION = f"Search a query using the Google search engine (via Serper), returning the top {REFERENCE_COUNT} results along with a short snippet about their contents"

    def __init__(self, name: str = NAME, description: str = DESCRIPTION, timeout: float = DEFAULT_SEARCH_ENGINE_TIMEOUT, reference_count: int = REFERENCE_COUNT):
        """
        Initialize the GoogleSearch tool.

        Args:
            name (str): The name of the tool, defaults to GoogleSearch.NAME.
            description (str): A description of the tool's purpose, defaults to GoogleSearch.DESCRIPTION.
            timeout (float): Maximum time in seconds to wait for search results.
            reference_count (int): Number of results to return, defaults to REFERENCE_COUNT.
        """
        self.timeout = timeout
        self.reference_count = reference_count
        self.api_key = os.getenv("SERPER_API_KEY", "")
        super().__init__(name=name, description=description)

    @property
    def json(self):
        return {"type": "function", "function": {"name": self.name, "description": self.description, "parameters": {"type": "object", "properties": {"query": {"type": "string", "description": "Query to be submitted to Google search engine."}}, "required": ["query"]}}}

    def _search_with_serper(self, query: str):
        """
        Search with Serper API and return the result contexts.
        """
        headers = {
            "X-API-KEY": self.api_key,
            "Content-Type": "application/json",
        }
        payload = {"q": query}

        response = requests.request(
            "POST",
            SERPER_SEARCH_ENDPOINT,
            headers=headers,
            json=payload,
            timeout=self.timeout,
        )
        if not response.ok:
            print(f"{response.status_code} {response.text}")
        json_content = response.json()
        try:
            contexts = json_content.get("organic", [])[:self.reference_count]
        except Exception:
            print(f"Error encountered: {json_content}")
            return []
        return contexts

    def forward(self, query: str) -> ToolOutput:
        """
        Execute a Google search with the given query.

        Args:
            query (str): Query to be submitted to Google search engine.

        Returns:
            ToolOutput: An object containing either the search results or an error message.
        """
        try:
            contexts = self._search_with_serper(query)
            results = {c.get("link", ""): c.get("snippet", "") for c in contexts}
            return ToolOutput(name=self.name or "google_search", output=results)
        except Exception as e:
            return ToolOutput(name=self.name or "google_search", error=f"{type(e).__name__} - {str(e)}")


if __name__ == "__main__":
    search = GoogleSearchTool()
    print(search(query="Give me current time right now in PST"))
