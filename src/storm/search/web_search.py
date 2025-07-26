import asyncio
from abc import ABC, abstractmethod
from typing import List, Dict, Optional
from urllib.parse import urlparse
import httpx
from pydantic import BaseModel
from loguru import logger


class SearchResult(BaseModel):
    """Structured search result"""
    title: str
    url: str
    snippet: str
    content: Optional[str] = None
    domain: str = ""
    
    def __post_init__(self):
        if self.url:
            from urllib.parse import urlparse
            self.domain = urlparse(self.url).netloc


class SearchProvider(ABC):
    """Abstract base class for search providers"""
    
    @abstractmethod
    async def search(self, query: str, num_results: int = 10) -> List[SearchResult]:
        pass


class SerperSearchProvider(SearchProvider):
    """Serper.dev search provider"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.client = httpx.AsyncClient()
        
    async def search(self, query: str, num_results: int = 10) -> List[SearchResult]:
        """Search using Serper API"""
        url = "https://google.serper.dev/search"
        
        payload = {
            "q": query,
            "num": num_results
        }
        
        headers = {
            "X-API-KEY": self.api_key,
            "Content-Type": "application/json"
        }
        
        try:
            response = await self.client.post(url, json=payload, headers=headers)
            response.raise_for_status()
            
            data = response.json()
            results = []
            
            for item in data.get("organic", []):
                result = SearchResult(
                    title=item.get("title", ""),
                    url=item.get("link", ""),
                    snippet=item.get("snippet", "")
                )
                results.append(result)
                
            return results
            
        except Exception as e:
            logger.error(f"Search failed for query '{query}': {e}")
            return []


class WebSearchManager:
    """Manages web search with filtering and caching"""
    
    def __init__(self, provider: SearchProvider, exclude_domains: Optional[List[str]] = None):
        self.provider = provider
        self.exclude_domains = exclude_domains or []
        self._cache = {}
        
    async def search_and_filter(
        self, 
        queries: List[str], 
        max_results_per_query: int = 5
    ) -> List[SearchResult]:
        """Search multiple queries and filter results"""
        all_results = []
        
        for query in queries:
            if query in self._cache:
                results = self._cache[query]
            else:
                results = await self.provider.search(query, max_results_per_query)
                self._cache[query] = results
                
            # Filter results
            filtered_results = self._filter_results(results)
            all_results.extend(filtered_results)
            
            # Small delay to be respectful to APIs
            await asyncio.sleep(0.1)
            
        return self._deduplicate_results(all_results)
    
    def _filter_results(self, results: List[SearchResult]) -> List[SearchResult]:
        """Filter results based on domain exclusions and quality"""
        filtered = []
        
        for result in results:
            # Skip excluded domains
            if any(domain in result.domain for domain in self.exclude_domains):
                continue
                
            # Skip low-quality results
            if len(result.snippet) < 50:
                continue
                
            # Skip social media and forums (can be expanded)
            low_quality_domains = ["reddit.com", "twitter.com", "facebook.com", "quora.com"]
            if any(domain in result.domain for domain in low_quality_domains):
                continue
                
            filtered.append(result)
            
        return filtered
    
    def _deduplicate_results(self, results: List[SearchResult]) -> List[SearchResult]:
        """Remove duplicate results based on URL"""
        seen_urls = set()
        deduplicated = []
        
        for result in results:
            if result.url not in seen_urls:
                seen_urls.add(result.url)
                deduplicated.append(result)
                
        return deduplicated