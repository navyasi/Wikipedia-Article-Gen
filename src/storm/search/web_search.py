# src/storm/search/web_search.py - CORRECTED VERSION
import asyncio
from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Any
from urllib.parse import urlparse
import httpx
from pydantic import BaseModel
from loguru import logger
from datetime import datetime


class SearchResult(BaseModel):
    """Structured search result"""
    title: str
    url: str
    snippet: str
    content: Optional[str] = None
    domain: str = ""
    quality_score: float = 1.0  # ✅ ADD THIS LINE
    
    def model_post_init(self, __context):
        """Called after model initialization"""
        if self.url:
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


# Add this to your WebSearchManager or create a new SourceQualityAssessor class

class SourceQualityAssessor:
    def __init__(self, llm_client):
        self.llm = llm_client
        
        # Wikipedia source quality guidelines as system prompt
        self.wikipedia_source_guidelines = """
You are a Wikipedia editor evaluating source quality. Use these Wikipedia-approved source types:

ACCEPTABLE SOURCES:
• Scholarly articles: Peer-reviewed papers in academic journals with original research or literature reviews
• Books and monographs: Academic or well-researched popular works by established authors/publishers
• Textbooks: Educational materials covering specific subject areas
• Dictionaries and encyclopedias: Reference works (except Wikipedia itself)
• Archival and primary sources: Historic documents, official records, original documents
• Magazine articles: Articles from established magazines with editorial oversight
• Newspaper articles: News reports from established newspapers with editorial standards
• Reports and grey literature: Government documents, conference proceedings, official reports
• Statistics: Data from official sources, census data, verified statistical analysis
• Theses and dissertations: Academic works from accredited institutions
• Websites with editorial control: Professional websites with clear editorial oversight and fact-checking

UNACCEPTABLE SOURCES:
• Social media posts (Twitter, Facebook, Instagram, TikTok, etc.)
• Forums and discussion boards (Reddit, 4chan, etc.)
• Personal blogs without editorial oversight
• User-generated content without editorial control
• Commercial/promotional content
• Opinion pieces presented as fact
• Crowdsourced content without verification
• Anonymous or unattributed content

EVALUATION CRITERIA:
1. Editorial oversight and fact-checking procedures
2. Author credentials and expertise
3. Publication standards and peer review
4. Institutional backing or reputation
5. Transparency of sources and methodology
6. Absence of commercial bias or promotional content
"""

    async def assess_source_batch(self, sources: List[SearchResult]) -> List[tuple[SearchResult, bool, str]]:
        """Assess multiple sources in a single LLM call for efficiency"""
        
        # Prepare source information for LLM
        source_info = ""
        for i, source in enumerate(sources, 1):
            title = getattr(source, 'title', 'No title')
            url = getattr(source, 'url', 'No URL')
            snippet = getattr(source, 'snippet', '') or getattr(source, 'content', '')
            
            source_info += f"""
SOURCE {i}:
URL: {url}
Title: {title}
Content: {snippet[:300]}...

"""

        prompt = f"""{self.wikipedia_source_guidelines}

Evaluate these sources for Wikipedia-style article writing. For each source, determine:
1. Is it ACCEPTABLE or UNACCEPTABLE for Wikipedia?
2. Brief reason (1-2 words: e.g., "peer-reviewed", "social-media", "government-report", "personal-blog")

SOURCES TO EVALUATE:
{source_info}

Respond in this exact format:
SOURCE 1: ACCEPTABLE - government-report
SOURCE 2: UNACCEPTABLE - social-media  
SOURCE 3: ACCEPTABLE - news-article
SOURCE 4: UNACCEPTABLE - personal-blog
...

Your assessment:"""

        try:
            response = await self.llm.generate(prompt, temperature=0.1)  # Low temperature for consistency
            return self._parse_assessment_response(response, sources)
        except Exception as e:
            logger.error(f"LLM source assessment failed: {e}")
            # Fallback: accept all sources if LLM fails
            return [(source, True, "llm-error") for source in sources]

    def _parse_assessment_response(self, response: str, sources: List[SearchResult]) -> List[tuple[SearchResult, bool, str]]:
        """Parse LLM response into structured results"""
        
        results = []
        lines = response.split('\n')
        
        for i, source in enumerate(sources):
            try:
                # Look for "SOURCE {i+1}:" in response
                source_line = None
                for line in lines:
                    if f"SOURCE {i+1}:" in line.upper():
                        source_line = line
                        break
                
                if source_line:
                    if "ACCEPTABLE" in source_line.upper():
                        # Extract reason after the dash
                        reason = "accepted"
                        if " - " in source_line:
                            reason = source_line.split(" - ", 1)[1].strip()
                        results.append((source, True, reason))
                    else:
                        # Extract reason after the dash  
                        reason = "rejected"
                        if " - " in source_line:
                            reason = source_line.split(" - ", 1)[1].strip()
                        results.append((source, False, reason))
                else:
                    # Fallback if parsing fails
                    results.append((source, True, "parse-error"))
                    
            except Exception as e:
                logger.warning(f"Error parsing assessment for source {i+1}: {e}")
                results.append((source, True, "parse-error"))
        
        return results

    async def assess_single_source(self, source: SearchResult) -> tuple[bool, str]:
        """Assess a single source (for real-time filtering)"""
        
        title = getattr(source, 'title', 'No title')
        url = getattr(source, 'url', 'No URL')
        snippet = getattr(source, 'snippet', '') or getattr(source, 'content', '')
        
        prompt = f"""{self.wikipedia_source_guidelines}

Evaluate this single source for Wikipedia-style article writing:

URL: {url}
Title: {title}
Content: {snippet[:400]}...

Is this source ACCEPTABLE or UNACCEPTABLE for Wikipedia? 
Provide a brief reason (1-3 words).

Response format: ACCEPTABLE - reason OR UNACCEPTABLE - reason

Your assessment:"""

        try:
            response = await self.llm.generate(prompt, temperature=0.1)
            
            if "ACCEPTABLE" in response.upper():
                reason = "accepted"
                if " - " in response:
                    reason = response.split(" - ", 1)[1].strip()
                return True, reason
            else:
                reason = "rejected"
                if " - " in response:
                    reason = response.split(" - ", 1)[1].strip()
                return False, reason
                
        except Exception as e:
            logger.error(f"Single source assessment failed: {e}")
            return True, "llm-error"  # Default to accepting if LLM fails


# Enhanced WebSearchManager using LLM assessment
class WebSearchManager:
    def __init__(self, provider, exclude_domains=None, llm_client=None):
        self.provider = provider
        self.exclude_domains = exclude_domains or []
        self._cache = {}
        
        # Initialize source quality assessor if LLM provided
        self.source_assessor = SourceQualityAssessor(llm_client) if llm_client else None
        
        # Keep minimal hardcoded blocks for obvious cases (performance)
        self.obvious_blocks = {
            'reddit.com', 'youtube.com', 'facebook.com', 'twitter.com', 'x.com',
            'instagram.com', 'tiktok.com', 'snapchat.com', 'discord.com'
        }

    def _quick_block_check(self, url: str) -> bool:
        """Quick check for obviously unacceptable sources (performance optimization)"""
        try:
            if '://' in url:
                domain = url.split('://')[1].split('/')[0]
            else:
                domain = url.split('/')[0]
            if domain.startswith('www.'):
                domain = domain[4:]
            
            # Check obvious social media
            if any(blocked in domain for blocked in self.obvious_blocks):
                return True
                
            # Check social media URL patterns
            social_patterns = ['/r/', '/watch?v=', '/status/', '/posts/', '/p/']
            if any(pattern in url.lower() for pattern in social_patterns):
                return True
                
        except Exception:
            pass
        
        return False

    async def _filter_results_with_llm(self, results: List[SearchResult]) -> List[SearchResult]:
        """Filter results using LLM assessment"""
        
        if not self.source_assessor:
            # Fallback to basic filtering if no LLM
            return self._filter_results_basic(results)
        
        # Quick block obvious bad sources first (performance)
        pre_filtered = []
        blocked_obvious = 0
        
        for result in results:
            url = getattr(result, 'url', '')
            
            # Skip manual excludes
            if any(excluded in url for excluded in self.exclude_domains):
                continue
                
            # Quick block obvious social media
            if self._quick_block_check(url):
                blocked_obvious += 1
                continue
                
            pre_filtered.append(result)
        
        logger.info(f"Quick-blocked {blocked_obvious} obvious low-quality sources")
        
        if not pre_filtered:
            return []
        
        # Use LLM for remaining sources
        try:
            # Process in batches of 10 for efficiency
            all_filtered = []
            batch_size = 10
            
            for i in range(0, len(pre_filtered), batch_size):
                batch = pre_filtered[i:i + batch_size]
                assessments = await self.source_assessor.assess_source_batch(batch)
                
                # Extract accepted sources
                for source, is_acceptable, reason in assessments:
                    if is_acceptable:
                        all_filtered.append(source)
                        logger.debug(f"LLM ACCEPTED: {getattr(source, 'title', '')[:50]} - {reason}")
                    else:
                        logger.debug(f"LLM BLOCKED: {getattr(source, 'title', '')[:50]} - {reason}")
            
            logger.info(f"LLM assessment: {len(pre_filtered)} → {len(all_filtered)} sources")
            return all_filtered
            
        except Exception as e:
            logger.error(f"LLM filtering failed, using basic filter: {e}")
            return self._filter_results_basic(pre_filtered)

    def _filter_results_basic(self, results: List[SearchResult]) -> List[SearchResult]:
        """Basic fallback filtering without LLM"""
        filtered = []
        
        for result in results:
            url = getattr(result, 'url', '')
            
            if any(excluded in url for excluded in self.exclude_domains):
                continue
                
            if self._quick_block_check(url):
                continue
                
            filtered.append(result)
        
        return filtered

    async def search_and_filter(
        self, 
        queries: List[str], 
        max_results_per_query: int = 5
    ) -> List[SearchResult]:
        """Search with intelligent LLM-based source quality assessment"""
        all_results = []
        
        for query in queries:
            if query in self._cache:
                results = self._cache[query]
            else:
                results = await self.provider.search(query, max_results_per_query)
                self._cache[query] = results
                
            # Use LLM-based filtering
            filtered_results = await self._filter_results_with_llm(results)
            all_results.extend(filtered_results)
            await asyncio.sleep(0.1)
            
        return self._deduplicate_results(all_results)