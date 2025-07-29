# src/storm/core.py - CORRECTED VERSION
import asyncio
from typing import List, Dict, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
from pydantic import BaseModel
import wikipedia
from loguru import logger
from datetime import datetime, timedelta
import calendar

from .models.llama_client import LlamaClient, LlamaConfig
from .search.web_search import WebSearchManager, SearchResult
from .utils.text_processing import TextProcessor

@dataclass
class ConversationTurn:
    question: str
    answer: str
    sources: List[SearchResult] = field(default_factory=list)

@dataclass
class Perspective:
    name: str
    description: str

class StormConfig(BaseModel):
    max_related_topics: int = 5
    max_perspectives: int = 5
    max_conversation_rounds: int = 5
    exclude_wikipedia: bool = False
    target_article_exclusion: bool = True
    llama_config: LlamaConfig = LlamaConfig()
    max_search_results: int = 10
    exclude_domains: List[str] = field(default_factory=list)

class StormSystem:
    def __init__(self, config: StormConfig, search_manager: WebSearchManager, target_topic: Optional[str] = None):
        self.config = config
        self.search_manager = search_manager
        self.target_topic = target_topic
        self.text_processor = TextProcessor()
        
        # Dynamic date context
        self.current_year = datetime.now().year
        self.current_month = datetime.now().month
        self.current_date = datetime.now()
        
        # Recent years for relevance scoring (last 3 years)
        self.recent_years = [str(year) for year in range(self.current_year - 2, self.current_year + 1)]
        
        if target_topic and config.target_article_exclusion:
            wikipedia_url = f"en.wikipedia.org/wiki/{target_topic.replace(' ', '_')}"
            self.search_manager.exclude_domains.append(wikipedia_url)

    def _get_trusted_domains(self) -> Dict[str, float]:
        """Get trusted domains with quality scores"""
        return {
            # Government and Official Sources
            "usgs.gov": 10.0,
            "earthquake.usgs.gov": 10.0,
            "cwa.gov.tw": 9.5,
            "whitehouse.gov": 9.5,
            "state.gov": 9.5,
            "cdc.gov": 9.5,
            "who.int": 9.5,
            "un.org": 9.0,
            "europa.eu": 9.0,
            "gov.uk": 9.0,
            "canada.ca": 9.0,
            "gov.au": 9.0,
            
            # Major News Organizations
            "reuters.com": 9.0,
            "apnews.com": 9.0,
            "bbc.com": 8.8,
            "cnn.com": 8.5,
            "nytimes.com": 8.5,
            "washingtonpost.com": 8.5,
            "guardian.com": 8.5,
            "wsj.com": 8.5,
            "ft.com": 8.5,
            "economist.com": 8.5,
            "npr.org": 8.3,
            "pbs.org": 8.3,
            "axios.com": 8.0,
            "politico.com": 8.0,
            "bloomberg.com": 8.3,
            
            # Scientific and Academic
            "nature.com": 9.5,
            "science.org": 9.5,
            "sciencemag.org": 9.5,
            "pnas.org": 9.0,
            "ieee.org": 9.0,
            "cell.com": 9.0,
            "lancet.com": 9.0,
            "nejm.org": 9.0,
            "arxiv.org": 8.5,
            "pubmed.ncbi.nlm.nih.gov": 8.5,
            
            # Specialized Trusted Sources
            "weather.gov": 9.0,
            "noaa.gov": 9.0,
            "nasa.gov": 9.5,
            "esa.int": 9.0,
            "worldbank.org": 8.5,
            "imf.org": 8.5,
            "oecd.org": 8.5
        }

    def _detect_temporal_context(self, topic: str) -> Dict[str, Any]:
        """Intelligently detect temporal context from topic"""
        topic_lower = topic.lower()
        
        context = {
            "is_recent_event": False,
            "specific_year": None,
            "event_type": None,
            "urgency_level": "normal",
            "temporal_keywords": []
        }
        
        # Detect specific years (including future planning)
        for year in range(2020, self.current_year + 3):  # Include next 2 years for planning
            if str(year) in topic:
                context["specific_year"] = year
                if year >= self.current_year - 1:  # Current or last year
                    context["is_recent_event"] = True
                break
        
        # Detect temporal keywords
        temporal_indicators = {
            "immediate": ["breaking", "latest", "just happened", "ongoing", "live", "now"],
            "recent": ["recent", "new", "fresh", "updated", "current", "today", "yesterday"],
            "this_period": ["this year", "this month", "this week", f"{self.current_year}"],
            "future": ["upcoming", "planned", "scheduled", "next", "future", "projected"]
        }
        
        for category, keywords in temporal_indicators.items():
            for keyword in keywords:
                if keyword in topic_lower:
                    context["temporal_keywords"].append(keyword)
                    if category in ["immediate", "recent"]:
                        context["is_recent_event"] = True
                        context["urgency_level"] = category
        
        # Detect event types that are often time-sensitive
        event_types = {
            "natural_disaster": ["earthquake", "hurricane", "typhoon", "tsunami", "flood", "wildfire", "volcano"],
            "political": ["election", "vote", "summit", "treaty", "sanctions", "policy"],
            "conflict": ["war", "invasion", "conflict", "attack", "ceasefire", "peace"],
            "health": ["pandemic", "outbreak", "vaccine", "health crisis", "epidemic"],
            "technology": ["launch", "release", "breakthrough", "innovation", "ai", "tech"],
            "economic": ["market", "recession", "inflation", "crisis", "trade", "economy"],
            "sports": ["olympics", "world cup", "championship", "tournament", "game"],
            "space": ["mission", "launch", "landing", "discovery", "satellite", "space"]
        }
        
        for event_type, keywords in event_types.items():
            if any(keyword in topic_lower for keyword in keywords):
                context["event_type"] = event_type
                context["is_recent_event"] = True
                break
        
        # Auto-detect if no year specified but seems recent
        if not context["specific_year"] and context["is_recent_event"]:
            context["specific_year"] = self.current_year
        
        return context

    async def _generate_dynamic_search_queries(self, base_query: str, topic: str) -> List[str]:
        """Generate search queries with dynamic temporal awareness"""
        
        temporal_context = self._detect_temporal_context(topic)
        queries = [base_query]  # Always include original
        
        # Add year-specific queries
        if temporal_context["specific_year"]:
            year = temporal_context["specific_year"]
            queries.extend([
                f"{base_query} {year}",
                f"{base_query} {year} latest",
                f"{base_query} {year} news",
                f"{base_query} {year} update"
            ])
        else:
            # Add current year and recent years
            queries.extend([
                f"{base_query} {self.current_year}",
                f"{base_query} {self.current_year - 1}",
                f"{base_query} recent",
                f"{base_query} latest"
            ])
        
        # Add urgency-based queries
        if temporal_context["urgency_level"] == "immediate":
            queries.extend([
                f"{base_query} breaking news",
                f"{base_query} live updates",
                f"{base_query} real time"
            ])
        elif temporal_context["urgency_level"] == "recent":
            queries.extend([
                f"{base_query} recent developments",
                f"{base_query} current status",
                f"{base_query} latest news"
            ])
        
        # Add event-type specific queries
        if temporal_context["event_type"]:
            event_type = temporal_context["event_type"]
            
            if event_type == "natural_disaster":
                queries.extend([
                    f"{base_query} magnitude casualties damage",
                    f"{base_query} official report",
                    f"{base_query} USGS geological survey",
                    f"{base_query} emergency response"
                ])
            elif event_type == "political":
                queries.extend([
                    f"{base_query} results outcome",
                    f"{base_query} official statement",
                    f"{base_query} government response"
                ])
            elif event_type == "technology":
                queries.extend([
                    f"{base_query} announcement specifications",
                    f"{base_query} technical details",
                    f"{base_query} industry analysis"
                ])
        
        # Add trusted source queries
        trusted_sources = ["reuters", "BBC", "AP news", "official", "government"]
        for source in trusted_sources[:2]:  # Limit to avoid too many queries
            queries.append(f"{base_query} {source}")
        
        # Remove duplicates while preserving order
        seen = set()
        unique_queries = []
        for q in queries:
            if q.lower() not in seen:
                seen.add(q.lower())
                unique_queries.append(q)
        
        return unique_queries[:12]  # Limit to 12 queries for efficiency

    async def _decompose_question(self, llm: LlamaClient, topic: str, question: str) -> List[str]:
        """Create temporally-aware search queries"""
        
        temporal_context = self._detect_temporal_context(topic)
        current_year = self.current_year  # This is already an int
        previous_year = current_year - 1  # Calculate previous year
        
        # Enhanced prompt with dynamic temporal awareness
        prompt = f"""Create 3-5 highly specific search queries to find the most recent, factual information.

    Topic: {topic}
    Question: {question}
    Current Year: {current_year}

    REQUIREMENTS:
    - Prioritize the most recent information available
    - Include specific dates when relevant ({current_year}, {previous_year})
    - Include technical terms and measurements
    - Target authoritative sources (government, major news, scientific)
    - Avoid generic or vague terms

    Format as:
    - specific recent query with current timeframe
    - technical/scientific query
    - official source query"""

        response = await llm.generate(prompt, temperature=0.3)
        
        # Parse basic queries
        base_queries = []
        for line in response.split('\n'):
            line = line.strip()
            if line.startswith('- '):
                base_queries.append(line[2:].strip())
        
        # Generate dynamic enhanced queries
        all_queries = []
        for query in base_queries[:3]:
            enhanced = await self._generate_dynamic_search_queries(query, topic)
            all_queries.extend(enhanced)
        
        return all_queries[:15]  # Return comprehensive set

    # === REST OF THE EXISTING METHODS (keep these as they were) ===
    
    async def generate_article_outline(self, topic: str) -> Tuple[str, List[SearchResult]]:
        """Generate article outline and collect references"""
        logger.info(f"Starting STORM process for topic: {topic}")
        
        async with LlamaClient(self.config.llama_config) as llm:
            perspectives = await self._discover_perspectives(llm, topic)
            logger.info(f"Discovered {len(perspectives)} perspectives")
            
            all_conversations = []
            all_sources = []
            
            for perspective in perspectives:
                logger.info(f"Conducting conversation from perspective: {perspective.name}")
                conversation, sources = await self._conduct_conversation(llm, topic, perspective)
                all_conversations.append((perspective, conversation))
                all_sources.extend(sources)
            
            outline = await self._generate_outline(llm, topic, all_conversations)
            logger.info("STORM process completed successfully")
            return outline, all_sources

    async def generate_full_article(self, topic: str) -> Tuple[str, str, List[SearchResult]]:
        """Generate both outline and full article"""  
        logger.info(f"Starting STORM full article generation for: {topic}")
        
        # First get the outline and sources
        outline, sources = await self.generate_article_outline(topic)
        
        # Then generate the full article
        async with LlamaClient(self.config.llama_config) as llm:
            full_article = await self._write_full_article(llm, topic, outline, sources)
        
        logger.info("STORM full article generation completed")
        return outline, full_article, sources

    async def generate_simple_article(self, topic: str) -> Tuple[str, str, List[SearchResult]]:
        """Generate a simple article from outline and sources"""
        outline, sources = await self.generate_article_outline(topic)
        
        async with LlamaClient(self.config.llama_config) as llm:
            # Prepare sources
            sources_text = ""
            for i, source in enumerate(sources[:15], 1):
                sources_text += f"[{i}] {source.title}: {source.snippet}\n"
            
            prompt = f"""Write a comprehensive Wikipedia-style article about "{topic}".

OUTLINE TO FOLLOW:
{outline}

AVAILABLE SOURCES:
{sources_text}

Requirements:
- Follow the outline structure
- Write in encyclopedia style (neutral, factual)
- Include an introduction paragraph
- Write 2-3 paragraphs for each major section
- Use information from the sources
- Don't include citations in brackets

Write a complete, well-structured article:"""

            article = await llm.generate(prompt, temperature=0.6)
            return outline, article, sources

    async def _write_full_article(self, llm: LlamaClient, topic: str, outline: str, sources: List[SearchResult]) -> str:
        """Generate full article from outline and sources"""
        
        # Split outline into sections
        sections = self._parse_outline_sections(outline)
        
        article_parts = []
        article_parts.append(f"# {topic}\n\n")
        
        # Generate introduction
        logger.info("Generating introduction...")
        intro = await self._generate_introduction(llm, topic, sources[:10])
        article_parts.append(f"{intro}\n\n")
        
        # Generate each section
        for section in sections:
            if section['level'] <= 2:  # Only generate content for main sections
                logger.info(f"Generating section: {section['title']}")
                section_content = await self._generate_section_content(
                    llm, topic, section, sources
                )
                
                # Add section header
                header_level = "#" * (section['level'] + 1)
                article_parts.append(f"{header_level} {section['title']}\n\n")
                
                # Add section content
                article_parts.append(f"{section_content}\n\n")
        
        return "".join(article_parts)

    def _parse_outline_sections(self, outline: str) -> List[Dict]:
        """Parse outline into structured sections"""
        sections = []
        lines = outline.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Count # symbols to determine level
            level = 0
            while level < len(line) and line[level] == '#':
                level += 1
            
            if level > 0 and level < len(line):
                title = line[level:].strip()
                if title:
                    sections.append({
                        'title': title,
                        'level': level,
                        'line': line
                    })
        
        return sections

    async def _generate_introduction(self, llm: LlamaClient, topic: str, sources: List[SearchResult]) -> str:
        """Generate article introduction"""
        
        # Prepare source context
        source_context = ""
        for i, source in enumerate(sources[:5], 1):
            source_context += f"Source {i}: {source.title} - {source.snippet}\n"
        
        prompt = f"""Write a comprehensive introduction for a Wikipedia article about "{topic}".

Available information:
{source_context}

Requirements:
- Write 2-3 paragraphs
- Provide clear definition and overview
- Mention key aspects that will be covered  
- Write in encyclopedia style
- Be factual and neutral
- Don't use first person
- Don't include citations

Introduction:"""

        return await llm.generate(prompt, temperature=0.6)

    async def _generate_section_content(
        self, 
        llm: LlamaClient, 
        topic: str, 
        section: Dict, 
        sources: List[SearchResult]
    ) -> str:
        """Generate content for a specific section"""
        
        # Find relevant sources for this section
        relevant_sources = self._find_relevant_sources_for_section(
            section['title'], sources
        )
        
        # Prepare source context
        source_context = ""
        for i, source in enumerate(relevant_sources[:8], 1):
            source_context += f"Source {i}: {source.title} - {source.snippet}\n"
        
        if not source_context:
            source_context = "Limited source information available."
        
        prompt = f"""Write detailed content for the "{section['title']}" section of a Wikipedia article about "{topic}".

Available information:
{source_context}

Requirements:
- Write 2-3 paragraphs
- Use encyclopedia style (neutral, factual)
- Include specific details from sources
- Don't repeat information from other sections
- Write complete sentences and paragraphs
- Don't include citations in brackets

Section content:"""

        return await llm.generate(prompt, temperature=0.6)

    def _find_relevant_sources_for_section(self, section_title: str, sources: List[SearchResult]) -> List[SearchResult]:
        """Find sources most relevant to a specific section"""
        
        # Simple keyword matching approach
        section_keywords = section_title.lower().split()
        
        scored_sources = []
        for source in sources:
            score = 0
            source_text = (source.title + " " + source.snippet).lower()
            
            # Count keyword matches
            for keyword in section_keywords:
                if keyword in source_text:
                    score += source_text.count(keyword)
            
            if score > 0:
                scored_sources.append((score, source))
        
        # Sort by relevance and return top sources
        scored_sources.sort(key=lambda x: x[0], reverse=True)
        return [source for score, source in scored_sources[:10]]
    
    async def _discover_perspectives(self, llm: LlamaClient, topic: str) -> List[Perspective]:
        related_topics = await self._generate_related_topics(llm, topic)
        related_tocs = []
        for related_topic in related_topics:
            toc = await self._get_wikipedia_toc(related_topic)
            if toc:
                related_tocs.append(toc)
        
        perspectives = await self._generate_perspectives(llm, topic, related_tocs)
        basic_perspective = Perspective(name="basic fact writer", description="focusing on broadly covering the basic facts about the topic")
        return [basic_perspective] + perspectives[:self.config.max_perspectives]
    
    async def _generate_related_topics(self, llm: LlamaClient, topic: str) -> List[str]:
        prompt = f"List {self.config.max_related_topics} topics closely related to: {topic}\nOne per line, no numbering:"
        response = await llm.generate(prompt, temperature=0.7)
        topics = [line.strip() for line in response.split('\n') if line.strip()]
        return topics[:self.config.max_related_topics]
    
    async def _get_wikipedia_toc(self, topic: str) -> Optional[str]:
        try:
            loop = asyncio.get_event_loop()
            page = await loop.run_in_executor(None, wikipedia.page, topic)
            sections = page.sections
            toc = f"Article: {topic}\nSections:\n"
            for i, section in enumerate(sections, 1):
                toc += f"{i}. {section}\n"
            return toc
        except Exception as e:
            logger.warning(f"Could not retrieve TOC for {topic}: {e}")
            return None
    
    async def _generate_perspectives(self, llm: LlamaClient, topic: str, related_tocs: List[str]) -> List[Perspective]:
        toc_context = "\n\n".join(related_tocs) if related_tocs else "No related topics available."
        prompt = f"""Create {self.config.max_perspectives} different perspectives for researching: {topic}

Format:
1. Perspective name: Description
2. Perspective name: Description
..."""
        response = await llm.generate(prompt, temperature=0.8)
        
        perspectives = []
        for line in response.split('\n'):
            line = line.strip()
            if line and any(line.startswith(f"{i}.") for i in range(1, 10)):
                try:
                    content = line.split('.', 1)[1].strip()
                    if ':' in content:
                        name, description = content.split(':', 1)
                        perspectives.append(Perspective(name=name.strip(), description=description.strip()))
                except Exception as e:
                    logger.warning(f"Could not parse perspective: {line}")
        return perspectives
    
    async def _conduct_conversation(self, llm: LlamaClient, topic: str, perspective: Perspective) -> Tuple[List[ConversationTurn], List[SearchResult]]:
        conversation = []
        all_sources = []
        
        for round_num in range(self.config.max_conversation_rounds):
            question = await self._generate_question(llm, topic, perspective, conversation)
            
            if "thank you" in question.lower() or len(question) < 10:
                break
            
            queries = await self._decompose_question(llm, topic, question)
            sources = await self.search_manager.search_and_filter(queries)
            all_sources.extend(sources)
            
            answer = await self._generate_answer(llm, topic, question, sources)
            turn = ConversationTurn(question=question, answer=answer, sources=sources)
            conversation.append(turn)
        
        return conversation, all_sources
    
    async def _generate_question(self, llm: LlamaClient, topic: str, perspective: Perspective, conversation_history: List[ConversationTurn]) -> str:
        history_text = ""
        for turn in conversation_history:
            history_text += f"Q: {turn.question}\nA: {turn.answer}\n\n"
        
        prompt = f"""You are researching: {topic}
Your perspective: {perspective.description}

Conversation history:
{history_text}

Ask ONE specific question to learn more. If no more questions, say "Thank you so much for your help!"

Question:"""
        return await llm.generate(prompt, temperature=0.9)
    
    async def _generate_answer(self, llm: LlamaClient, topic: str, question: str, sources: List[SearchResult]) -> str:
        source_info = ""
        for i, source in enumerate(sources[:5], 1):
            source_info += f"Source {i}: {source.title}\n{source.snippet}\n\n"
        
        if not source_info:
            source_info = "No relevant sources found."
        
        prompt = f"Answer this question about {topic} using the provided sources:\n\nQuestion: {question}\n\nSources:\n{source_info}\n\nAnswer:"
        return await llm.generate(prompt, temperature=0.7)
    
    async def _generate_outline(self, llm: LlamaClient, topic: str, conversations: List[Tuple[Perspective, List[ConversationTurn]]]) -> str:
        basic_outline = await self._generate_basic_outline(llm, topic)
        
        insights = ""
        for perspective, conversation in conversations:
            insights += f"\nPerspective: {perspective.name}\n"
            for turn in conversation:
                insights += f"Q: {turn.question}\nA: {turn.answer[:200]}...\n"
        
        refine_prompt = f"Improve this outline using research insights:\n\nTopic: {topic}\n\nCurrent outline:\n{basic_outline}\n\nResearch insights:\n{insights}\n\nCreate comprehensive outline using # and ## headers:"
        return await llm.generate(refine_prompt, temperature=0.7)
    
    async def _generate_basic_outline(self, llm: LlamaClient, topic: str) -> str:
        prompt = f"Create a comprehensive Wikipedia article outline for: {topic}\n\nUse format:\n# Main Section\n## Subsection\n\nInclude all major aspects:"
        return await llm.generate(prompt, temperature=0.7)
    

# === ADD THIS ENTIRE CLASS TO THE END OF core.py ===

class UniversalStormSystem:
    """Universal STORM system that works for any topic dynamically"""
    
    def __init__(self, config, search_manager, target_topic=None):
        self.config = config
        self.search_manager = search_manager
        self.target_topic = target_topic
        
    async def generate_universal_article(self, topic: str) -> Tuple[str, str, List[SearchResult]]:
        """Generate article for any topic using dynamic approach"""
        logger.info(f"Starting universal research for: {topic}")
        
        async with LlamaClient(self.config.llama_config) as llm:
            
            # STEP 1: Extract key concepts and generate dynamic queries
            search_queries = await self._generate_dynamic_queries(llm, topic)
            logger.info(f"Generated {len(search_queries)} dynamic queries")
            
            # STEP 2: Search and filter for relevance
            all_sources = await self._search_and_filter_universal(search_queries, topic)
            logger.info(f"Found {len(all_sources)} relevant sources")
            
            # DEBUG: Log sources found
            logger.info(f"DEBUG: Sources collected: {[(getattr(s, 'title', 'No title')[:50], getattr(s, 'url', 'No URL')) for s in all_sources[:3]]}")
            
            # STEP 3: Generate outline based on actual content found
            try:
                logger.info("Generating outline...")
                outline = await self._generate_content_based_outline(llm, topic, all_sources)
                logger.info(f"Generated outline: {len(outline)} characters")
                
                # Debug: Show first few lines of outline
                outline_preview = '\n'.join(outline.split('\n')[:5])
                logger.info(f"Outline preview: {outline_preview}")
                
            except Exception as e:
                logger.error(f"Error generating outline: {e}")
                import traceback
                logger.error(f"Outline generation traceback: {traceback.format_exc()}")
                outline = self._generate_basic_outline_fallback(topic)
            
            # STEP 4: Generate confident, fact-based article
            try:
                logger.info("Generating article...")
                article = await self._generate_confident_article(llm, topic, outline, all_sources)
                logger.info(f"Generated article: {len(article)} characters")
                
                # Check if article is too short or generic
                if len(article) < 500:
                    logger.warning(f"Article seems too short ({len(article)} chars), regenerating...")
                    article = await self._generate_fallback_article(llm, topic, all_sources)
                    
            except Exception as e:
                logger.error(f"Error generating article: {e}")
                import traceback
                logger.error(f"Traceback: {traceback.format_exc()}")
                # Generate a fallback article
                article = await self._generate_fallback_article(llm, topic, all_sources)
            
            # Ensure we're returning the sources
            logger.info(f"Returning {len(all_sources)} sources to main")
            return outline, article, all_sources
    
    async def _generate_dynamic_queries(self, llm: LlamaClient, topic: str) -> List[str]:
        """Dynamically generate search queries based on topic analysis - ENHANCED VERSION"""
        
        current_year = datetime.now().year
        
        # Start with basic queries
        queries = [
            topic,
            f"{topic} {current_year}",
            f"{topic} tutorial",
            f"{topic} guide"
        ]
        
        # Use LLM to understand the topic and generate more queries
        prompt = f"""Generate 6-8 specific search queries to find comprehensive information about: "{topic}"

Consider different aspects:
- Basic explanations and definitions
- Technical details and implementation
- Examples and applications  
- Recent developments
- Educational resources
- Practical guides

Generate varied search queries that would find different types of information.

Format as a simple list:
- query 1
- query 2
- query 3
..."""

        try:
            response = await llm.generate(prompt, temperature=0.7)
            
            # Parse queries from response
            llm_queries = []
            for line in response.split('\n'):
                line = line.strip()
                if line.startswith('- '):
                    query = line[2:].strip()
                    if query and len(query) > 3:
                        llm_queries.append(query)
            
            # Combine base queries with LLM queries
            all_queries = queries + llm_queries[:6]
            
            # Remove duplicates while preserving order
            seen = set()
            unique_queries = []
            for q in all_queries:
                if q.lower() not in seen:
                    seen.add(q.lower())
                    unique_queries.append(q)
            
            logger.info(f"Generated {len(unique_queries)} queries: {unique_queries}")
            return unique_queries[:10]  # Limit to 10 queries
            
        except Exception as e:
            logger.warning(f"LLM query generation failed: {e}")
            # Fallback to basic queries
            return queries
    
    async def _search_and_filter_universal(self, queries: List[str], topic: str) -> List[SearchResult]:
        """Search and filter results for any topic - SIMPLIFIED VERSION"""
        
        logger.info(f"Searching with {len(queries)} queries: {queries}")
        
        # TEMPORARY FIX: Use basic search without complex LLM filtering
        all_sources = []
        
        for query in queries:
            try:
                logger.info(f"Searching for: {query}")
                
                # Use provider directly to avoid the missing method issue
                if hasattr(self.search_manager, 'provider'):
                    sources = await self.search_manager.provider.search(query, num_results=8)
                    logger.info(f"Found {len(sources)} results for: {query}")
                    
                    # Simple filtering - just remove obvious social media
                    filtered_sources = []
                    for source in sources:
                        url = getattr(source, 'url', '').lower()
                        title = getattr(source, 'title', '').lower()
                        
                        # Only block obvious social media, accept everything else
                        if not any(blocked in url for blocked in [
                            'reddit.com', 'youtube.com', 'facebook.com', 
                            'twitter.com', 'x.com', 'instagram.com', 'tiktok.com'
                        ]):
                            filtered_sources.append(source)
                            logger.debug(f"Accepted: {getattr(source, 'title', '')[:50]}")
                    
                    all_sources.extend(filtered_sources)
                    await asyncio.sleep(0.3)
                    
                else:
                    logger.error("No search provider available")
                    
            except Exception as e:
                logger.warning(f"Search failed for query '{query}': {e}")
                continue
        
        # Simple deduplication
        unique_sources = []
        seen_urls = set()
        
        for source in all_sources:
            url = getattr(source, 'url', '')
            if url and url not in seen_urls:
                seen_urls.add(url)
                unique_sources.append(source)
        
        logger.info(f"Total unique sources found: {len(unique_sources)}")
        
        # Return first 25 sources
        final_sources = unique_sources[:25]
        
        # Debug: log first few sources
        for i, source in enumerate(final_sources[:3]):
            try:
                title = getattr(source, 'title', 'No title')
                logger.info(f"Source {i+1}: {title[:50]}...")
            except Exception as e:
                logger.warning(f"Error logging source {i+1}: {e}")
        
        return final_sources
    
    def _extract_key_terms(self, topic: str) -> Set[str]:
        """Extract key terms from topic for relevance checking"""
        
        # Remove common stop words
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
            'how', 'what', 'when', 'where', 'why', 'who', 'which', 'that', 'this', 'these', 'those',
            'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did'
        }
        
        # Extract meaningful terms (3+ characters, not stop words)
        terms = set()
        for word in topic.lower().split():
            cleaned_word = ''.join(c for c in word if c.isalnum())
            if len(cleaned_word) >= 3 and cleaned_word not in stop_words:
                terms.add(cleaned_word)
        
        return terms
    
    def _filter_universal_relevance(self, sources: List[SearchResult], topic_terms: Set[str], original_topic: str) -> List[SearchResult]:
        """Universal relevance filter that works for any topic - more lenient since search_manager already filtered"""
        
        relevant = []
        
        for source in sources:
            try:
                # Get source text safely
                title = getattr(source, 'title', '')
                snippet = getattr(source, 'snippet', '') or getattr(source, 'content', '')
                url = getattr(source, 'url', '')
                source_text = (title + " " + snippet).lower()
                
                # Additional quality check - block remaining low-quality sources
                if self._is_low_quality_source(url, title, snippet):
                    logger.debug(f"Blocked low-quality source: {title[:50]}")
                    continue
                
                # Count term matches
                matches = sum(1 for term in topic_terms if term in source_text)
                
                # Calculate relevance score
                relevance_ratio = matches / len(topic_terms) if topic_terms else 0
                
                # Very lenient thresholds since search_manager already filtered
                # Accept almost everything that has at least one term match
                min_threshold = 0.1  # Only 10% of terms need to match
                
                # Also check if the exact topic appears in the source
                exact_topic_match = original_topic.lower() in source_text
                
                # Accept if any relevance is found
                if relevance_ratio >= min_threshold or exact_topic_match or matches >= 1:
                    # Don't try to assign new fields to the SearchResult object
                    relevant.append(source)
                    logger.debug(f"Accepted source: {title[:50]} (ratio: {relevance_ratio:.2f}, matches: {matches})")
                else:
                    logger.debug(f"Rejected source: {title[:50]} (ratio: {relevance_ratio:.2f}, matches: {matches})")
                    
            except Exception as e:
                logger.warning(f"Error processing source: {e}")
                # Include the source anyway if we can't process it
                relevant.append(source)
        
        logger.info(f"Universal relevance filter: {len(sources)} -> {len(relevant)} sources")
        return relevant

    def _is_low_quality_source(self, url: str, title: str, snippet: str) -> bool:
        """Additional low-quality source detection"""
        
        # Extract domain
        try:
            if '://' in url:
                domain = url.split('://')[1].split('/')[0]
            else:
                domain = url.split('/')[0]
            if domain.startswith('www.'):
                domain = domain[4:]
        except Exception:
            return False
        
        # Block social media and forums that might have slipped through
        social_domains = [
            'reddit.com', 'youtube.com', 'facebook.com', 'twitter.com', 'x.com',
            'instagram.com', 'tiktok.com', 'quora.com', 'yahoo.com',
            'medium.com', 'substack.com', 'blogspot.com', 'wordpress.com'
        ]
        
        if any(social in domain for social in social_domains):
            return True
        
        # Check for forum/discussion patterns in URL
        forum_patterns = ['/forum/', '/discussion/', '/thread/', '/post/', '/comment/']
        if any(pattern in url.lower() for pattern in forum_patterns):
            return True
        
        # Check title for low-quality indicators
        low_quality_indicators = [
            'reddit', 'youtube', 'comment', 'discussion', 'forum',
            'my opinion', 'what i think', 'personal blog', 'user review'
        ]
        
        title_lower = title.lower()
        if any(indicator in title_lower for indicator in low_quality_indicators):
            return True
        
        return False
    
    def _deduplicate_and_rank_universal(self, sources: List[SearchResult], topic_terms: Set[str]) -> List[SearchResult]:
        """Universal ranking system with safe attribute access - no field assignment"""
        
        # Remove duplicates and create scoring tuples
        seen_urls = set()
        scored_sources = []
        
        for source in sources:
            try:
                url = getattr(source, 'url', f'no_url_{len(scored_sources)}')
                if url not in seen_urls:
                    seen_urls.add(url)
                    
                    # Calculate score without modifying the source object
                    score = 1  # Base score
                    
                    # Get domain safely
                    domain = getattr(source, 'domain', '')
                    if not domain:
                        if url and '://' in url:
                            domain = url.split('://')[1].split('/')[0]
                    
                    # Domain quality bonus (universal trusted sources)
                    trusted_domains = [
                        'reuters.com', 'apnews.com', 'bbc.com', 'cnn.com', 'nytimes.com',
                        'washingtonpost.com', 'guardian.com', 'nature.com', 'science.org',
                        'gov', 'edu', 'who.int', 'un.org', 'nasa.gov', 'noaa.gov'
                    ]
                    
                    if any(trusted_domain in domain for trusted_domain in trusted_domains):
                        score += 3
                    
                    # Recency bonus
                    current_year = datetime.now().year
                    title = getattr(source, 'title', '')
                    snippet = getattr(source, 'snippet', '') or getattr(source, 'content', '')
                    text = title + ' ' + snippet
                    
                    if any(str(year) in text for year in [current_year, current_year-1]):
                        score += 2
                    
                    # Content length bonus (more substantial sources)
                    if len(snippet) > 200:
                        score += 1
                    
                    # Store as (score, source) tuple
                    scored_sources.append((score, source))
                    
            except Exception as e:
                logger.warning(f"Error processing source for ranking: {e}")
                scored_sources.append((1, source))  # Default score
        
        # Sort by score (descending) and return just the sources
        try:
            scored_sources.sort(key=lambda x: x[0], reverse=True)
            ranked_sources = [source for score, source in scored_sources]
            logger.info(f"Ranked {len(ranked_sources)} sources successfully")
            return ranked_sources
        except Exception as e:
            logger.warning(f"Error sorting sources: {e}")
            return [source for score, source in scored_sources]
    
    async def _generate_content_based_outline(self, llm: LlamaClient, topic: str, sources: List[SearchResult]) -> str:
        """Generate outline based on actual content found"""
        
        # Analyze source content to determine what aspects are covered
        content_analysis = ""
        source_count = 0
        
        for i, source in enumerate(sources[:12], 1):
            try:
                title = getattr(source, 'title', f'Source {i}')
                snippet = getattr(source, 'snippet', '') or getattr(source, 'content', '')
                
                if title and snippet:
                    content_analysis += f"{i}. {title}\n   Key info: {snippet[:100]}...\n\n"
                    source_count += 1
            except Exception as e:
                logger.warning(f"Error processing source {i} for outline: {e}")
        
        if source_count == 0:
            logger.warning("No usable source content for outline, generating basic outline")
            return self._generate_basic_outline_fallback(topic)
        
        logger.info(f"Using {source_count} sources for outline generation")
        
        prompt = f"""Based on the available sources about "{topic}", create a comprehensive Wikipedia-style outline.

Available content from {source_count} sources:
{content_analysis}

Create an outline that:
- Covers the main aspects found in the sources
- Is organized logically for a Wikipedia article
- Includes specific sections based on the actual information available
- Follows Wikipedia structure (Introduction, main topics, details, conclusion)

Use this exact format:
# Main Section
## Subsection
### Sub-subsection

Example for reference:
# Introduction
## Background
## Key Events
### Primary Elections
### General Election Campaign
### Election Day
## Results
### Vote Counts
### Electoral College
## Aftermath
## References

Generate a comprehensive outline for {topic}:"""

        try:
            outline = await llm.generate(prompt, temperature=0.6)
            
            # Validate outline
            if not outline or len(outline.strip()) < 50:
                logger.warning("Generated outline too short, using fallback")
                return self._generate_basic_outline_fallback(topic)
            
            # Check if outline has proper formatting
            if '#' not in outline:
                logger.warning("Generated outline lacks proper formatting, using fallback")
                return self._generate_basic_outline_fallback(topic)
            
            logger.info(f"Successfully generated outline: {len(outline)} characters")
            return outline
            
        except Exception as e:
            logger.error(f"Error generating content-based outline: {e}")
            return self._generate_basic_outline_fallback(topic)

    def _generate_basic_outline_fallback(self, topic: str) -> str:
        """Generate a basic outline when source-based generation fails"""
        
        logger.info(f"Generating basic fallback outline for: {topic}")
        
        # Check if topic is about elections
        topic_lower = topic.lower()
        
        if any(election_word in topic_lower for election_word in ['election', 'presidential', 'campaign', 'vote']):
            return f"""# {topic}

## Background
### Historical Context
### Key Issues

## Candidates
### Democratic Candidate
### Republican Candidate
### Third Party Candidates

## Campaign
### Primary Elections
### General Election Campaign
### Debates
### Polling

## Election Day
### Voting Process
### Turnout
### Key States

## Results
### Popular Vote
### Electoral College
### State-by-State Results

## Analysis
### Vote Demographics
### Key Factors
### Historical Comparison

## Aftermath
### Transition Period
### Reactions
### Impact

## References"""
        
        # Generic outline for other topics
        return f"""# {topic}

## Introduction
### Definition
### Overview

## Background
### History
### Development

## Key Concepts
### Main Features
### Important Aspects

## Applications
### Uses
### Examples

## Current Status
### Recent Developments
### Modern Context

## Impact
### Significance
### Effects

## Future Outlook
### Trends
### Predictions

## References"""
    
    async def _generate_fallback_article(self, llm: LlamaClient, topic: str, sources: List[SearchResult]) -> str:
        """Generate a simple fallback article WITH CITATIONS when main generation fails"""
        
        # Prepare numbered source summaries
        source_summaries = ""
        source_map = {}
        
        for i, source in enumerate(sources[:15], 1):
            try:
                title = getattr(source, 'title', f'Source {i}')
                snippet = getattr(source, 'snippet', '') or getattr(source, 'content', '')
                url = getattr(source, 'url', '')
                
                if snippet:
                    source_summaries += f"[{i}] {title}: {snippet[:150]}...\n"
                    source_map[i] = {'title': title, 'url': url}
            except Exception as e:
                logger.warning(f"Error processing source {i} for fallback: {e}")
        
        if not source_summaries:
            # Absolute fallback with no sources
            return self._generate_emergency_article(topic)
        
        prompt = f"""Write a comprehensive article about "{topic}" using the numbered sources provided.

NUMBERED SOURCES:
{source_summaries}

CRITICAL REQUIREMENTS:
- ALWAYS include citation numbers [1], [2], [3] when making claims
- Each paragraph should have multiple citations
- Use specific facts, dates, and details from the sources
- Write in encyclopedia style (neutral, factual, third-person)
- Include 4-6 paragraphs with citations throughout
- End with a References section

CITATION EXAMPLES:
- "According to official reports, the event occurred on [date] [1]."
- "Multiple sources confirm the impact was significant [2][3]."

Write the complete cited article:"""

        try:
            article = await llm.generate(prompt, temperature=0.6)
            
            # Add references section if not present
            if "References" not in article:
                article += "\n\n## References\n\n"
                for i, source_info in source_map.items():
                    article += f"{i}. [{source_info['title']}]({source_info['url']})\n"
            
            logger.info(f"Fallback article with citations generated: {len(article)} characters")
            return article
            
        except Exception as e:
            logger.error(f"Even fallback generation failed: {e}")
            return self._generate_emergency_article(topic)

    def _generate_emergency_article(self, topic: str) -> str:
        """Emergency article when everything else fails"""
        return f"""# {topic}

{topic} is a significant topic that has gained attention in recent times. This article provides an overview based on available information.

## Overview

The topic of {topic} encompasses various important aspects that contribute to its overall significance. Current research and analysis provide insights into the key characteristics and implications of this subject.

## Background

{topic} has developed over time, with various factors contributing to its current state. Understanding the background helps provide context for the topic's significance and relevance.

## Current Status

Recent developments related to {topic} continue to evolve. The current situation reflects ongoing changes and developments in this area.

## Impact and Significance

The importance of {topic} can be understood through its various impacts and the attention it has received from multiple sources and perspectives.

## References

*Note: This article was generated as an emergency fallback. Please refer to the sources list at the end of this document for specific references and citations.*
"""

    async def _generate_confident_article(self, llm: LlamaClient, topic: str, outline: str, sources: List[SearchResult]) -> str:
        """Generate confident article WITH CITATIONS"""
        
        logger.info(f"Generating article with {len(sources)} sources")
        
        # Prepare rich source content with numbered references
        source_content = ""
        source_reference_map = {}
        usable_sources = 0
        
        for i, source in enumerate(sources[:20], 1):
            try:
                title = getattr(source, 'title', f'Source {i}')
                snippet = getattr(source, 'snippet', '') or getattr(source, 'content', '')
                url = getattr(source, 'url', '')
                
                if snippet and len(snippet.strip()) > 20:  # Require substantial content
                    source_content += f"[{i}] {title}:\n{snippet}\nURL: {url}\n\n"
                    source_reference_map[i] = {'title': title, 'url': url, 'snippet': snippet}
                    usable_sources += 1
                    
            except Exception as e:
                logger.warning(f"Error processing source {i}: {e}")
        
        logger.info(f"Prepared {usable_sources} usable sources for article generation")
        
        if usable_sources == 0:
            logger.warning("No usable source content available, using outline-based fallback")
            return await self._generate_outline_based_article(llm, topic, outline)
        
        # Enhanced prompt that REQUIRES citations
        prompt = f"""Write a comprehensive, authoritative Wikipedia-style article about "{topic}" using the provided sources and following the outline.

OUTLINE TO FOLLOW:
{outline}

NUMBERED SOURCES ({usable_sources} available):
{source_content}

CRITICAL WRITING INSTRUCTIONS:
- Write with confidence and authority using the numbered sources provided
- ALWAYS include citation numbers [1], [2], [3] etc. when making factual claims
- Each major fact or claim MUST be supported by a source citation
- Use multiple sources to support important claims: [1][2] or [1,3,5]
- Present information as factual statements with proper citations
- Include specific details like dates, numbers, locations, names from the sources
- Write in encyclopedia style: neutral, factual, third-person
- Each section should be 2-4 paragraphs with citations throughout
- Follow the outline structure closely

CITATION EXAMPLES:
- "The election took place on November 5, 2024 [1]."
- "Voter turnout was approximately 66% according to multiple reports [2][4]."
- "The results were certified by all 50 states [3]."

Write the complete article with proper citations following the outline:"""

        try:
            article = await llm.generate(prompt, temperature=0.5)
            
            # Validate the article has citations
            if not self._validate_citations(article):
                logger.warning("Generated article lacks citations, adding them...")
                article = await self._add_citations_to_article(llm, article, source_reference_map)
                
            # Ensure References section exists
            if "## References" not in article and "# References" not in article:
                article += self._generate_references_section(source_reference_map)
                
            logger.info(f"Successfully generated article with sources: {len(article)} characters")
            return article
            
        except Exception as e:
            logger.error(f"Error in confident article generation: {e}")
            return await self._generate_outline_based_article(llm, topic, outline)

    async def _generate_outline_based_article(self, llm: LlamaClient, topic: str, outline: str) -> str:
        """Generate article based on outline when sources are insufficient"""
        
        logger.info("Generating outline-based article without specific sources")
        
        prompt = f"""Write a comprehensive Wikipedia-style article about "{topic}" following the provided outline structure.

OUTLINE TO FOLLOW:
{outline}

INSTRUCTIONS:
- Follow the outline structure exactly
- Write 2-3 paragraphs for each major section
- Use your knowledge about the topic
- Write in encyclopedia style (neutral, factual, third-person)
- Include specific details like dates, locations, and key facts
- Make each section substantial and informative
- Write with authority and confidence

Write the complete article following the outline:"""

        try:
            article = await llm.generate(prompt, temperature=0.6)
            
            if not article or len(article.strip()) < 500:
                logger.warning("Outline-based article too short, using emergency fallback")
                return self._generate_emergency_article(topic)
            
            logger.info(f"Generated outline-based article: {len(article)} characters")
            return article
            
        except Exception as e:
            logger.error(f"Error generating outline-based article: {e}")
            return self._generate_emergency_article(topic)

    def _validate_citations(self, article: str) -> bool:
        """Check if article contains proper citations"""
        import re
        
        # Look for citation patterns like [1], [2], [1,2], [1][2], etc.
        citation_patterns = [
            r'\[\d+\]',           # [1]
            r'\[\d+,\d+\]',       # [1,2]  
            r'\[\d+\]\[\d+\]',    # [1][2]
        ]
        
        citation_count = 0
        for pattern in citation_patterns:
            citation_count += len(re.findall(pattern, article))
        
        # Require at least 5 citations for a good article
        return citation_count >= 5

    async def _add_citations_to_article(self, llm: LlamaClient, article: str, source_map: Dict) -> str:
        """Add citations to an article that lacks them"""
        
        source_list = ""
        for i, source_info in source_map.items():
            source_list += f"[{i}] {source_info['title']}: {source_info['snippet'][:100]}...\n"
        
        prompt = f"""Add proper citations to this article using the numbered sources provided.

ARTICLE (needs citations):
{article}

AVAILABLE SOURCES:
{source_list}

INSTRUCTIONS:
- Add citation numbers [1], [2], [3] etc. after factual statements
- Match facts in the article to appropriate sources
- Add [X] immediately after each claim that can be supported by a source
- Don't change the article content, just add citations

Return the article with citations added:"""

        try:
            return await llm.generate(prompt, temperature=0.3)
        except Exception as e:
            logger.error(f"Error adding citations: {e}")
            return article  # Return original if citation addition fails

    def _generate_references_section(self, source_map: Dict) -> str:
        """Generate a references section"""
        
        references = "\n\n## References\n\n"
        
        for i, source_info in source_map.items():
            title = source_info['title']
            url = source_info['url']
            references += f"{i}. [{title}]({url})\n"
        
        return references