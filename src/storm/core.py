# src/storm/core.py - COMPLETE VERSION WITH FULL ARTICLE GENERATION
import asyncio
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from pydantic import BaseModel
import wikipedia
from loguru import logger

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
        
        if target_topic and config.target_article_exclusion:
            wikipedia_url = f"en.wikipedia.org/wiki/{target_topic.replace(' ', '_')}"
            self.search_manager.exclude_domains.append(wikipedia_url)
    
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
    
    async def _decompose_question(self, llm: LlamaClient, topic: str, question: str) -> List[str]:
        prompt = f"Break this question into 2-4 search queries:\n\nTopic: {topic}\nQuestion: {question}\n\nFormat as:\n- query 1\n- query 2"
        response = await llm.generate(prompt, temperature=0.7)
        
        queries = []
        for line in response.split('\n'):
            line = line.strip()
            if line.startswith('- '):
                queries.append(line[2:].strip())
        return queries[:4]
    
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