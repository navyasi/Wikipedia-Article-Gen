import asyncio
import json
from typing import List, Optional, Dict, Any
import httpx
from pydantic import BaseModel
from loguru import logger
import os


class LlamaConfig(BaseModel):
    """Configuration for Llama client"""
    host: str = "http://localhost:11434"
    model: str = "llama3.2:latest"
    temperature: float = 0.7
    max_tokens: int = 2048
    timeout: int = 60


class LlamaClient:
    """Async client for Ollama/Llama 3.2 with proper error handling and retries"""
    
    def __init__(self, config: LlamaConfig):
        self.config = config
        self.client = httpx.AsyncClient(timeout=config.timeout)
        
    async def __aenter__(self):
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.client.aclose()
    
    async def generate(
        self, 
        prompt: str, 
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_retries: int = 3
    ) -> str:
        """Generate text using Llama model with retries"""
        
        temperature = temperature or self.config.temperature
        
        payload = {
            "model": self.config.model,
            "prompt": prompt,
            "system": system_prompt,
            "options": {
                "temperature": temperature,
                "num_predict": self.config.max_tokens
            },
            "stream": False
        }
        
        for attempt in range(max_retries):
            try:
                response = await self.client.post(
                    f"{self.config.host}/api/generate",
                    json=payload
                )
                response.raise_for_status()
                
                result = response.json()
                return result.get("response", "").strip()
                
            except httpx.RequestError as e:
                logger.warning(f"Request failed (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt == max_retries - 1:
                    raise
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
                
            except httpx.HTTPStatusError as e:
                logger.error(f"HTTP error: {e.response.status_code} - {e.response.text}")
                raise
                
    async def chat(
        self, 
        messages: List[Dict[str, str]], 
        temperature: Optional[float] = None,
        max_retries: int = 3
    ) -> str:
        """Chat completion using Ollama chat API"""
        
        temperature = temperature or self.config.temperature
        
        payload = {
            "model": self.config.model,
            "messages": messages,
            "options": {
                "temperature": temperature,
                "num_predict": self.config.max_tokens
            },
            "stream": False
        }
        
        for attempt in range(max_retries):
            try:
                response = await self.client.post(
                    f"{self.config.host}/api/chat",
                    json=payload
                )
                response.raise_for_status()
                
                result = response.json()
                return result.get("message", {}).get("content", "").strip()
                
            except httpx.RequestError as e:
                logger.warning(f"Chat request failed (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt == max_retries - 1:
                    raise
                await asyncio.sleep(2 ** attempt)
                
            except httpx.HTTPStatusError as e:
                logger.error(f"HTTP error: {e.response.status_code} - {e.response.text}")
                raise