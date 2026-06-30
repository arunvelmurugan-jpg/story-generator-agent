"""
AWS Bedrock LLM Provider

Supports AWS Bedrock foundation models:
- Amazon Titan
- Anthropic Claude (via Bedrock)
- Meta Llama 2
- Cohere Command
- AI21 Jurassic
- Stability AI

Equivalent to n8n's AWS Bedrock Chat Model node.
"""

import logging
import json
from typing import List, Dict, Any, Optional, AsyncIterator
from dataclasses import dataclass

from ..base import BaseLLMProvider, LLMResponse, StreamChunk, Message

logger = logging.getLogger(__name__)


@dataclass
class BedrockConfig:
    """Configuration for AWS Bedrock provider."""
    region: str = "us-east-1"
    model_id: str = "anthropic.claude-3-sonnet-20240229-v1:0"
    temperature: float = 0.7
    max_tokens: int = 4096
    top_p: float = 0.9
    top_k: int = 250
    stop_sequences: Optional[List[str]] = None
    aws_access_key_id: Optional[str] = None
    aws_secret_access_key: Optional[str] = None
    aws_session_token: Optional[str] = None
    profile_name: Optional[str] = None
    endpoint_url: Optional[str] = None


class BedrockProvider(BaseLLMProvider):
    """
    AWS Bedrock LLM provider.
    
    Supports multiple foundation models through AWS Bedrock:
    - anthropic.claude-3-opus-20240229-v1:0
    - anthropic.claude-3-sonnet-20240229-v1:0
    - anthropic.claude-3-haiku-20240307-v1:0
    - anthropic.claude-v2:1
    - amazon.titan-text-express-v1
    - amazon.titan-text-lite-v1
    - meta.llama2-70b-chat-v1
    - meta.llama2-13b-chat-v1
    - cohere.command-text-v14
    - cohere.command-light-text-v14
    - ai21.j2-ultra-v1
    - ai21.j2-mid-v1
    """
    
    SUPPORTED_MODELS = [
        "anthropic.claude-3-opus-20240229-v1:0",
        "anthropic.claude-3-sonnet-20240229-v1:0",
        "anthropic.claude-3-haiku-20240307-v1:0",
        "anthropic.claude-v2:1",
        "anthropic.claude-v2",
        "anthropic.claude-instant-v1",
        "amazon.titan-text-express-v1",
        "amazon.titan-text-lite-v1",
        "amazon.titan-text-premier-v1:0",
        "amazon.titan-embed-text-v1",
        "amazon.titan-embed-text-v2:0",
        "meta.llama2-70b-chat-v1",
        "meta.llama2-13b-chat-v1",
        "meta.llama3-8b-instruct-v1:0",
        "meta.llama3-70b-instruct-v1:0",
        "cohere.command-text-v14",
        "cohere.command-light-text-v14",
        "cohere.command-r-v1:0",
        "cohere.command-r-plus-v1:0",
        "cohere.embed-english-v3",
        "cohere.embed-multilingual-v3",
        "ai21.j2-ultra-v1",
        "ai21.j2-mid-v1",
        "ai21.jamba-instruct-v1:0",
        "mistral.mistral-7b-instruct-v0:2",
        "mistral.mixtral-8x7b-instruct-v0:1",
        "mistral.mistral-large-2402-v1:0",
    ]
    
    def __init__(self, config: Optional[BedrockConfig] = None, **kwargs):
        self.config = config or BedrockConfig(**kwargs)
        self._client = None
        self._runtime_client = None
    
    @property
    def provider_name(self) -> str:
        return "bedrock"
    
    @property
    def supported_models(self) -> List[str]:
        return self.SUPPORTED_MODELS
    
    def _get_boto3_session(self):
        """Get or create boto3 session."""
        try:
            import boto3
            
            session_kwargs = {}
            if self.config.profile_name:
                session_kwargs["profile_name"] = self.config.profile_name
            if self.config.region:
                session_kwargs["region_name"] = self.config.region
            
            return boto3.Session(**session_kwargs)
        except ImportError:
            raise ImportError("boto3 is required for Bedrock provider. Install with: pip install boto3")
    
    def _get_runtime_client(self):
        """Get or create Bedrock runtime client."""
        if self._runtime_client is None:
            session = self._get_boto3_session()
            
            client_kwargs = {"service_name": "bedrock-runtime"}
            if self.config.endpoint_url:
                client_kwargs["endpoint_url"] = self.config.endpoint_url
            if self.config.aws_access_key_id:
                client_kwargs["aws_access_key_id"] = self.config.aws_access_key_id
            if self.config.aws_secret_access_key:
                client_kwargs["aws_secret_access_key"] = self.config.aws_secret_access_key
            if self.config.aws_session_token:
                client_kwargs["aws_session_token"] = self.config.aws_session_token
            
            self._runtime_client = session.client(**client_kwargs)
        
        return self._runtime_client
    
    def _format_messages_for_claude(self, messages: List[Message]) -> Dict[str, Any]:
        """Format messages for Claude models on Bedrock."""
        system_prompt = ""
        formatted_messages = []
        
        for msg in messages:
            if msg.role == "system":
                system_prompt = msg.content
            else:
                formatted_messages.append({
                    "role": msg.role,
                    "content": [{"type": "text", "text": msg.content}]
                })
        
        return {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            "top_k": self.config.top_k,
            "system": system_prompt,
            "messages": formatted_messages
        }
    
    def _format_messages_for_titan(self, messages: List[Message]) -> Dict[str, Any]:
        """Format messages for Amazon Titan models."""
        prompt = ""
        for msg in messages:
            if msg.role == "system":
                prompt += f"System: {msg.content}\n\n"
            elif msg.role == "user":
                prompt += f"User: {msg.content}\n\n"
            elif msg.role == "assistant":
                prompt += f"Assistant: {msg.content}\n\n"
        
        prompt += "Assistant:"
        
        return {
            "inputText": prompt,
            "textGenerationConfig": {
                "maxTokenCount": self.config.max_tokens,
                "temperature": self.config.temperature,
                "topP": self.config.top_p,
                "stopSequences": self.config.stop_sequences or []
            }
        }
    
    def _format_messages_for_llama(self, messages: List[Message]) -> Dict[str, Any]:
        """Format messages for Meta Llama models."""
        prompt = "<s>"
        for msg in messages:
            if msg.role == "system":
                prompt += f"[INST] <<SYS>>\n{msg.content}\n<</SYS>>\n\n"
            elif msg.role == "user":
                prompt += f"[INST] {msg.content} [/INST]"
            elif msg.role == "assistant":
                prompt += f" {msg.content} </s><s>"
        
        return {
            "prompt": prompt,
            "max_gen_len": self.config.max_tokens,
            "temperature": self.config.temperature,
            "top_p": self.config.top_p
        }
    
    def _format_messages(self, messages: List[Message], model_id: str) -> Dict[str, Any]:
        """Format messages based on model type."""
        if "claude" in model_id.lower():
            return self._format_messages_for_claude(messages)
        elif "titan" in model_id.lower():
            return self._format_messages_for_titan(messages)
        elif "llama" in model_id.lower():
            return self._format_messages_for_llama(messages)
        else:
            return self._format_messages_for_claude(messages)
    
    def _parse_response(self, response_body: Dict[str, Any], model_id: str) -> str:
        """Parse response based on model type."""
        if "claude" in model_id.lower():
            content = response_body.get("content", [])
            if content and isinstance(content, list):
                return content[0].get("text", "")
            return ""
        elif "titan" in model_id.lower():
            results = response_body.get("results", [])
            if results:
                return results[0].get("outputText", "")
            return ""
        elif "llama" in model_id.lower():
            return response_body.get("generation", "")
        else:
            return str(response_body)
    
    async def complete(
        self,
        messages: List[Message],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> LLMResponse:
        """Generate completion using AWS Bedrock."""
        import asyncio
        
        model_id = model or self.config.model_id
        
        if temperature is not None:
            self.config.temperature = temperature
        if max_tokens is not None:
            self.config.max_tokens = max_tokens
        
        body = self._format_messages(messages, model_id)
        
        try:
            client = self._get_runtime_client()
            
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: client.invoke_model(
                    modelId=model_id,
                    body=json.dumps(body),
                    contentType="application/json",
                    accept="application/json"
                )
            )
            
            response_body = json.loads(response["body"].read())
            content = self._parse_response(response_body, model_id)
            
            usage = {}
            if "claude" in model_id.lower():
                usage = {
                    "prompt_tokens": response_body.get("usage", {}).get("input_tokens", 0),
                    "completion_tokens": response_body.get("usage", {}).get("output_tokens", 0),
                    "total_tokens": (
                        response_body.get("usage", {}).get("input_tokens", 0) +
                        response_body.get("usage", {}).get("output_tokens", 0)
                    )
                }
            
            return LLMResponse(
                content=content,
                model=model_id,
                provider=self.provider_name,
                usage=usage,
                metadata={
                    "stop_reason": response_body.get("stop_reason"),
                    "model_id": model_id
                }
            )
        except Exception as e:
            logger.error(f"Bedrock completion error: {e}")
            raise
    
    async def stream(
        self,
        messages: List[Message],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> AsyncIterator[StreamChunk]:
        """Stream completion using AWS Bedrock."""
        import asyncio
        
        model_id = model or self.config.model_id
        
        if temperature is not None:
            self.config.temperature = temperature
        if max_tokens is not None:
            self.config.max_tokens = max_tokens
        
        body = self._format_messages(messages, model_id)
        
        try:
            client = self._get_runtime_client()
            
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: client.invoke_model_with_response_stream(
                    modelId=model_id,
                    body=json.dumps(body),
                    contentType="application/json",
                    accept="application/json"
                )
            )
            
            for event in response.get("body", []):
                chunk = event.get("chunk")
                if chunk:
                    chunk_data = json.loads(chunk.get("bytes", b"{}").decode())
                    
                    if "claude" in model_id.lower():
                        if chunk_data.get("type") == "content_block_delta":
                            delta = chunk_data.get("delta", {})
                            content = delta.get("text", "")
                            yield StreamChunk(content=content, done=False)
                        elif chunk_data.get("type") == "message_stop":
                            yield StreamChunk(content="", done=True)
                    else:
                        content = self._parse_response(chunk_data, model_id)
                        yield StreamChunk(content=content, done=False)
            
            yield StreamChunk(content="", done=True)
        except Exception as e:
            logger.error(f"Bedrock streaming error: {e}")
            raise
    
    async def generate_embeddings(
        self,
        texts: List[str],
        model: Optional[str] = None
    ) -> List[List[float]]:
        """Generate embeddings using Bedrock Titan or Cohere."""
        import asyncio
        
        model_id = model or "amazon.titan-embed-text-v1"
        client = self._get_runtime_client()
        embeddings = []
        
        for text in texts:
            if "titan" in model_id.lower():
                body = {"inputText": text}
            elif "cohere" in model_id.lower():
                body = {"texts": [text], "input_type": "search_document"}
            else:
                body = {"inputText": text}
            
            try:
                response = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: client.invoke_model(
                        modelId=model_id,
                        body=json.dumps(body),
                        contentType="application/json",
                        accept="application/json"
                    )
                )
                
                response_body = json.loads(response["body"].read())
                
                if "titan" in model_id.lower():
                    embeddings.append(response_body.get("embedding", []))
                elif "cohere" in model_id.lower():
                    embs = response_body.get("embeddings", [[]])
                    embeddings.append(embs[0] if embs else [])
                else:
                    embeddings.append(response_body.get("embedding", []))
            except Exception as e:
                logger.error(f"Bedrock embeddings error: {e}")
                raise
        
        return embeddings
    
    async def list_models(self) -> List[Dict[str, Any]]:
        """List available foundation models in Bedrock."""
        try:
            session = self._get_boto3_session()
            bedrock_client = session.client("bedrock")
            
            response = bedrock_client.list_foundation_models()
            return response.get("modelSummaries", [])
        except Exception as e:
            logger.error(f"Bedrock list models error: {e}")
            return []
    
    async def health_check(self) -> bool:
        """Check Bedrock connectivity."""
        try:
            models = await self.list_models()
            return len(models) > 0
        except Exception:
            return False


__all__ = ["BedrockProvider", "BedrockConfig"]
