"""
Mock LLM Provider for PHTN.AI Sub-Agent Framework

Provides a demo/testing LLM provider that generates realistic responses
when no actual LLM API keys are configured. Useful for:
- Local development without API keys
- Testing the framework flow
- Demo purposes
"""

import asyncio
import random
from typing import Any, Dict, List, Optional, AsyncIterator
from datetime import datetime

from ..base import BaseLLMProvider, LLMResponse, StreamChunk


class MockLLMProvider(BaseLLMProvider):
    """
    Mock LLM provider for testing and demo purposes.
    
    Generates contextual responses based on the input message,
    simulating a real LLM's behavior including:
    - ReAct-style reasoning
    - Tool calling decisions
    - Final answers
    """
    
    provider_name = "mock"
    
    def __init__(self, model: str = "mock-gpt-4", **kwargs):
        super().__init__(model, **kwargs)
        self._call_count = 0
    
    async def complete(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs,
    ) -> LLMResponse:
        """Generate a mock completion response."""
        await asyncio.sleep(0.3 + random.random() * 0.5)
        
        self._call_count += 1
        
        last_user_message = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                last_user_message = msg.get("content", "")
                break
        
        is_observation = "Observation:" in last_user_message
        has_tool_result = is_observation or "Error executing tool" in last_user_message
        
        if self._call_count == 1:
            response_content = self._generate_initial_response(last_user_message, tools)
        elif has_tool_result:
            response_content = self._generate_post_tool_response(last_user_message)
        else:
            response_content = self._generate_final_answer(last_user_message)
        
        input_tokens = sum(len(m.get("content", "")) // 4 for m in messages)
        output_tokens = len(response_content) // 4
        
        return LLMResponse(
            content=response_content,
            model=self.model,
            usage={
                "input": input_tokens,
                "output": output_tokens,
                "total": input_tokens + output_tokens
            },
            finish_reason="stop",
            raw_response={"mock": True, "call_count": self._call_count}
        )
    
    def _generate_initial_response(self, user_message: str, tools: Optional[List[Dict]] = None) -> str:
        """Generate initial ReAct response with tool call."""
        user_lower = user_message.lower()
        
        if any(word in user_lower for word in ["help", "support", "ticket", "issue", "problem"]):
            tool_name = "search_knowledge_base"
            tool_input = {"query": user_message[:100], "limit": 5}
            thought = f"The user is asking about support or has an issue. I should search the knowledge base to find relevant information."
        elif any(word in user_lower for word in ["customer", "account", "lookup", "find"]):
            tool_name = "lookup_customer"
            tool_input = {"query": user_message[:50]}
            thought = f"The user wants information about a customer or account. I should look up the customer details."
        elif any(word in user_lower for word in ["create", "new", "ticket", "escalate"]):
            tool_name = "create_ticket"
            tool_input = {"title": "Support Request", "description": user_message[:200], "priority": "medium"}
            thought = f"The user wants to create a support ticket. I should help them create one."
        else:
            return self._generate_final_answer(user_message)
        
        return f"""Thought: {thought}

Action: {tool_name} with input: {{"query": "{user_message[:50]}...", "limit": 5}}"""
    
    def _generate_post_tool_response(self, observation: str) -> str:
        """Generate response after receiving tool observation."""
        return f"""Thought: I received the tool result. Based on this information, I can now provide a helpful response to the user.

Final Answer: Based on my search, I found relevant information to help you. Here's what I can tell you:

1. **Knowledge Base Results**: I searched our documentation and found several relevant articles that may help with your query.

2. **Recommendations**: 
   - Check our FAQ section for common questions
   - Review the troubleshooting guide for step-by-step solutions
   - Contact support if you need further assistance

3. **Next Steps**: If this doesn't fully address your question, please provide more details and I'll be happy to help further.

Is there anything specific you'd like me to elaborate on?"""
    
    def _generate_final_answer(self, user_message: str) -> str:
        """Generate a direct final answer."""
        user_lower = user_message.lower()
        
        if any(word in user_lower for word in ["hello", "hi", "hey", "greetings"]):
            response = "Hello! I'm your Customer Support Agent. I'm here to help you with any questions or issues you may have. How can I assist you today?"
        elif any(word in user_lower for word in ["help", "what can you do", "capabilities"]):
            response = """I'm your Customer Support Agent, and I can help you with:

1. **Information Lookup**: Search our knowledge base for answers to your questions
2. **Customer Support**: Look up account information and resolve issues  
3. **Ticket Management**: Create, track, and escalate support tickets
4. **General Assistance**: Answer questions about our products and services

What would you like help with today?"""
        elif any(word in user_lower for word in ["thank", "thanks", "bye", "goodbye"]):
            response = "You're welcome! If you have any more questions in the future, don't hesitate to reach out. Have a great day!"
        else:
            response = f"""Thank you for your message. I understand you're asking about: "{user_message[:100]}..."

Let me help you with that. As your Customer Support Agent, I can:
- Search our knowledge base for relevant information
- Look up your account details if needed
- Create a support ticket if this requires further assistance

Would you like me to search for more information, or is there something specific I can help you with?"""
        
        return f"""Thought: I can provide a helpful response directly to the user without needing to use any tools.

Final Answer: {response}"""
    
    async def stream(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs,
    ) -> AsyncIterator[StreamChunk]:
        """Stream mock completion response."""
        response = await self.complete(messages, tools, **kwargs)
        
        words = response.content.split()
        for i, word in enumerate(words):
            await asyncio.sleep(0.02)
            yield StreamChunk(
                content=word + " ",
                finish_reason=None if i < len(words) - 1 else "stop",
                usage=response.usage if i == len(words) - 1 else None
            )
    
    def reset(self):
        """Reset the call count for a new conversation."""
        self._call_count = 0
    
    async def health_check(self) -> Dict[str, Any]:
        """Check mock provider health - always healthy."""
        return {
            "status": "healthy",
            "provider": "mock",
            "model": self.model,
            "message": "Mock LLM provider is ready for demo/testing"
        }
