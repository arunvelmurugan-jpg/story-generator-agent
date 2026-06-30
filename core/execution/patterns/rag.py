"""
RAG Execution Pattern for PHTN.AI Sub-Agent Framework

Implements Retrieval-Augmented Generation pattern.
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, AsyncIterator, TYPE_CHECKING

from .base import BasePattern, ExecutionContext

if TYPE_CHECKING:
    from ...agent import AgentInput, AgentOutput

logger = logging.getLogger(__name__)


class RAGPattern(BasePattern):
    """
    Retrieval-Augmented Generation (RAG) execution pattern.
    
    This pattern:
    1. Retrieves relevant documents based on the query
    2. Augments the context with retrieved information
    3. Generates response grounded in retrieved content
    
    Best for:
    - Knowledge-based Q&A
    - Document-grounded responses
    - Reducing hallucinations
    - Enterprise knowledge retrieval
    """
    
    pattern_name = "rag"
    
    RAG_SYSTEM_PROMPT = """You are an AI assistant that answers questions based on the provided context.

Instructions:
1. Use ONLY the information from the provided context to answer questions
2. If the context doesn't contain enough information, say so clearly
3. Cite the relevant parts of the context when possible
4. Do not make up information not present in the context

Context:
{context}

Remember: Only use information from the context above. If you're unsure, acknowledge the limitation.
"""
    
    async def execute(
        self,
        input_data: "AgentInput",
        context: ExecutionContext,
    ) -> "AgentOutput":
        """Execute RAG pattern."""
        from ...agent import AgentOutput, ContentType
        
        self.reset_trace()
        
        rag_config = self.config.rag_config or {}
        retrieval_first = rag_config.get("retrieval_first", True)
        citation_required = rag_config.get("citation_required", False)
        
        query = input_data.content
        if isinstance(query, dict):
            query = query.get("query", str(query))
        
        if retrieval_first or not context.retrieved_documents:
            retrieved_docs = await self._retrieve_documents(str(query), context)
            context.retrieved_documents = retrieved_docs
        
        self.add_trace_step(
            step_type="retrieval",
            input_data={"query": query},
            output_data={"documents": len(context.retrieved_documents)},
        )
        
        messages = self._build_messages(input_data, context, citation_required)
        
        response = await self.call_llm(messages)
        content = response.get("content", "")
        
        output = AgentOutput(
            content=content,
            content_type=ContentType.TEXT,
            success=True,
            token_usage=response.get("usage", {}),
            metadata={
                "retrieved_documents": len(context.retrieved_documents),
                "sources": [
                    {
                        "id": doc.get("id"),
                        "score": doc.get("score"),
                        "source": doc.get("metadata", {}).get("source"),
                    }
                    for doc in context.retrieved_documents
                ],
            },
        )
        
        return await self.post_process(output, input_data, context)
    
    async def execute_stream(
        self,
        input_data: "AgentInput",
        context: ExecutionContext,
    ) -> AsyncIterator[Dict[str, Any]]:
        """Execute RAG pattern with streaming."""
        self.reset_trace()
        
        rag_config = self.config.rag_config or {}
        retrieval_first = rag_config.get("retrieval_first", True)
        citation_required = rag_config.get("citation_required", False)
        
        yield {"type": "start", "pattern": self.pattern_name}
        
        query = input_data.content
        if isinstance(query, dict):
            query = query.get("query", str(query))
        
        yield {"type": "retrieval_start"}
        
        if retrieval_first or not context.retrieved_documents:
            retrieved_docs = await self._retrieve_documents(str(query), context)
            context.retrieved_documents = retrieved_docs
        
        yield {
            "type": "retrieval_complete",
            "document_count": len(context.retrieved_documents),
            "sources": [
                {
                    "id": doc.get("id"),
                    "score": doc.get("score"),
                }
                for doc in context.retrieved_documents[:5]
            ],
        }
        
        messages = self._build_messages(input_data, context, citation_required)
        
        yield {"type": "generation_start"}
        
        full_content = ""
        async for chunk in self.llm_client.stream(messages):
            content_delta = chunk.get("content", "")
            full_content += content_delta
            yield {
                "type": "content",
                "content": content_delta,
            }
        
        yield {
            "type": "end",
            "document_count": len(context.retrieved_documents),
        }
    
    async def _retrieve_documents(
        self,
        query: str,
        context: ExecutionContext,
    ) -> List[Dict[str, Any]]:
        """Retrieve relevant documents."""
        rag_config = self.config.rag_config or {}
        top_k = rag_config.get("top_k", 5)
        
        try:
            docs = await self.memory_manager.semantic_search(
                query=query,
                top_k=top_k,
            )
            return docs
        except Exception as e:
            logger.warning(f"Retrieval failed: {e}")
            return []
    
    def _build_messages(
        self,
        input_data: "AgentInput",
        context: ExecutionContext,
        citation_required: bool,
    ) -> List[Dict[str, Any]]:
        """Build messages for RAG."""
        retrieved_context = self._format_retrieved_documents(context.retrieved_documents)
        
        system_prompt = self.RAG_SYSTEM_PROMPT.format(context=retrieved_context)
        
        if citation_required:
            system_prompt += "\n\nIMPORTANT: Always cite the source document when using information from the context."
        
        if context.memory_context:
            system_prompt += f"\n\nConversation history:\n{context.memory_context}"
        
        messages = [{"role": "system", "content": system_prompt}]
        
        messages.extend(context.messages)
        
        user_content = input_data.content
        if isinstance(user_content, dict):
            user_content = user_content.get("query", str(user_content))
        
        messages.append({"role": "user", "content": str(user_content)})
        
        return messages
    
    def _format_retrieved_documents(self, documents: List[Dict[str, Any]]) -> str:
        """Format retrieved documents for context."""
        if not documents:
            return "No relevant documents found."
        
        formatted = []
        for i, doc in enumerate(documents, 1):
            content = doc.get("content", doc.get("text", ""))
            source = doc.get("metadata", {}).get("source", f"Document {i}")
            score = doc.get("score", "N/A")
            
            formatted.append(f"[Document {i}] (Source: {source}, Relevance: {score})")
            formatted.append(content)
            formatted.append("")
        
        return "\n".join(formatted)
