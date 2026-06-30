"""
Toxicity Detection for PHTN.AI Sub-Agent Framework

Provides toxicity detection using various providers:
- Google Perspective API
- OpenAI Moderation API
- HuggingFace models
- Local pattern matching
"""

import asyncio
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from enum import Enum

from ..observability.otel_logging import get_logger

logger = get_logger(__name__)


class ToxicityProvider(str, Enum):
    """Supported toxicity detection providers."""
    PERSPECTIVE = "perspective"
    OPENAI = "openai"
    HUGGINGFACE = "huggingface"
    LOCAL = "local"


class ToxicityCategory(str, Enum):
    """Toxicity categories."""
    TOXICITY = "toxicity"
    SEVERE_TOXICITY = "severe_toxicity"
    IDENTITY_ATTACK = "identity_attack"
    INSULT = "insult"
    PROFANITY = "profanity"
    THREAT = "threat"
    SEXUALLY_EXPLICIT = "sexually_explicit"
    HARASSMENT = "harassment"
    HATE_SPEECH = "hate_speech"
    VIOLENCE = "violence"
    SELF_HARM = "self_harm"


@dataclass
class ToxicityScore:
    """Toxicity detection score."""
    category: ToxicityCategory
    score: float
    flagged: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "category": self.category.value,
            "score": self.score,
            "flagged": self.flagged,
        }


@dataclass
class ToxicityResult:
    """Toxicity detection result."""
    is_toxic: bool
    overall_score: float
    scores: List[ToxicityScore] = field(default_factory=list)
    flagged_categories: List[str] = field(default_factory=list)
    provider: str = "unknown"
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_toxic": self.is_toxic,
            "overall_score": self.overall_score,
            "scores": [s.to_dict() for s in self.scores],
            "flagged_categories": self.flagged_categories,
            "provider": self.provider,
            "metadata": self.metadata,
        }


@dataclass
class ToxicityConfig:
    """Toxicity detection configuration."""
    provider: ToxicityProvider = ToxicityProvider.LOCAL
    api_key: Optional[str] = None
    threshold: float = 0.7
    categories: List[ToxicityCategory] = field(default_factory=lambda: [
        ToxicityCategory.TOXICITY,
        ToxicityCategory.SEVERE_TOXICITY,
        ToxicityCategory.HARASSMENT,
        ToxicityCategory.HATE_SPEECH,
    ])
    action: str = "flag"
    model: Optional[str] = None


class BaseToxicityDetector(ABC):
    """Abstract base class for toxicity detectors."""
    
    def __init__(self, config: ToxicityConfig):
        self.config = config
    
    @abstractmethod
    async def detect(self, text: str) -> ToxicityResult:
        """Detect toxicity in text."""
        pass


class LocalToxicityDetector(BaseToxicityDetector):
    """Local pattern-based toxicity detector."""
    
    TOXIC_PATTERNS = {
        ToxicityCategory.PROFANITY: [
            r"\b(fuck|shit|damn|ass|bitch|crap|bastard)\b",
        ],
        ToxicityCategory.THREAT: [
            r"\b(kill|murder|hurt|attack|destroy|die)\s+(you|them|him|her)\b",
            r"\bi('ll|'m going to|will)\s+(kill|hurt|attack)\b",
        ],
        ToxicityCategory.HARASSMENT: [
            r"\b(stupid|idiot|moron|dumb|loser)\b",
            r"\byou('re| are)\s+(worthless|useless|pathetic)\b",
        ],
        ToxicityCategory.HATE_SPEECH: [
            r"\b(hate|despise)\s+(all|every)\s+\w+\b",
        ],
        ToxicityCategory.SELF_HARM: [
            r"\b(suicide|kill myself|end my life|self.?harm)\b",
        ],
    }
    
    async def detect(self, text: str) -> ToxicityResult:
        text_lower = text.lower()
        scores = []
        flagged_categories = []
        max_score = 0.0
        
        for category, patterns in self.TOXIC_PATTERNS.items():
            if category not in self.config.categories:
                continue
            
            match_count = 0
            for pattern in patterns:
                matches = re.findall(pattern, text_lower, re.IGNORECASE)
                match_count += len(matches)
            
            score = min(1.0, match_count * 0.3) if match_count > 0 else 0.0
            flagged = score >= self.config.threshold
            
            scores.append(ToxicityScore(
                category=category,
                score=score,
                flagged=flagged,
            ))
            
            if flagged:
                flagged_categories.append(category.value)
            
            max_score = max(max_score, score)
        
        return ToxicityResult(
            is_toxic=max_score >= self.config.threshold,
            overall_score=max_score,
            scores=scores,
            flagged_categories=flagged_categories,
            provider="local",
        )


class PerspectiveToxicityDetector(BaseToxicityDetector):
    """Google Perspective API toxicity detector."""
    
    CATEGORY_MAPPING = {
        ToxicityCategory.TOXICITY: "TOXICITY",
        ToxicityCategory.SEVERE_TOXICITY: "SEVERE_TOXICITY",
        ToxicityCategory.IDENTITY_ATTACK: "IDENTITY_ATTACK",
        ToxicityCategory.INSULT: "INSULT",
        ToxicityCategory.PROFANITY: "PROFANITY",
        ToxicityCategory.THREAT: "THREAT",
        ToxicityCategory.SEXUALLY_EXPLICIT: "SEXUALLY_EXPLICIT",
    }
    
    async def detect(self, text: str) -> ToxicityResult:
        try:
            import aiohttp
            import os
            
            api_key = self.config.api_key or os.getenv("PERSPECTIVE_API_KEY")
            if not api_key:
                logger.warning("Perspective API key not configured, falling back to local")
                return await LocalToxicityDetector(self.config).detect(text)
            
            url = f"https://commentanalyzer.googleapis.com/v1alpha1/comments:analyze?key={api_key}"
            
            requested_attributes = {}
            for category in self.config.categories:
                if category in self.CATEGORY_MAPPING:
                    requested_attributes[self.CATEGORY_MAPPING[category]] = {}
            
            if not requested_attributes:
                requested_attributes = {"TOXICITY": {}}
            
            payload = {
                "comment": {"text": text},
                "requestedAttributes": requested_attributes,
                "languages": ["en"],
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload) as resp:
                    if resp.status != 200:
                        logger.error(f"Perspective API error: {resp.status}")
                        return await LocalToxicityDetector(self.config).detect(text)
                    
                    data = await resp.json()
            
            scores = []
            flagged_categories = []
            max_score = 0.0
            
            attribute_scores = data.get("attributeScores", {})
            for category in self.config.categories:
                api_name = self.CATEGORY_MAPPING.get(category)
                if api_name and api_name in attribute_scores:
                    score_data = attribute_scores[api_name]
                    score = score_data.get("summaryScore", {}).get("value", 0.0)
                    flagged = score >= self.config.threshold
                    
                    scores.append(ToxicityScore(
                        category=category,
                        score=score,
                        flagged=flagged,
                    ))
                    
                    if flagged:
                        flagged_categories.append(category.value)
                    
                    max_score = max(max_score, score)
            
            return ToxicityResult(
                is_toxic=max_score >= self.config.threshold,
                overall_score=max_score,
                scores=scores,
                flagged_categories=flagged_categories,
                provider="perspective",
            )
            
        except Exception as e:
            logger.error(f"Perspective API error: {e}")
            return await LocalToxicityDetector(self.config).detect(text)


class OpenAIToxicityDetector(BaseToxicityDetector):
    """OpenAI Moderation API toxicity detector."""
    
    CATEGORY_MAPPING = {
        ToxicityCategory.HATE_SPEECH: "hate",
        ToxicityCategory.HARASSMENT: "harassment",
        ToxicityCategory.SELF_HARM: "self-harm",
        ToxicityCategory.SEXUALLY_EXPLICIT: "sexual",
        ToxicityCategory.VIOLENCE: "violence",
    }
    
    async def detect(self, text: str) -> ToxicityResult:
        try:
            import openai
            import os
            
            api_key = self.config.api_key or os.getenv("OPENAI_API_KEY")
            if not api_key:
                logger.warning("OpenAI API key not configured, falling back to local")
                return await LocalToxicityDetector(self.config).detect(text)
            
            client = openai.AsyncOpenAI(api_key=api_key)
            response = await client.moderations.create(input=text)
            
            result = response.results[0]
            category_scores = result.category_scores
            
            scores = []
            flagged_categories = []
            max_score = 0.0
            
            for category in self.config.categories:
                api_name = self.CATEGORY_MAPPING.get(category)
                if api_name:
                    score = getattr(category_scores, api_name.replace("-", "_"), 0.0)
                    flagged = score >= self.config.threshold
                    
                    scores.append(ToxicityScore(
                        category=category,
                        score=score,
                        flagged=flagged,
                    ))
                    
                    if flagged:
                        flagged_categories.append(category.value)
                    
                    max_score = max(max_score, score)
            
            return ToxicityResult(
                is_toxic=result.flagged or max_score >= self.config.threshold,
                overall_score=max_score,
                scores=scores,
                flagged_categories=flagged_categories,
                provider="openai",
            )
            
        except ImportError:
            logger.error("OpenAI package not installed")
            return await LocalToxicityDetector(self.config).detect(text)
        except Exception as e:
            logger.error(f"OpenAI Moderation error: {e}")
            return await LocalToxicityDetector(self.config).detect(text)


class HuggingFaceToxicityDetector(BaseToxicityDetector):
    """HuggingFace model-based toxicity detector."""
    
    def __init__(self, config: ToxicityConfig):
        super().__init__(config)
        self._model = None
        self._tokenizer = None
    
    async def detect(self, text: str) -> ToxicityResult:
        try:
            from transformers import pipeline
            
            if self._model is None:
                model_name = self.config.model or "unitary/toxic-bert"
                self._model = pipeline("text-classification", model=model_name)
                logger.info(f"Loaded toxicity model: {model_name}")
            
            results = self._model(text)
            
            scores = []
            max_score = 0.0
            
            for result in results:
                label = result.get("label", "").lower()
                score = result.get("score", 0.0)
                
                if "toxic" in label:
                    category = ToxicityCategory.TOXICITY
                elif "hate" in label:
                    category = ToxicityCategory.HATE_SPEECH
                elif "threat" in label:
                    category = ToxicityCategory.THREAT
                else:
                    category = ToxicityCategory.TOXICITY
                
                flagged = score >= self.config.threshold
                scores.append(ToxicityScore(
                    category=category,
                    score=score,
                    flagged=flagged,
                ))
                
                max_score = max(max_score, score)
            
            return ToxicityResult(
                is_toxic=max_score >= self.config.threshold,
                overall_score=max_score,
                scores=scores,
                flagged_categories=[s.category.value for s in scores if s.flagged],
                provider="huggingface",
            )
            
        except ImportError:
            logger.error("Transformers package not installed")
            return await LocalToxicityDetector(self.config).detect(text)
        except Exception as e:
            logger.error(f"HuggingFace toxicity error: {e}")
            return await LocalToxicityDetector(self.config).detect(text)


def create_toxicity_detector(config: ToxicityConfig) -> BaseToxicityDetector:
    """
    Factory function to create toxicity detector.
    
    Args:
        config: Toxicity configuration
        
    Returns:
        Toxicity detector instance
    """
    detectors = {
        ToxicityProvider.PERSPECTIVE: PerspectiveToxicityDetector,
        ToxicityProvider.OPENAI: OpenAIToxicityDetector,
        ToxicityProvider.HUGGINGFACE: HuggingFaceToxicityDetector,
        ToxicityProvider.LOCAL: LocalToxicityDetector,
    }
    
    detector_class = detectors.get(config.provider, LocalToxicityDetector)
    return detector_class(config)
