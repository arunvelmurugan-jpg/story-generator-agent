"""
Evaluation Dataset Management
"""

import json
import logging
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class EvaluationExample:
    """A single evaluation example."""
    id: str
    query: str
    expected_response: Optional[str] = None
    context: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)


@dataclass
class EvaluationDataset:
    """Collection of evaluation examples."""
    name: str
    description: str
    examples: List[EvaluationExample]
    version: str = "1.0"
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __len__(self) -> int:
        return len(self.examples)
    
    def __iter__(self):
        return iter(self.examples)
    
    def filter_by_tags(self, tags: List[str]) -> "EvaluationDataset":
        """Filter examples by tags."""
        filtered = [e for e in self.examples if any(t in e.tags for t in tags)]
        return EvaluationDataset(
            name=f"{self.name}_filtered",
            description=f"Filtered by tags: {tags}",
            examples=filtered,
            version=self.version,
            metadata=self.metadata
        )
    
    def sample(self, n: int) -> "EvaluationDataset":
        """Get a random sample of examples."""
        import random
        sampled = random.sample(self.examples, min(n, len(self.examples)))
        return EvaluationDataset(
            name=f"{self.name}_sample",
            description=f"Sample of {n} examples",
            examples=sampled,
            version=self.version,
            metadata=self.metadata
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "metadata": self.metadata,
            "examples": [
                {
                    "id": e.id,
                    "query": e.query,
                    "expected_response": e.expected_response,
                    "context": e.context,
                    "metadata": e.metadata,
                    "tags": e.tags
                }
                for e in self.examples
            ]
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EvaluationDataset":
        """Create from dictionary."""
        examples = [
            EvaluationExample(
                id=e.get("id", str(i)),
                query=e["query"],
                expected_response=e.get("expected_response"),
                context=e.get("context"),
                metadata=e.get("metadata", {}),
                tags=e.get("tags", [])
            )
            for i, e in enumerate(data.get("examples", []))
        ]
        return cls(
            name=data.get("name", "unnamed"),
            description=data.get("description", ""),
            examples=examples,
            version=data.get("version", "1.0"),
            metadata=data.get("metadata", {})
        )


class DatasetLoader:
    """Utility for loading evaluation datasets."""
    
    @staticmethod
    def from_json(path: str) -> EvaluationDataset:
        """Load dataset from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
        return EvaluationDataset.from_dict(data)
    
    @staticmethod
    def from_jsonl(path: str) -> EvaluationDataset:
        """Load dataset from JSONL file."""
        examples = []
        with open(path, 'r') as f:
            for i, line in enumerate(f):
                if line.strip():
                    data = json.loads(line)
                    examples.append(EvaluationExample(
                        id=data.get("id", str(i)),
                        query=data["query"],
                        expected_response=data.get("expected_response"),
                        context=data.get("context"),
                        metadata=data.get("metadata", {}),
                        tags=data.get("tags", [])
                    ))
        
        return EvaluationDataset(
            name=Path(path).stem,
            description=f"Loaded from {path}",
            examples=examples
        )
    
    @staticmethod
    def from_csv(path: str) -> EvaluationDataset:
        """Load dataset from CSV file."""
        import csv
        examples = []
        with open(path, 'r') as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                examples.append(EvaluationExample(
                    id=row.get("id", str(i)),
                    query=row.get("query", ""),
                    expected_response=row.get("expected_response"),
                    context=row.get("context"),
                    tags=row.get("tags", "").split(",") if row.get("tags") else []
                ))
        
        return EvaluationDataset(
            name=Path(path).stem,
            description=f"Loaded from {path}",
            examples=examples
        )
    
    @staticmethod
    def save_json(dataset: EvaluationDataset, path: str):
        """Save dataset to JSON file."""
        with open(path, 'w') as f:
            json.dump(dataset.to_dict(), f, indent=2)


__all__ = ["EvaluationExample", "EvaluationDataset", "DatasetLoader"]
