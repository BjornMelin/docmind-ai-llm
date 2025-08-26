"""Cache-specific Pydantic models for DocMind AI Dual-Layer Caching System.

This module contains data models specifically designed for the dual-layer caching
architecture combining LlamaIndex IngestionCache (Layer 1) with GPTCache semantic
similarity caching (Layer 2), following ADR-009 requirements for 80-95%
performance gains.

HYBRID MODEL ORGANIZATION:
Following the hybrid model organization strategy, these models are colocated within
the cache domain because they are:
- Tightly coupled to cache coordination workflows
- Specific to IngestionCache and GPTCache operations
- Domain-specific to dual-layer caching pipeline
- Used primarily within the caching modules

Models:
    CacheLayer: Enum for cache layer identifiers
    CacheHitResult: Result of cache hit operations
    CacheStats: Cache statistics and performance metrics
    CacheCoordinationResult: Result of multi-agent cache coordination
    CacheError: Custom exception for cache operation errors

Architecture Decision:
    These models are placed within the cache module following domain-driven
    design principles. They are domain-specific and tightly coupled to the
    caching system, making this the most appropriate location per
    the hybrid organization strategy.
"""

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class CacheLayer(str, Enum):
    """Cache layer identifiers."""

    INGESTION = "ingestion"
    SEMANTIC = "semantic"
    GPT = "gpt"


class CacheHitResult(BaseModel):
    """Result of cache hit operation."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    hit: bool = Field(description="Whether cache hit occurred")
    layer: CacheLayer = Field(description="Cache layer that provided the hit")
    data: Any = Field(default=None, description="Cached data")
    similarity_score: float | None = Field(
        default=None, description="Semantic similarity score"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Cache hit metadata"
    )


class CacheStats(BaseModel):
    """Cache statistics and performance metrics."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    total_requests: int = Field(default=0, description="Total cache requests")
    ingestion_hits: int = Field(default=0, description="IngestionCache hits")
    ingestion_misses: int = Field(default=0, description="IngestionCache misses")
    semantic_hits: int = Field(default=0, description="Semantic cache hits")
    semantic_misses: int = Field(default=0, description="Semantic cache misses")
    hit_rate: float = Field(default=0.0, description="Overall hit rate (0.0-1.0)")
    size_mb: float = Field(default=0.0, description="Total cache size in MB")
    performance_score: float = Field(
        default=0.0, description="Performance score (0.0-1.0)"
    )

    def calculate_hit_rate(self) -> float:
        """Calculate and update overall hit rate.

        Returns:
            Overall hit rate as a float between 0.0 and 1.0
        """
        total_hits = self.ingestion_hits + self.semantic_hits
        if self.total_requests > 0:
            self.hit_rate = total_hits / self.total_requests
        else:
            self.hit_rate = 0.0
        return self.hit_rate

    def get_ingestion_hit_rate(self) -> float:
        """Calculate IngestionCache hit rate.

        Returns:
            IngestionCache hit rate as a float between 0.0 and 1.0
        """
        total_ingestion_requests = self.ingestion_hits + self.ingestion_misses
        if total_ingestion_requests > 0:
            return self.ingestion_hits / total_ingestion_requests
        return 0.0

    def get_semantic_hit_rate(self) -> float:
        """Calculate Semantic cache hit rate.

        Returns:
            Semantic cache hit rate as a float between 0.0 and 1.0
        """
        total_semantic_requests = self.semantic_hits + self.semantic_misses
        if total_semantic_requests > 0:
            return self.semantic_hits / total_semantic_requests
        return 0.0


class CacheCoordinationResult(BaseModel):
    """Result of multi-agent cache coordination."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    active_agents: int = Field(description="Number of active agents")
    cache_stats: CacheStats = Field(description="Aggregated cache statistics")
    coordination_time: float = Field(description="Coordination processing time")
    shared_entries: int = Field(default=0, description="Number of shared cache entries")
    conflicts_resolved: int = Field(
        default=0, description="Number of cache conflicts resolved"
    )

    def get_coordination_efficiency(self) -> float:
        """Calculate coordination efficiency score.

        Returns:
            Coordination efficiency score based on hit rate and coordination time
        """
        base_efficiency = self.cache_stats.hit_rate
        time_penalty = min(0.2, self.coordination_time / 10.0)  # Max 20% penalty
        return max(0.0, base_efficiency - time_penalty)


class CacheError(Exception):
    """Custom exception for cache operation errors."""

    pass
