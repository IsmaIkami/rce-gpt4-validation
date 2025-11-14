#!/usr/bin/env python3
"""
Domain Mapping Module for RCE-LLM Empirical Validation
Author: Ismail Sialyen
Publication: DOI 10.5281/zenodo.17360372

This module provides robust mapping between semantic query domains and RCE API domains.

IMPORTANT DISTINCTION:
- Domain: RCE API parameter (general, medical, legal, financial, technical)
- Coherence Modules: Relationship types tested (mu_reason, mu_units, mu_time, etc.)

The domain determines which RCE knowledge base and reasoning context to use.
The coherence modules determine which semantic relationships are validated.
"""

from typing import Dict, List, Optional
from enum import Enum


class RCEDomain(str, Enum):
    """Valid RCE API domains as per API specification"""
    GENERAL = "general"
    MEDICAL = "medical"
    LEGAL = "legal"
    FINANCIAL = "financial"
    TECHNICAL = "technical"


class CoherenceModule(str, Enum):
    """RCE coherence modules (relationship types)"""
    MU_REASON = "mu_reason"      # Pure logic and reasoning
    MU_UNITS = "mu_units"        # Unit conversion and dimensional analysis
    MU_TIME = "mu_time"          # Temporal reasoning and constraints
    MU_ARITH = "mu_arith"        # Arithmetic validation
    MU_COREF = "mu_coref"        # Coreference resolution
    MU_ENTAIL = "mu_entail"      # Textual entailment


# Semantic domain to RCE API domain mapping
# This mapping is based on the nature of the query content
SEMANTIC_TO_API_DOMAIN = {
    # Logic and reasoning queries
    "logic": RCEDomain.GENERAL,
    "reasoning": RCEDomain.GENERAL,

    # Unit and measurement queries
    "units": RCEDomain.TECHNICAL,
    "measurement": RCEDomain.TECHNICAL,

    # Medical domain queries
    "medical": RCEDomain.MEDICAL,
    "health": RCEDomain.MEDICAL,
    "clinical": RCEDomain.MEDICAL,

    # Legal domain queries
    "legal": RCEDomain.LEGAL,
    "law": RCEDomain.LEGAL,
    "regulatory": RCEDomain.LEGAL,

    # Financial domain queries
    "financial": RCEDomain.FINANCIAL,
    "economic": RCEDomain.FINANCIAL,
    "investment": RCEDomain.FINANCIAL,

    # Temporal queries
    "temporal": RCEDomain.GENERAL,
    "time": RCEDomain.GENERAL,

    # Arithmetic queries
    "arithmetic": RCEDomain.GENERAL,
    "math": RCEDomain.GENERAL,
    "numerical": RCEDomain.GENERAL,

    # Factual queries
    "factual": RCEDomain.GENERAL,
    "knowledge": RCEDomain.GENERAL,

    # Coreference queries
    "coreference": RCEDomain.GENERAL,
    "reference": RCEDomain.GENERAL,

    # Technical queries
    "technical": RCEDomain.TECHNICAL,
    "engineering": RCEDomain.TECHNICAL,
    "scientific": RCEDomain.TECHNICAL,
}


class DomainMapper:
    """
    Robust domain mapping with validation and fallback strategies.

    This class ensures all queries can be mapped to valid RCE API domains,
    with comprehensive error handling and logging.
    """

    def __init__(self, strict_mode: bool = False, logger: Optional[object] = None):
        """
        Initialize domain mapper.

        Args:
            strict_mode: If True, raise exceptions on unknown domains.
                        If False, use fallback to 'general'.
            logger: Optional logger for warnings and info messages.
        """
        self.strict_mode = strict_mode
        self.logger = logger
        self._mapping_cache: Dict[str, str] = {}
        self._unknown_domains: set = set()

    def map_domain(self, semantic_domain: str) -> str:
        """
        Map semantic domain to RCE API domain.

        Args:
            semantic_domain: The semantic domain from query metadata

        Returns:
            Valid RCE API domain string

        Raises:
            ValueError: If strict_mode=True and domain is unknown
        """
        if not semantic_domain:
            return RCEDomain.GENERAL.value

        # Check cache first
        if semantic_domain in self._mapping_cache:
            return self._mapping_cache[semantic_domain]

        # Normalize to lowercase for case-insensitive matching
        normalized_domain = semantic_domain.lower().strip()

        # Check if already a valid RCE domain
        try:
            if normalized_domain in [d.value for d in RCEDomain]:
                result = normalized_domain
                self._mapping_cache[semantic_domain] = result
                return result
        except Exception:
            pass

        # Map semantic domain to RCE API domain
        if normalized_domain in SEMANTIC_TO_API_DOMAIN:
            result = SEMANTIC_TO_API_DOMAIN[normalized_domain].value
            self._mapping_cache[semantic_domain] = result
            return result

        # Unknown domain handling
        if normalized_domain not in self._unknown_domains:
            self._unknown_domains.add(normalized_domain)
            if self.logger:
                self.logger.warning(
                    f"Unknown semantic domain '{semantic_domain}'. "
                    f"Using fallback: {RCEDomain.GENERAL.value}"
                )

        if self.strict_mode:
            raise ValueError(
                f"Unknown semantic domain: '{semantic_domain}'. "
                f"Valid domains: {list(SEMANTIC_TO_API_DOMAIN.keys())}"
            )

        # Fallback to general
        result = RCEDomain.GENERAL.value
        self._mapping_cache[semantic_domain] = result
        return result

    def validate_domain(self, domain: str) -> bool:
        """
        Check if a domain is a valid RCE API domain.

        Args:
            domain: Domain string to validate

        Returns:
            True if valid RCE API domain, False otherwise
        """
        try:
            return domain.lower() in [d.value for d in RCEDomain]
        except Exception:
            return False

    def validate_coherence_modules(self, modules: List[str]) -> List[str]:
        """
        Validate and filter coherence module list.

        Args:
            modules: List of coherence module identifiers

        Returns:
            List of valid coherence modules
        """
        if not modules:
            return []

        valid_modules = [m.value for m in CoherenceModule]
        validated = []

        for module in modules:
            if module in valid_modules:
                validated.append(module)
            elif self.logger:
                self.logger.warning(f"Invalid coherence module: {module}")

        return validated

    def get_mapping_stats(self) -> Dict[str, any]:
        """
        Get statistics about domain mapping usage.

        Returns:
            Dictionary with mapping statistics
        """
        return {
            "total_mappings": len(self._mapping_cache),
            "unknown_domains": list(self._unknown_domains),
            "cache_size": len(self._mapping_cache)
        }

    @staticmethod
    def get_all_valid_domains() -> List[str]:
        """Get list of all valid RCE API domains."""
        return [d.value for d in RCEDomain]

    @staticmethod
    def get_all_coherence_modules() -> List[str]:
        """Get list of all valid coherence modules."""
        return [m.value for m in CoherenceModule]


# Convenience functions for standalone usage
def map_domain(semantic_domain: str, strict: bool = False) -> str:
    """
    Standalone function to map semantic domain to RCE API domain.

    Args:
        semantic_domain: The semantic domain to map
        strict: If True, raise exception on unknown domain

    Returns:
        Valid RCE API domain string
    """
    mapper = DomainMapper(strict_mode=strict)
    return mapper.map_domain(semantic_domain)


def validate_domain(domain: str) -> bool:
    """Validate if domain is a valid RCE API domain."""
    mapper = DomainMapper()
    return mapper.validate_domain(domain)


if __name__ == "__main__":
    # Self-test
    import sys

    mapper = DomainMapper(strict_mode=False)

    print("=== RCE Domain Mapper Self-Test ===\n")

    # Test valid RCE domains
    print("Testing valid RCE API domains:")
    for domain in RCEDomain:
        result = mapper.map_domain(domain.value)
        status = "✓" if result == domain.value else "✗"
        print(f"  {status} {domain.value} -> {result}")

    # Test semantic domain mappings
    print("\nTesting semantic domain mappings:")
    test_cases = [
        ("logic", RCEDomain.GENERAL.value),
        ("medical", RCEDomain.MEDICAL.value),
        ("units", RCEDomain.TECHNICAL.value),
        ("temporal", RCEDomain.GENERAL.value),
        ("arithmetic", RCEDomain.GENERAL.value),
        ("unknown_domain", RCEDomain.GENERAL.value),  # Should fallback
    ]

    for semantic, expected in test_cases:
        result = mapper.map_domain(semantic)
        status = "✓" if result == expected else "✗"
        print(f"  {status} '{semantic}' -> {result} (expected: {expected})")

    # Test coherence modules
    print("\nTesting coherence module validation:")
    test_modules = ["mu_reason", "mu_units", "mu_invalid", "mu_time"]
    validated = mapper.validate_coherence_modules(test_modules)
    print(f"  Input: {test_modules}")
    print(f"  Valid: {validated}")

    # Show stats
    print("\nMapping statistics:")
    stats = mapper.get_mapping_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")

    print("\n=== Self-Test Complete ===")
    sys.exit(0)
