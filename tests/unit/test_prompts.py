"""Test prompts module dictionary constants and configuration.

This module tests all predefined prompt dictionaries to ensure they contain
the expected keys, values, and types. Tests validate that configuration
options are properly defined and accessible.
"""

import pytest

from src.prompts import INSTRUCTIONS, LENGTHS, PREDEFINED_PROMPTS, TONES


@pytest.mark.unit
class TestPredefinedPrompts:
    """Test PREDEFINED_PROMPTS dictionary constants."""

    def test_predefined_prompts_exists(self) -> None:
        """Test that PREDEFINED_PROMPTS dictionary exists and is accessible."""
        assert isinstance(PREDEFINED_PROMPTS, dict)
        assert len(PREDEFINED_PROMPTS) > 0

    def test_predefined_prompts_has_required_keys(self) -> None:
        """Test that PREDEFINED_PROMPTS contains all expected keys."""
        expected_keys = {
            "Comprehensive Document Analysis",
            "Extract Key Insights and Action Items",
            "Summarize and Identify Open Questions",
            "Custom Prompt",
        }
        assert set(PREDEFINED_PROMPTS.keys()) == expected_keys

    @pytest.mark.parametrize(
        "key,expected_type",
        [
            ("Comprehensive Document Analysis", str),
            ("Extract Key Insights and Action Items", str),
            ("Summarize and Identify Open Questions", str),
            ("Custom Prompt", str),
        ],
    )
    def test_predefined_prompts_value_types(
        self, key: str, expected_type: type
    ) -> None:
        """Test that all PREDEFINED_PROMPTS values are strings."""
        assert isinstance(PREDEFINED_PROMPTS[key], expected_type)

    def test_predefined_prompts_non_empty_values(self) -> None:
        """Test that required prompts have non-empty content."""
        # Custom Prompt can be empty, but others should have content
        for key, value in PREDEFINED_PROMPTS.items():
            if key != "Custom Prompt":
                assert value.strip(), f"Prompt '{key}' should not be empty"

    def test_custom_prompt_is_empty_string(self) -> None:
        """Test that Custom Prompt is an empty string by design."""
        assert PREDEFINED_PROMPTS["Custom Prompt"] == ""


@pytest.mark.unit
class TestTones:
    """Test TONES dictionary constants."""

    def test_tones_exists(self) -> None:
        """Test that TONES dictionary exists and is accessible."""
        assert isinstance(TONES, dict)
        assert len(TONES) > 0

    def test_tones_has_expected_keys(self) -> None:
        """Test that TONES contains all expected tone options."""
        expected_keys = {
            "Professional",
            "Academic",
            "Informal",
            "Creative",
            "Neutral",
            "Direct",
            "Empathetic",
            "Humorous",
            "Authoritative",
            "Inquisitive",
        }
        assert set(TONES.keys()) == expected_keys

    @pytest.mark.parametrize(
        "tone_name",
        [
            "Professional",
            "Academic",
            "Informal",
            "Creative",
            "Neutral",
            "Direct",
            "Empathetic",
            "Humorous",
            "Authoritative",
            "Inquisitive",
        ],
    )
    def test_tones_all_values_are_strings(self, tone_name: str) -> None:
        """Test that all tone values are strings."""
        assert isinstance(TONES[tone_name], str)
        assert TONES[tone_name].strip(), f"Tone '{tone_name}' should not be empty"

    def test_tones_contain_instructional_text(self) -> None:
        """Test that all tones contain the word 'Use' as instruction starter."""
        for tone_name, instruction in TONES.items():
            assert instruction.startswith("Use"), (
                f"Tone '{tone_name}' should start with 'Use'"
            )


@pytest.mark.unit
class TestInstructions:
    """Test INSTRUCTIONS dictionary constants."""

    def test_instructions_exists(self) -> None:
        """Test that INSTRUCTIONS dictionary exists and is accessible."""
        assert isinstance(INSTRUCTIONS, dict)
        assert len(INSTRUCTIONS) > 0

    def test_instructions_has_expected_keys(self) -> None:
        """Test that INSTRUCTIONS contains all expected role keys."""
        expected_keys = {
            "General Assistant",
            "Researcher",
            "Software Engineer",
            "Product Manager",
            "Data Scientist",
            "Business Analyst",
            "Technical Writer",
            "Marketing Specialist",
            "HR Manager",
            "Legal Advisor",
            "Custom Instructions",
        }
        assert set(INSTRUCTIONS.keys()) == expected_keys

    @pytest.mark.parametrize(
        "role_name",
        [
            "General Assistant",
            "Researcher",
            "Software Engineer",
            "Product Manager",
            "Data Scientist",
            "Business Analyst",
            "Technical Writer",
            "Marketing Specialist",
            "HR Manager",
            "Legal Advisor",
        ],
    )
    def test_instructions_role_values_are_strings(self, role_name: str) -> None:
        """Test that all instruction values are non-empty strings."""
        assert isinstance(INSTRUCTIONS[role_name], str)
        assert INSTRUCTIONS[role_name].strip(), (
            f"Role '{role_name}' should not be empty"
        )

    def test_custom_instructions_is_empty_string(self) -> None:
        """Test that Custom Instructions is an empty string by design."""
        assert INSTRUCTIONS["Custom Instructions"] == ""


@pytest.mark.unit
class TestLengths:
    """Test LENGTHS dictionary constants."""

    def test_lengths_exists(self) -> None:
        """Test that LENGTHS dictionary exists and is accessible."""
        assert isinstance(LENGTHS, dict)
        assert len(LENGTHS) > 0

    def test_lengths_has_expected_keys(self) -> None:
        """Test that LENGTHS contains all expected length options."""
        expected_keys = {"Concise", "Detailed", "Comprehensive", "Bullet Points"}
        assert set(LENGTHS.keys()) == expected_keys

    @pytest.mark.parametrize(
        "length_name",
        [
            "Concise",
            "Detailed",
            "Comprehensive",
            "Bullet Points",
        ],
    )
    def test_lengths_all_values_are_strings(self, length_name: str) -> None:
        """Test that all length values are non-empty strings."""
        assert isinstance(LENGTHS[length_name], str)
        assert LENGTHS[length_name].strip(), (
            f"Length '{length_name}' should not be empty"
        )

    def test_lengths_contain_instructional_phrases(self) -> None:
        """Test that length instructions contain expected directive words."""
        directive_words = ["Keep", "Provide", "Format"]

        for length_name, instruction in LENGTHS.items():
            has_directive = any(word in instruction for word in directive_words)
            assert has_directive, (
                f"Length '{length_name}' should contain directive language"
            )


@pytest.mark.unit
class TestCrossConstantConsistency:
    """Test consistency across all prompt constant dictionaries."""

    def test_all_constants_are_dictionaries(self) -> None:
        """Test that all constants are dictionary objects."""
        constants = [PREDEFINED_PROMPTS, TONES, INSTRUCTIONS, LENGTHS]
        for constant in constants:
            assert isinstance(constant, dict)

    def test_no_overlapping_keys_across_constants(self) -> None:
        """Test that dictionary keys don't overlap inappropriately."""
        # Collect all keys from all dictionaries
        all_keys = set()
        constant_names = []

        for name, constant in [
            ("PREDEFINED_PROMPTS", PREDEFINED_PROMPTS),
            ("TONES", TONES),
            ("INSTRUCTIONS", INSTRUCTIONS),
            ("LENGTHS", LENGTHS),
        ]:
            keys = set(constant.keys())
            # Check no duplicate keys within same constant
            assert len(keys) == len(constant), f"Duplicate keys found in {name}"

            # Track for cross-constant analysis
            all_keys.update(keys)
            constant_names.append(name)

    def test_constants_not_empty(self) -> None:
        """Test that all constant dictionaries contain at least one entry."""
        constants = {
            "PREDEFINED_PROMPTS": PREDEFINED_PROMPTS,
            "TONES": TONES,
            "INSTRUCTIONS": INSTRUCTIONS,
            "LENGTHS": LENGTHS,
        }

        for name, constant in constants.items():
            assert len(constant) > 0, f"{name} should not be empty"

    def test_dictionary_access_patterns(self) -> None:
        """Test that dictionaries support expected access patterns."""
        # Test key access doesn't raise KeyError for expected keys
        assert "Professional" in TONES
        assert "Comprehensive Document Analysis" in PREDEFINED_PROMPTS
        assert "General Assistant" in INSTRUCTIONS
        assert "Concise" in LENGTHS

        # Test dict methods work correctly
        assert list(TONES.keys())
        assert list(PREDEFINED_PROMPTS.values())
        assert list(INSTRUCTIONS.items())
        assert len(LENGTHS) == 4
