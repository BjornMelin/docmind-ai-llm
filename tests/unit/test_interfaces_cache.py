"""Test cache interface abstract base class.

This module tests the CacheInterface ABC to ensure it properly defines
abstract methods and enforces implementation requirements for concrete
cache classes.
"""

import inspect
from abc import ABC
from typing import Any

import pytest

from src.interfaces.cache import CacheInterface


@pytest.mark.unit
class TestCacheInterfaceStructure:
    """Test CacheInterface abstract base class structure."""

    def test_cache_interface_is_abstract_base_class(self) -> None:
        """Test that CacheInterface inherits from ABC."""
        assert issubclass(CacheInterface, ABC)
        assert inspect.isabstract(CacheInterface)

    def test_cache_interface_cannot_be_instantiated(self) -> None:
        """Test that CacheInterface cannot be instantiated directly."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            CacheInterface()  # type: ignore

    def test_cache_interface_has_expected_abstract_methods(self) -> None:
        """Test that CacheInterface defines all required abstract methods."""
        abstract_methods = CacheInterface.__abstractmethods__
        expected_methods = {
            "get_document",
            "store_document",
            "clear_cache",
            "get_cache_stats",
        }
        assert abstract_methods == expected_methods

    def test_abstract_methods_have_correct_signatures(self) -> None:
        """Test that abstract methods have expected signatures."""
        # Test get_document signature
        get_doc_sig = inspect.signature(CacheInterface.get_document)
        assert len(get_doc_sig.parameters) == 2  # self + path
        assert "path" in get_doc_sig.parameters
        assert get_doc_sig.parameters["path"].annotation == str
        assert get_doc_sig.return_annotation == Any | None

        # Test store_document signature
        store_doc_sig = inspect.signature(CacheInterface.store_document)
        assert len(store_doc_sig.parameters) == 3  # self + path + result
        assert "path" in store_doc_sig.parameters
        assert "result" in store_doc_sig.parameters
        assert store_doc_sig.parameters["path"].annotation == str
        assert store_doc_sig.parameters["result"].annotation == Any
        assert store_doc_sig.return_annotation == bool

        # Test clear_cache signature
        clear_sig = inspect.signature(CacheInterface.clear_cache)
        assert len(clear_sig.parameters) == 1  # self only
        assert clear_sig.return_annotation == bool

        # Test get_cache_stats signature
        stats_sig = inspect.signature(CacheInterface.get_cache_stats)
        assert len(stats_sig.parameters) == 1  # self only
        assert stats_sig.return_annotation == dict[str, Any]


@pytest.mark.unit
class TestCacheInterfaceMethodDecorators:
    """Test that interface methods are properly decorated."""

    @pytest.mark.parametrize(
        "method_name",
        ["get_document", "store_document", "clear_cache", "get_cache_stats"],
    )
    def test_methods_are_abstract(self, method_name: str) -> None:
        """Test that all interface methods are decorated with @abstractmethod."""
        method = getattr(CacheInterface, method_name)
        assert hasattr(method, "__isabstractmethod__")
        assert method.__isabstractmethod__ is True

    def test_methods_are_async(self) -> None:
        """Test that all interface methods are async."""
        async_methods = [
            "get_document",
            "store_document",
            "clear_cache",
            "get_cache_stats",
        ]

        for method_name in async_methods:
            method = getattr(CacheInterface, method_name)
            assert inspect.iscoroutinefunction(method)


@pytest.mark.unit
class TestConcreteImplementationRequirements:
    """Test requirements for concrete implementations of CacheInterface."""

    def test_incomplete_implementation_fails(self) -> None:
        """Test that partial implementations cannot be instantiated."""

        class IncompleteCache(CacheInterface):
            """Incomplete cache implementation missing required methods."""

            async def get_document(self, path: str) -> Any | None:
                return None

            # Missing: store_document, clear_cache, get_cache_stats

        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            IncompleteCache()  # type: ignore

    def test_complete_implementation_succeeds(self) -> None:
        """Test that complete implementations can be instantiated."""

        class CompleteCache(CacheInterface):
            """Complete cache implementation with all required methods."""

            async def get_document(self, path: str) -> Any | None:
                return None

            async def store_document(self, path: str, result: Any) -> bool:
                return True

            async def clear_cache(self) -> bool:
                return True

            async def get_cache_stats(self) -> dict[str, Any]:
                return {}

        # Should not raise any exception
        cache = CompleteCache()
        assert isinstance(cache, CacheInterface)
        assert isinstance(cache, CompleteCache)

    def test_wrong_method_signature_still_instantiates(self) -> None:
        """Test that wrong signatures don't prevent instantiation (runtime check)."""

        class WrongSignatureCache(CacheInterface):
            """Cache with wrong method signatures."""

            async def get_document(self) -> None:  # Missing path parameter
                return None

            async def store_document(self, path: str, result: Any) -> bool:
                return True

            async def clear_cache(self) -> bool:
                return True

            async def get_cache_stats(self) -> dict[str, Any]:
                return {}

        # Python allows this at instantiation time - typing is for static analysis
        cache = WrongSignatureCache()
        assert isinstance(cache, CacheInterface)


@pytest.mark.unit
class TestCacheInterfaceDocstrings:
    """Test that interface methods have proper documentation."""

    def test_all_methods_have_docstrings(self) -> None:
        """Test that all abstract methods have docstrings."""
        methods_with_docs = [
            "get_document",
            "store_document",
            "clear_cache",
            "get_cache_stats",
        ]

        for method_name in methods_with_docs:
            method = getattr(CacheInterface, method_name)
            assert method.__doc__ is not None
            assert method.__doc__.strip() != ""

    def test_docstrings_contain_expected_sections(self) -> None:
        """Test that docstrings contain Args and Returns sections."""
        # Test get_document docstring
        get_doc = CacheInterface.get_document.__doc__
        assert "Args:" in get_doc
        assert "Returns:" in get_doc
        assert "path:" in get_doc

        # Test store_document docstring
        store_doc = CacheInterface.store_document.__doc__
        assert "Args:" in store_doc
        assert "Returns:" in store_doc
        assert "path:" in store_doc
        assert "result:" in store_doc

        # Test clear_cache docstring
        clear_cache = CacheInterface.clear_cache.__doc__
        assert "Returns:" in clear_cache

        # Test get_cache_stats docstring
        get_stats = CacheInterface.get_cache_stats.__doc__
        assert "Returns:" in get_stats


@pytest.mark.unit
class TestCacheInterfaceTypeAnnotations:
    """Test that interface has proper type annotations."""

    def test_module_imports_typing(self) -> None:
        """Test that cache interface module imports required typing."""
        import src.interfaces.cache as cache_module

        # Check that module has necessary imports
        assert hasattr(cache_module, "ABC")
        assert hasattr(cache_module, "abstractmethod")
        assert hasattr(cache_module, "Any")

    def test_class_has_type_annotations(self) -> None:
        """Test that CacheInterface uses proper type hints."""
        # All methods should have full type annotations
        methods = ["get_document", "store_document", "clear_cache", "get_cache_stats"]

        for method_name in methods:
            method = getattr(CacheInterface, method_name)
            sig = inspect.signature(method)

            # Check that return type is annotated
            assert sig.return_annotation != inspect.Signature.empty

            # Check that parameters (except self) are annotated
            for param_name, param in sig.parameters.items():
                if param_name != "self":
                    assert param.annotation != inspect.Parameter.empty


@pytest.mark.unit
class TestInterfaceUsagePatterns:
    """Test common usage patterns with the interface."""

    def test_interface_can_be_used_in_type_hints(self) -> None:
        """Test that CacheInterface can be used as a type hint."""

        def function_accepting_cache(cache: CacheInterface) -> bool:
            """Function that accepts any CacheInterface implementation."""
            return isinstance(cache, CacheInterface)

        # This should not raise any errors during definition
        assert callable(function_accepting_cache)

    def test_interface_supports_isinstance_checks(self) -> None:
        """Test that isinstance works correctly with implementations."""

        class TestCache(CacheInterface):
            async def get_document(self, path: str) -> Any | None:
                return None

            async def store_document(self, path: str, result: Any) -> bool:
                return True

            async def clear_cache(self) -> bool:
                return True

            async def get_cache_stats(self) -> dict[str, Any]:
                return {}

        cache = TestCache()
        assert isinstance(cache, CacheInterface)
        assert isinstance(cache, ABC)

    def test_interface_supports_hasattr_checks(self) -> None:
        """Test that hasattr works correctly for abstract methods."""
        assert hasattr(CacheInterface, "get_document")
        assert hasattr(CacheInterface, "store_document")
        assert hasattr(CacheInterface, "clear_cache")
        assert hasattr(CacheInterface, "get_cache_stats")
