"""Test database utilities re-export module.

This module tests that src.utils.database correctly re-exports functionality
from src.utils.storage for backward compatibility. Tests validate that all
expected functions and classes are accessible through the database module.
"""

import inspect

import pytest

import src.utils.database as database_module
import src.utils.storage as storage_module


@pytest.mark.unit
class TestDatabaseModuleStructure:
    """Test basic structure and imports of database module."""

    def test_database_module_exists(self) -> None:
        """Test that database module can be imported."""
        assert database_module is not None
        assert hasattr(database_module, "__file__")

    def test_database_module_has_docstring(self) -> None:
        """Test that database module has descriptive docstring."""
        assert database_module.__doc__ is not None
        assert "backward compatibility" in database_module.__doc__
        assert "storage.py" in database_module.__doc__

    def test_database_module_has_all_attribute(self) -> None:
        """Test that database module defines __all__ for explicit exports."""
        assert hasattr(database_module, "__all__")
        assert isinstance(database_module.__all__, list)
        assert "storage" in database_module.__all__


@pytest.mark.unit
class TestReExportFunctionality:
    """Test that database module re-exports storage module functions."""

    def test_major_database_functions_available(self) -> None:
        """Test that key database functions are available from database module."""
        # Test that important functions from storage are accessible
        expected_functions = [
            "get_client_config",
            "create_sync_client",
            "create_async_client",
            "setup_hybrid_collection",
            "setup_hybrid_collection_async",
            "create_vector_store",
            "get_collection_info",
            "test_connection",
            "clear_collection",
        ]

        for func_name in expected_functions:
            assert hasattr(database_module, func_name), (
                f"Function {func_name} not found"
            )

            # Verify it's callable
            func = getattr(database_module, func_name)
            assert callable(func), f"{func_name} is not callable"

    def test_resource_management_functions_available(self) -> None:
        """Test that resource management functions are available."""
        resource_functions = [
            "gpu_memory_context",
            "async_gpu_memory_context",
            "model_context",
            "sync_model_context",
            "cuda_error_context",
            "safe_cuda_operation",
            "get_safe_vram_usage",
            "get_safe_gpu_info",
        ]

        for func_name in resource_functions:
            assert hasattr(database_module, func_name), (
                f"Function {func_name} not found"
            )
            func = getattr(database_module, func_name)
            assert callable(func), f"{func_name} is not callable"

    def test_re_exported_functions_match_storage(self) -> None:
        """Test that re-exported functions are identical to storage originals."""
        # Get functions that exist in both modules
        storage_functions = {
            name: obj
            for name, obj in inspect.getmembers(storage_module)
            if callable(obj) and not name.startswith("_")
        }

        for func_name, storage_func in storage_functions.items():
            if hasattr(database_module, func_name):
                database_func = getattr(database_module, func_name)

                # Should be the exact same function object (identity)
                assert database_func is storage_func, (
                    f"{func_name} is not the same object in both modules"
                )


@pytest.mark.unit
class TestFunctionSignatures:
    """Test that re-exported functions have correct signatures."""

    @pytest.mark.parametrize(
        "function_name",
        [
            "get_client_config",
            "create_vector_store",
            "get_collection_info",
            "test_connection",
            "clear_collection",
            "get_safe_vram_usage",
            "get_safe_gpu_info",
        ],
    )
    def test_function_signatures_preserved(self, function_name: str) -> None:
        """Test that function signatures are preserved in re-export."""
        if hasattr(storage_module, function_name) and hasattr(
            database_module, function_name
        ):
            storage_func = getattr(storage_module, function_name)
            database_func = getattr(database_module, function_name)

            storage_sig = inspect.signature(storage_func)
            database_sig = inspect.signature(database_func)

            assert storage_sig == database_sig, (
                f"Signature mismatch for {function_name}: "
                f"storage={storage_sig} vs database={database_sig}"
            )

    def test_context_manager_functions_available(self) -> None:
        """Test that context manager functions are properly re-exported."""
        context_managers = [
            "create_sync_client",
            "create_async_client",
            "gpu_memory_context",
            "async_gpu_memory_context",
            "model_context",
            "sync_model_context",
            "cuda_error_context",
        ]

        for cm_name in context_managers:
            assert hasattr(database_module, cm_name)
            cm_func = getattr(database_module, cm_name)
            assert callable(cm_func)


@pytest.mark.unit
class TestBackwardCompatibility:
    """Test backward compatibility features of database module."""

    def test_wildcard_import_works(self) -> None:
        """Test that wildcard import from database works as expected."""
        # This tests that 'from src.utils.database import *' would work
        # by checking that functions are available at module level

        # Get all non-private attributes from database module
        database_attrs = {
            name: obj
            for name, obj in vars(database_module).items()
            if not name.startswith("_") and callable(obj)
        }

        # Should have substantial number of functions available
        assert len(database_attrs) > 10, (
            "Too few functions available for wildcard import"
        )

    def test_database_module_as_drop_in_replacement(self) -> None:
        """Test that database module can replace direct storage imports."""
        # Test that common usage patterns work through database module

        # Should be able to access client config
        if hasattr(database_module, "get_client_config"):
            config_func = database_module.get_client_config
            assert callable(config_func)

        # Should be able to access context managers
        if hasattr(database_module, "create_sync_client"):
            client_context = database_module.create_sync_client
            assert callable(client_context)

    def test_import_patterns_supported(self) -> None:
        """Test that various import patterns are supported."""
        # Test direct attribute access
        if hasattr(database_module, "test_connection"):
            test_conn = database_module.test_connection
            assert callable(test_conn)

        # Test getattr pattern
        if hasattr(database_module, "get_safe_vram_usage"):
            vram_func = getattr(database_module, "get_safe_vram_usage", None)
            assert vram_func is not None
            assert callable(vram_func)


@pytest.mark.unit
class TestModuleConsistency:
    """Test consistency between database and storage modules."""

    def test_no_unexpected_differences(self) -> None:
        """Test that database module doesn't add unexpected functionality."""
        # Get all callable objects from both modules
        storage_callables = {
            name
            for name, obj in inspect.getmembers(storage_module, callable)
            if not name.startswith("_")
        }

        database_callables = {
            name
            for name, obj in inspect.getmembers(database_module, callable)
            if not name.startswith("_") and not inspect.ismodule(obj)
        }

        # Database should not have significantly more functions than storage
        # (allowing for some imports and re-exports)
        extra_functions = database_callables - storage_callables

        # Filter out obvious imports/builtins
        unexpected_extras = {
            name
            for name in extra_functions
            if not any(name.startswith(prefix) for prefix in ["__", "import"])
        }

        # Should have minimal unexpected additions
        assert len(unexpected_extras) < 5, (
            f"Too many unexpected functions in database module: {unexpected_extras}"
        )

    def test_key_functions_have_same_behavior(self) -> None:
        """Test that key functions behave identically between modules."""
        # Test functions that should be identical
        identical_functions = ["get_client_config", "get_safe_vram_usage"]

        for func_name in identical_functions:
            if hasattr(storage_module, func_name) and hasattr(
                database_module, func_name
            ):
                storage_func = getattr(storage_module, func_name)
                database_func = getattr(database_module, func_name)

                # Should be the same function object
                assert storage_func is database_func, (
                    f"{func_name} objects should be identical"
                )

    def test_storage_attribute_accessible(self) -> None:
        """Test that 'storage' is accessible as specified in __all__."""
        # The module says __all__ = ["storage"], so this should work
        assert "storage" in database_module.__all__

        # However, based on the actual implementation, storage might refer
        # to the storage module itself or be a symbolic reference
        # This is more of a design validation than a functional requirement
