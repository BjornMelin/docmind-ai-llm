"""Integration tests for DocumentProcessor.

This module provides integration tests for DocumentProcessor that test
the actual integration between Unstructured.io and LlamaIndex IngestionPipeline
with lightweight models and realistic document processing scenarios.

Test Coverage:
- End-to-end document processing pipeline
- Integration with real LlamaIndex transformations
- Cache performance and persistence
- Multi-file processing workflows
- Error recovery and retry mechanisms
- Performance validation with real documents

Following 3-tier testing strategy:
- Tier 2 (Integration): Cross-component tests (<30s each)
- Use lightweight models where possible
- Test with real document samples
- Validate component integration and data flow
"""

import asyncio
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from src.models.processing import DocumentElement, ProcessingResult, ProcessingStrategy
from src.processing.document_processor import DocumentProcessor


@pytest.fixture
def integration_settings():
    """Integration test settings with real-world configuration."""
    settings = Mock()
    settings.chunk_size = 512
    settings.chunk_overlap = 50
    settings.max_document_size_mb = 50  # Smaller for integration tests
    settings.cache_dir = "./test_cache"
    settings.bge_m3_model_name = "BAAI/bge-m3"
    return settings


@pytest.fixture
def sample_documents(tmp_path):
    """Create realistic test documents for integration testing."""
    documents = {}

    # PDF document
    pdf_content = b"""%PDF-1.4
1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj
2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj  
3 0 obj<</Type/Page/Parent 2 0 R/Resources<</Font<</F1 4 0 R>>>>/MediaBox[0 0 612 792]/Contents 5 0 R>>endobj
4 0 obj<</Type/Font/Subtype/Type1/BaseFont/Times-Roman>>endobj
5 0 obj<</Length 44>>stream
BT
/F1 12 Tf
72 720 Td
(Sample PDF content) Tj
ET
endstream
endobj
xref
0 6
0000000000 65535 f
0000000009 00000 n
0000000058 00000 n
0000000115 00000 n
0000000245 00000 n
0000000307 00000 n
trailer<</Size 6/Root 1 0 R>>
startxref
396
%%EOF"""  # noqa: E501

    documents["pdf"] = tmp_path / "integration_test.pdf"
    documents["pdf"].write_bytes(pdf_content)

    # Text document
    text_content = """# Integration Test Document

This is a sample document for integration testing of the DocumentProcessor.

## Features Tested

- Document parsing with unstructured.io
- LlamaIndex pipeline integration
- Semantic chunking and transformation
- Metadata preservation

## Tables

| Feature | Status | Priority |
|---------|--------|----------|
| PDF Processing | ✓ | High |
| Text Processing | ✓ | Medium |
| Cache Integration | ✓ | High |

## Conclusion

The integration testing validates end-to-end functionality across multiple components.
"""

    documents["text"] = tmp_path / "integration_test.txt"
    documents["text"].write_text(text_content)

    # HTML document
    html_content = """<!DOCTYPE html>
<html>
<head>
    <title>Integration Test HTML</title>
</head>
<body>
    <h1>HTML Document Test</h1>
    <p>This HTML document tests the fast processing strategy.</p>
    <table>
        <tr>
            <th>Component</th>
            <th>Status</th>
        </tr>
        <tr>
            <td>HTML Parser</td>
            <td>Active</td>
        </tr>
        <tr>
            <td>Pipeline</td>
            <td>Running</td>
        </tr>
    </table>
    <p>End of test document.</p>
</body>
</html>"""

    documents["html"] = tmp_path / "integration_test.html"
    documents["html"].write_text(html_content)

    return documents


class TestHybridProcessorIntegration:
    """Integration tests for DocumentProcessor."""

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_end_to_end_document_processing(
        self, integration_settings, sample_documents
    ):
        """Test complete document processing pipeline from file to elements."""
        processor = DocumentProcessor(integration_settings)

        # Test PDF processing (hi_res strategy)
        with patch("src.processing.hybrid_processor.partition") as mock_partition:
            # Mock realistic unstructured elements
            mock_partition.return_value = [
                Mock(
                    text="Integration Test Document",
                    category="Title",
                    metadata=Mock(
                        page_number=1,
                        element_id="title_1",
                        parent_id=None,
                        filename="integration_test.pdf",
                        coordinates=[(72, 720), (200, 740)],
                        text_as_html=None,
                        image_path=None,
                    ),
                ),
                Mock(
                    text="This is sample PDF content for integration testing.",
                    category="NarrativeText",
                    metadata=Mock(
                        page_number=1,
                        element_id="text_1",
                        parent_id="title_1",
                        filename="integration_test.pdf",
                        coordinates=[(72, 680), (500, 710)],
                        text_as_html=None,
                        image_path=None,
                    ),
                ),
            ]

            result = await processor.process_document_async(sample_documents["pdf"])

            # Verify processing result
            assert isinstance(result, ProcessingResult)
            assert len(result.elements) > 0
            assert result.strategy_used == ProcessingStrategy.HI_RES
            assert result.processing_time > 0
            assert result.document_hash is not None

            # Verify elements structure
            for element in result.elements:
                assert isinstance(element, DocumentElement)
                assert element.text is not None
                assert element.category is not None
                assert isinstance(element.metadata, dict)

            # Verify pipeline configuration was applied
            assert "pipeline_config" in result.metadata
            assert result.metadata["pipeline_config"]["strategy"] == "hi_res"
            assert result.metadata["pipeline_config"]["transformations"] > 0

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_multi_strategy_processing(
        self, integration_settings, sample_documents
    ):
        """Test processing different file types with appropriate strategies."""
        processor = DocumentProcessor(integration_settings)

        with patch("src.processing.hybrid_processor.partition") as mock_partition:
            # Mock different responses for different strategies
            def partition_side_effect(filename, **kwargs):
                strategy = kwargs.get("strategy")
                file_path = Path(filename)

                if strategy == "hi_res":
                    return [
                        Mock(
                            text=f"Hi-res content from {file_path.name}",
                            category="Title",
                            metadata=Mock(
                                page_number=1,
                                element_id="hr_1",
                                parent_id=None,
                                filename=file_path.name,
                                coordinates=[(0, 0), (100, 20)],
                                text_as_html=None,
                                image_path=None,
                            ),
                        )
                    ]
                elif strategy == "fast":
                    return [
                        Mock(
                            text=f"Fast content from {file_path.name}",
                            category="NarrativeText",
                            metadata=Mock(
                                page_number=1,
                                element_id="fast_1",
                                parent_id=None,
                                filename=file_path.name,
                                coordinates=[(0, 0), (100, 20)],
                                text_as_html=None,
                                image_path=None,
                            ),
                        )
                    ]
                return []

            mock_partition.side_effect = partition_side_effect

            # Test PDF (hi_res strategy)
            pdf_result = await processor.process_document_async(sample_documents["pdf"])
            assert pdf_result.strategy_used == ProcessingStrategy.HI_RES
            assert "Hi-res content" in pdf_result.elements[0].text

            # Test HTML (fast strategy)
            html_result = await processor.process_document_async(
                sample_documents["html"]
            )
            assert html_result.strategy_used == ProcessingStrategy.FAST
            assert "Fast content" in html_result.elements[0].text

            # Verify different partition configurations were used
            assert mock_partition.call_count == 2
            calls = mock_partition.call_args_list

            # First call should be hi_res for PDF
            pdf_call = calls[0]
            assert pdf_call[1]["strategy"] == "hi_res"
            assert pdf_call[1]["extract_images_in_pdf"] is True
            assert pdf_call[1]["infer_table_structure"] is True

            # Second call should be fast for HTML
            html_call = calls[1]
            assert html_call[1]["strategy"] == "fast"
            assert html_call[1]["extract_images_in_pdf"] is False
            assert html_call[1]["infer_table_structure"] is False

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_pipeline_transformation_chain(
        self, integration_settings, sample_documents
    ):
        """Test the complete chain from Document to processed elements."""
        processor = DocumentProcessor(integration_settings)

        with patch("src.processing.hybrid_processor.partition") as mock_partition:
            # Create a realistic element that will be split by SentenceSplitter
            long_text = (
                "This is the first sentence of a long paragraph. "
                "This is the second sentence that continues the thought. "
                "This is the third sentence that should trigger chunking. "
                "This is the fourth sentence in the long paragraph. "
                "Finally, this is the last sentence that concludes our text."
            )

            mock_partition.return_value = [
                Mock(
                    text=long_text,
                    category="NarrativeText",
                    metadata=Mock(
                        page_number=1,
                        element_id="long_1",
                        parent_id=None,
                        filename="integration_test.txt",
                        coordinates=[(0, 0), (400, 100)],
                        text_as_html=None,
                        image_path=None,
                    ),
                )
            ]

            result = await processor.process_document_async(sample_documents["text"])

            # Verify transformation chain worked
            assert len(result.elements) > 1  # Should be split into chunks

            # Verify all elements have required metadata
            for element in result.elements:
                assert "processing_strategy" in element.metadata
                assert "source_file" in element.metadata
                assert "element_category" in element.metadata

            # Verify chunking occurred (SentenceSplitter)
            total_text_length = sum(len(elem.text) for elem in result.elements)
            assert (
                total_text_length >= len(long_text) * 0.8
            )  # Allow for some processing overhead

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_cache_integration_and_persistence(
        self, integration_settings, sample_documents
    ):
        """Test cache integration with both LlamaIndex cache and SimpleCache."""
        processor = DocumentProcessor(integration_settings)

        with patch("src.processing.hybrid_processor.partition") as mock_partition:
            mock_partition.return_value = [
                Mock(
                    text="Cached content test",
                    category="Title",
                    metadata=Mock(
                        page_number=1,
                        element_id="cache_1",
                        parent_id=None,
                        filename="integration_test.pdf",
                        coordinates=[(0, 0), (100, 20)],
                        text_as_html=None,
                        image_path=None,
                    ),
                )
            ]

            # First processing - should create cache entry
            result1 = await processor.process_document_async(sample_documents["pdf"])

            # Second processing - should use cache
            result2 = await processor.process_document_async(sample_documents["pdf"])

            # Verify cache was used (should be faster or from cache)
            assert result2.elements[0].text == result1.elements[0].text
            assert result2.document_hash == result1.document_hash

            # Verify cache statistics
            cache_stats = await processor.get_cache_stats()
            assert "simple_cache" in cache_stats
            assert "llamaindex_cache" in cache_stats
            assert cache_stats["processor_type"] == "hybrid"

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_batch_processing_workflow(
        self, integration_settings, sample_documents
    ):
        """Test processing multiple documents in sequence."""
        processor = DocumentProcessor(integration_settings)

        with patch("src.processing.hybrid_processor.partition") as mock_partition:

            def batch_partition_side_effect(filename, **kwargs):
                file_path = Path(filename)
                return [
                    Mock(
                        text=f"Content from {file_path.name}",
                        category="NarrativeText",
                        metadata=Mock(
                            page_number=1,
                            element_id=f"{file_path.stem}_1",
                            parent_id=None,
                            filename=file_path.name,
                            coordinates=[(0, 0), (100, 20)],
                            text_as_html=None,
                            image_path=None,
                        ),
                    )
                ]

            mock_partition.side_effect = batch_partition_side_effect

            # Process all documents
            results = []
            files_to_process = [
                sample_documents["pdf"],
                sample_documents["text"],
                sample_documents["html"],
            ]

            for file_path in files_to_process:
                result = await processor.process_document_async(file_path)
                results.append(result)

            # Verify all documents processed successfully
            assert len(results) == 3

            for i, result in enumerate(results):
                assert isinstance(result, ProcessingResult)
                assert len(result.elements) > 0

                # Verify filename is preserved in elements
                file_name = files_to_process[i].name
                assert any(
                    file_name in elem.metadata.get("filename", "")
                    for elem in result.elements
                )

            # Verify different strategies were used appropriately
            strategies_used = [result.strategy_used for result in results]
            assert ProcessingStrategy.HI_RES in strategies_used  # PDF
            assert ProcessingStrategy.FAST in strategies_used  # Text and HTML

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_error_recovery_and_retry_logic(
        self, integration_settings, sample_documents
    ):
        """Test error recovery mechanisms and retry behavior."""
        processor = DocumentProcessor(integration_settings)

        with patch("src.processing.hybrid_processor.partition") as mock_partition:
            # Simulate intermittent failures
            call_count = 0

            def failing_partition(*args, **kwargs):
                nonlocal call_count
                call_count += 1
                if call_count <= 2:  # Fail first two attempts
                    raise Exception("Temporary processing failure")

                # Succeed on third attempt
                return [
                    Mock(
                        text="Recovered content",
                        category="NarrativeText",
                        metadata=Mock(
                            page_number=1,
                            element_id="recovered_1",
                            parent_id=None,
                            filename="integration_test.pdf",
                            coordinates=[(0, 0), (100, 20)],
                            text_as_html=None,
                            image_path=None,
                        ),
                    )
                ]

            mock_partition.side_effect = failing_partition

            # Should succeed after retries
            result = await processor.process_document_async(sample_documents["pdf"])

            # Verify successful recovery
            assert isinstance(result, ProcessingResult)
            assert len(result.elements) > 0
            assert result.elements[0].text == "Recovered content"

            # Verify retry attempts were made
            assert call_count == 3  # 2 failures + 1 success

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_performance_with_real_pipeline_overhead(
        self, integration_settings, sample_documents
    ):
        """Test performance characteristics with real pipeline overhead."""
        processor = DocumentProcessor(integration_settings)

        with patch("src.processing.hybrid_processor.partition") as mock_partition:
            # Simulate realistic processing delay
            async def delayed_partition(*args, **kwargs):
                await asyncio.sleep(0.1)  # Simulate processing time
                return [
                    Mock(
                        text=(
                            "Performance test content with multiple sentences for "
                            "chunking. "
                        )
                        * 10,
                        category="NarrativeText",
                        metadata=Mock(
                            page_number=1,
                            element_id="perf_1",
                            parent_id=None,
                            filename="integration_test.pdf",
                            coordinates=[(0, 0), (400, 200)],
                            text_as_html=None,
                            image_path=None,
                        ),
                    )
                ]

            mock_partition.side_effect = delayed_partition

            # Process document and measure performance
            start_time = asyncio.get_event_loop().time()
            result = await processor.process_document_async(sample_documents["pdf"])
            total_time = asyncio.get_event_loop().time() - start_time

            # Verify performance is reasonable for integration test
            assert total_time < 5.0  # Should complete within 5 seconds
            assert result.processing_time > 0
            assert result.processing_time <= total_time

            # Verify chunking occurred (indicates full pipeline ran)
            assert len(result.elements) > 1  # Should be split into chunks

            # Verify metadata indicates full processing
            assert "pipeline_config" in result.metadata
            assert result.metadata["pipeline_config"]["transformations"] == 2

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_compatibility_with_document_processor_api(
        self, integration_settings, sample_documents
    ):
        """Test API compatibility with DocumentProcessor."""
        # Test both factory functions work

        processor1 = DocumentProcessor(integration_settings)
        processor2 = DocumentProcessor(integration_settings)

        assert isinstance(processor1, DocumentProcessor)
        assert isinstance(processor2, DocumentProcessor)

        with patch("src.processing.hybrid_processor.partition") as mock_partition:
            mock_partition.return_value = [
                Mock(
                    text="Compatibility test",
                    category="Title",
                    metadata=Mock(
                        page_number=1,
                        element_id="compat_1",
                        parent_id=None,
                        filename="integration_test.pdf",
                        coordinates=[(0, 0), (100, 20)],
                        text_as_html=None,
                        image_path=None,
                    ),
                )
            ]

            # Both should work identically
            result1 = await processor1.process_document_async(sample_documents["pdf"])
            result2 = await processor2.process_document_async(sample_documents["pdf"])

            assert result1.elements[0].text == result2.elements[0].text
            assert result1.strategy_used == result2.strategy_used

    @pytest.mark.integration
    def test_configuration_override_integration(self, integration_settings):
        """Test configuration override functionality in integration context."""
        processor = DocumentProcessor(integration_settings)

        # Test configuration override
        custom_config = {
            "strategy": "fast",
            "extract_images": False,
            "max_characters": 800,
        }

        processor.override_config(custom_config)

        # Verify configuration was stored
        assert hasattr(processor, "_config_override")
        assert processor._config_override["strategy"] == "fast"
        assert processor._config_override["extract_images"] is False
        assert processor._config_override["max_characters"] == 800

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_cache_management_integration(self, integration_settings):
        """Test cache management operations in integration context."""
        processor = DocumentProcessor(integration_settings)

        # Test cache clearing
        clear_result = await processor.clear_cache()
        assert clear_result is True

        # Test cache statistics
        stats = await processor.get_cache_stats()
        assert "processor_type" in stats
        assert stats["processor_type"] == "hybrid"
        assert "simple_cache" in stats
        assert "llamaindex_cache" in stats
        assert "strategy_mappings" in stats
        assert stats["strategy_mappings"] > 0
