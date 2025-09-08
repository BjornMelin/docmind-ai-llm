from src.config.settings import DocMindSettings


def test_retrieval_env_mapping_defaults():
    s = DocMindSettings()
    assert s.retrieval.fusion_mode in {"rrf", "dbsf"}
    assert isinstance(s.retrieval.fused_top_k, int)
    assert s.retrieval.fused_top_k > 0
    assert isinstance(s.retrieval.rrf_k, int)
    assert s.retrieval.rrf_k > 0


def test_retrieval_mapping_override():
    s = DocMindSettings(
        retrieval={"fusion_mode": "dbsf", "fused_top_k": 99, "rrf_k": 12}
    )
    assert s.retrieval.fusion_mode == "dbsf"
    assert s.retrieval.fused_top_k == 99
    assert s.retrieval.rrf_k == 12
