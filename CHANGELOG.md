# Changelog

All notable changes to this project will be documented in this file.

The format is based on Keep a Changelog and this project adheres to Semantic Versioning.

## 0.1.0 (2026-01-14)


### Features

* add documentation quality gates and improve contributing guidelines ([22daec9](https://github.com/BjornMelin/docmind-ai-llm/commit/22daec9dc71e79cc1d0db363d915fa8f844833b5))
* add ingestion demo and finalize changelog ([e51a152](https://github.com/BjornMelin/docmind-ai-llm/commit/e51a1526917ff7954ac45fc9ed5fa4676867b54b))
* add ingestion models and hashing helpers ([a2b66df](https://github.com/BjornMelin/docmind-ai-llm/commit/a2b66dfced47d47ea5034a3b5bfea938b79c6ee8))
* add OTEL scaffolding with feature flags ([805b74c](https://github.com/BjornMelin/docmind-ai-llm/commit/805b74c56ad51e547875ee258f533c5f0afc7ebf))
* **adrs:** introduce ADR-057 for chat persistence and hybrid memory using LangGraph SQLite ([9f8364c](https://github.com/BjornMelin/docmind-ai-llm/commit/9f8364cdb503912901eab793a8505ea18bfdc78f))
* **agent-tools:** fallback to vector search when hybrid returns no results for primary query ([01a9849](https://github.com/BjornMelin/docmind-ai-llm/commit/01a98494ff8200674fac2d5617dfb0e4fd26fad3))
* **agents:** add timeout fallback analytics and always-on optimization metrics; add helper to build optimization metrics; ensure timeout flag in both branches ([75260e5](https://github.com/BjornMelin/docmind-ai-llm/commit/75260e50211a360a2c2c3f032fbd0c4d328319d7))
* **agents:** coordinator timeout fallback, always-on metrics, streaming tests, ADR updates ([4243409](https://github.com/BjornMelin/docmind-ai-llm/commit/424340935b2848b5b136992e92a9e1887220024c))
* **analytics:** add local analytics manager and analytics page (foundational)\n\n- AnalyticsConfig and AnalyticsManager for local metrics\n- Streamlit Analytics page placeholder\n\nRefs: ADR-032 ([c5c2235](https://github.com/BjornMelin/docmind-ai-llm/commit/c5c2235d96f2fe95f07a2389047a4ee54578a1af))
* **app:** add prompt telemetry logging (template id/name/version)\n\n- log prompt metadata after render in analysis (async + sync)\n- keep logging local via JSONL with sampling/rotation\n\ndocs(spec-020,adr-018): standardize SPEC-020 header/sections; mark ADR-018 Implemented; link developer guide\n\n- SPEC-020 now follows standard YAML header and includes file operations\n- ADR-018 status Implemented with SPEC-020 compatibility note\n- Add developer guide for adding templates; linked from ADR/README\n\ntest(prompting): expand unit coverage (validators, missing-vars) and keep tests stable\n ([b9e3b2c](https://github.com/BjornMelin/docmind-ai-llm/commit/b9e3b2c129cc42ab8dfacc65f807d05bb1eb6367))
* **cache:** add configurable local cache manager ([cd07c33](https://github.com/BjornMelin/docmind-ai-llm/commit/cd07c3354eab0b2d150d8578ef780cdf8adc7c0b))
* **check_links:** enhance link checking script with improved handling and reporting ([8eeba34](https://github.com/BjornMelin/docmind-ai-llm/commit/8eeba344a182faea66834fb136cfda7f004cc968))
* **check_links:** refactor link checking logic and improve markdown file handling ([f2c61ed](https://github.com/BjornMelin/docmind-ai-llm/commit/f2c61ed665d186eb0465a652eaea994699837872))
* **config,docs,ui,tests:** migrate to openai.*, security.*, hybrid.*; enforce /v1 normalization; remove import-time I/O; unify hybrid gating; update UI to read-only policy; update docs and examples; clean .env.example; update tests ([14e266e](https://github.com/BjornMelin/docmind-ai-llm/commit/14e266e371b89f5ef60cdc0c969e658d043cfc0a))
* **config:** add DBSF env toggle and telemetry enabled mapping; add prefetch limits; startup log resolved mode ([8035ade](https://github.com/BjornMelin/docmind-ai-llm/commit/8035adee8240632c3fbf6328a8736d427078f93e))
* **config:** backend-aware OpenAI-like '/v1' normalization in LLM factory ([0f0eb8d](https://github.com/BjornMelin/docmind-ai-llm/commit/0f0eb8d03a8052aad9c5ad74653ae747ce2fffa7))
* **config:** migrate to Pydantic v2 validators/computed fields and remove import-time side effects ([d9ce487](https://github.com/BjornMelin/docmind-ai-llm/commit/d9ce4877bcb7e4c6152db70755c5d0193f28e1a0))
* consolidate reranking+multimodal (device policy, SigLIP adapter, minimal telemetry, docs) ([59cd81e](https://github.com/BjornMelin/docmind-ai-llm/commit/59cd81ebae8674bd1ab2f60855dbec16fb0bc72a))
* enforce Qdrant server-side hybrid (RRF default; DBSF flag) + IDF sparse + dedup + telemetry ([a313660](https://github.com/BjornMelin/docmind-ai-llm/commit/a313660ee0315c95a04b788c2b60bc6d05194f8e))
* enhance CI workflow with Node.js setup and improve link checking regex ([8c29501](https://github.com/BjornMelin/docmind-ai-llm/commit/8c2950119b7d081b7623b815d55c6b5cbe92eae3))
* **env:** implement environment variable persistence helper and integrate with settings ([9c43db3](https://github.com/BjornMelin/docmind-ai-llm/commit/9c43db31ac0cb4d9442a739cc5d091009e42f4f5))
* **eval:** add determinism utilities, canonical sort/round6, and doc_id mapping helpers ([8bcdf3f](https://github.com/BjornMelin/docmind-ai-llm/commit/8bcdf3ff4628f574ef7cb7cdc3fee3785f631e4d))
* **eval:** add JSON Schemas and validator; enforce dynamic header ↔ k consistency for BEIR leaderboards ([6679693](https://github.com/BjornMelin/docmind-ai-llm/commit/66796939a3c484ee95eaf0ed4d41d6cdd27fe716))
* **eval:** deterministic BEIR/RAGAS harness with dynamic [@k](https://github.com/k) and schema validation ([727e327](https://github.com/BjornMelin/docmind-ai-llm/commit/727e32718819651273e9fe65557eb59520cec3b1))
* **eval:** respect --k and emit dynamic @{k} metrics; add schema_version/sample_count; determinism first; doc_mapping.json for reproducibility ([d626b83](https://github.com/BjornMelin/docmind-ai-llm/commit/d626b838f3c974c852d8484a114d01e273a2449e))
* **graphrag:** label-preserving exports and retriever-first seeding; integrate into UI\n\n- export_graph_jsonl preserves relation labels (fallback 'related')\n- get_export_seed_ids prefers graph retriever → vector → deterministic fallback\n- Documents page and ingest adapter use seed helper\n\nRefs: SPEC-006, ADR-038 ([e9a855f](https://github.com/BjornMelin/docmind-ai-llm/commit/e9a855f4a2063d2371998c2936048c947a5fad48))
* **graphrag:** Phase-2 router+persistence\n\n- persistence: add SnapshotManager (atomic tmp-&gt;rename, manifest with corpus/config hashes, dir lock)\n- retrieval: add router_factory.build_router_engine (vector+graph tools, Pydantic selector when available, safe fallback)\n- graph: library-first exports JSONL/Parquet schema (subject/object/depth/path_id), default depth=1\n- ui: Documents toggle 'Build GraphRAG (beta)', snapshot + exports; Chat staleness badge\n- tests: unit (snapshot, router, helpers), integration (router, exports), e2e (chat smoke)\n- docs: CHANGELOG entries for Phase-2 ([9ee59c4](https://github.com/BjornMelin/docmind-ai-llm/commit/9ee59c408853cf4a73eb829878a72e59cc7275ca))
* harden container build and compose ([e95ec62](https://github.com/BjornMelin/docmind-ai-llm/commit/e95ec62ee81dc62a7945602cf67c9e803416fad5))
* harden container build and compose ([4e024bf](https://github.com/BjornMelin/docmind-ai-llm/commit/4e024bf5fd951690d07e538e6acdf0e4bbda3572))
* ingestion and observability overhaul ([6f386f8](https://github.com/BjornMelin/docmind-ai-llm/commit/6f386f835b224ca0c6eb48fbbf94581243b87ecb))
* **ingestion:** add canonical hashing bundle ([a20edb4](https://github.com/BjornMelin/docmind-ai-llm/commit/a20edb4717b78b67427dba84a7fd26deee3b7c5d))
* **ingestion:** enhance docstring clarity across ingestion pipeline functions ([90265f6](https://github.com/BjornMelin/docmind-ai-llm/commit/90265f6fcd13c38a4558bb0272f43d6ce2b8eb42))
* **ingestion:** introduce pipeline builder ([022c266](https://github.com/BjornMelin/docmind-ai-llm/commit/022c2664371b5819121191135cafe1494b7561d4))
* **ingest:** tolerate missing embeddings and expand coverage ([c9abcc7](https://github.com/BjornMelin/docmind-ai-llm/commit/c9abcc7aa6cb8db6b90e451f5400147b160c8956))
* **integrations:** add startup_init and centralize endpoint security policy ([859d7aa](https://github.com/BjornMelin/docmind-ai-llm/commit/859d7aa77bfafec90ac753f98f92d3277a91158f))
* **multimodal:** ship end-to-end pipeline + persistence ([ec0d012](https://github.com/BjornMelin/docmind-ai-llm/commit/ec0d012caaa489eba51ff81ed70180e08475ade2))
* **multimodal:** ship end-to-end pipeline + persistence ([93519b2](https://github.com/BjornMelin/docmind-ai-llm/commit/93519b2a629b19db7d203951d34c64c0699c577d))
* **observability:** add OpenTelemetry metrics documentation and demo script ([32092ec](https://github.com/BjornMelin/docmind-ai-llm/commit/32092ec744d2182e8271a00e85b5238d69a3b63f))
* **observability:** centralize otel configuration and instrumentation ([b07f18a](https://github.com/BjornMelin/docmind-ai-llm/commit/b07f18a8d3e5baf37ddb5cf9c130e17bea71bd64))
* **observability:** normalize OTLP http endpoints and correlate logs ([901234b](https://github.com/BjornMelin/docmind-ai-llm/commit/901234bce9ea80eced29a4716fb8106749f29722))
* **observability:** update dependencies and enhance documentation ([bc4e8a2](https://github.com/BjornMelin/docmind-ai-llm/commit/bc4e8a26ea54533b5feb2800cbe36b6bebb5d290))
* **ocr:** add heuristic controller ([d9e88f8](https://github.com/BjornMelin/docmind-ai-llm/commit/d9e88f860ec3673e56b93d8139ca19c9bc9724ae))
* **persistence:** enrich snapshot manifest and relpath hashing; Chat autoload policy\n\n- Manifest adds schema_version, persist_format_version, versions map\n- compute_corpus_hash uses POSIX relpaths (base_dir)\n- Chat autoloads latest non-stale snapshot; staleness badge\n\nRefs: SPEC-014, ADR-038 ([84d1267](https://github.com/BjornMelin/docmind-ai-llm/commit/84d1267f268b005a6574ba4308a7728ff48c2974))
* **persistence:** refresh snapshot locking and manifest telemetry ([3aefcc0](https://github.com/BjornMelin/docmind-ai-llm/commit/3aefcc0166d62c0bcf104a5a0be43292a35a8d42))
* **prompting:** add pure build_prompt_context helper and tests ([0d0e51a](https://github.com/BjornMelin/docmind-ai-llm/commit/0d0e51a72cc2f714f81aa961917fa2123f4f3177))
* **prompts:** add multiple implementation prompts for containerization hardening, keyword tool, legacy entrypoint removal, multimodal ingestion, hybrid retrieval, and UI persistence ([8f14b98](https://github.com/BjornMelin/docmind-ai-llm/commit/8f14b986ab765a550f58bb69de740de1c14bdfa6))
* **prompts:** add multiple implemented prompts for system enhancements ([9f8364c](https://github.com/BjornMelin/docmind-ai-llm/commit/9f8364cdb503912901eab793a8505ea18bfdc78f))
* remove legacy main entrypoint ([f458fb5](https://github.com/BjornMelin/docmind-ai-llm/commit/f458fb591675262c3772031e45895433fa920f6a))
* remove legacy main entrypoint ([71e1112](https://github.com/BjornMelin/docmind-ai-llm/commit/71e1112dd5db9341c93fa9131ae14d7029bcceaf))
* **retrieval,config:** wire flags + executor; unify SigLIP adapter ([7ad7d79](https://github.com/BjornMelin/docmind-ai-llm/commit/7ad7d79024ce22bb4532ffc15fa21ed9f9e1114a))
* **retrieval,router:** apply reranking via node_postprocessors in RouterFactory and ToolFactory ([6796e89](https://github.com/BjornMelin/docmind-ai-llm/commit/6796e8944026f23c6a6a820ba3867d2d7e0f02e5))
* **retrieval,router:** apply reranking via node_postprocessors in RouterFactory and ToolFactory ([a594faf](https://github.com/BjornMelin/docmind-ai-llm/commit/a594fafec3efd00ba3cc299821b20b2fceb4b116))
* **retrieval:** add ColPali/SigLIP settings; keep pylint line-length compliance in embeddings/core ([9aaf883](https://github.com/BjornMelin/docmind-ai-llm/commit/9aaf883a77e607cdc69466cfe4600ae058da1f10))
* **retrieval:** add server-side hybrid retriever and integrate hybrid tool in router\n\n- New ServerHybridRetriever (Qdrant Query API Prefetch + FusionQuery)\n- Router factory registers hybrid_search tool and selector preference\n- Expose fusion_mode and dedup_key via settings; Qdrant collection usage\n\nRefs: ADR-003, SPEC-004 ([aea07d0](https://github.com/BjornMelin/docmind-ai-llm/commit/aea07d00564674a16a198498f5cfdca7249ce18b))
* **retrieval:** add sparse-only keyword_search tool ([49d5e5b](https://github.com/BjornMelin/docmind-ai-llm/commit/49d5e5b15a94630cbfd2258fa8116e487969d1f0))
* **retrieval:** add sparse-only keyword_search tool ([8da48b0](https://github.com/BjornMelin/docmind-ai-llm/commit/8da48b008ed4aad796cb79554102dd0873419e59))
* **retrieval:** add telemetry and headroom limit; deterministic dedup by page_id before truncation ([b970646](https://github.com/BjornMelin/docmind-ai-llm/commit/b970646f85f397b00e335ed48abaebca20af7c5c))
* **retrieval:** harden optional llama-index integration ([7a27e32](https://github.com/BjornMelin/docmind-ai-llm/commit/7a27e32ed2360a874eb8753195fa6b31304dc1e1))
* **retrieval:** unify server-side hybrid gating to single flag per ADR-024 ([98fd826](https://github.com/BjornMelin/docmind-ai-llm/commit/98fd826cb4ada6f10395c630af6097d905199325))
* **router,graphrag,persistence,ui,docs,tests:** server-side hybrid + router unification + GraphRAG persistence/exports + multipage UI ([7ff92cf](https://github.com/BjornMelin/docmind-ai-llm/commit/7ff92cfb396e258c2333a87fb7ae89056e6ebbb5))
* **router:** wire settings-driven prefetch limits and fusion mode into hybrid retriever params ([296adf1](https://github.com/BjornMelin/docmind-ai-llm/commit/296adf17c83f09fd26f066e8e1fc6774016d3ea9))
* **settings:** harden endpoint allowlist validation and add cache_version salt ([a19b7cd](https://github.com/BjornMelin/docmind-ai-llm/commit/a19b7cda92bdd6c67ed3152f4d36b71da1b750e5))
* **storage:** ensure Qdrant hybrid collection with named vectors and IDF (idempotent) ([18baa40](https://github.com/BjornMelin/docmind-ai-llm/commit/18baa402f74cf43b76266a74624b016bbef00840))
* **tests:** add unit tests for router tool and prompting renderer ([a161f7e](https://github.com/BjornMelin/docmind-ai-llm/commit/a161f7ea698f6b3ef2ec4623e4e680333dd24f7a))
* **tests:** enhance testing infrastructure for ingestion and observability ([b7bd426](https://github.com/BjornMelin/docmind-ai-llm/commit/b7bd42621aebedf45437b2ae1649ea4729ecbfc9))
* **tools,eval:** add BEIR/RAGAS eval CLIs and model pull CLI; add eval README\n\n- tools/eval: run_beir.py, run_ragas.py\n- tools/models: pull.py\n- data/eval: README with usage instructions\n\nRefs: SPEC-010 (ADR-039) ([caea901](https://github.com/BjornMelin/docmind-ai-llm/commit/caea9013e2c45bebf405a3fb2e695c265fb42bfe))
* **ui,docs:** GraphRAG Phase‑2 polish ([f2fd18f](https://github.com/BjornMelin/docmind-ai-llm/commit/f2fd18f0bb6fc292d93448e7af03bfa6f0a0fae5))
* **ui:** add Clear caches helper and Settings control with tests ([2c5bd4c](https://github.com/BjornMelin/docmind-ai-llm/commit/2c5bd4ceff635ddf781f6d6a824ee2ca6c568397))
* **ui:** integrate ingestion pipeline and telemetry ([3bcc3d7](https://github.com/BjornMelin/docmind-ai-llm/commit/3bcc3d7944a2e3c29d4fb798c06002e3fea6a035))
* **ui:** show resolved normalized backend base URL and remove hybrid toggle ([803598c](https://github.com/BjornMelin/docmind-ai-llm/commit/803598c96395fb656b128af239fc5d0e8a784e45))
* update pre-commit hook versions and add markdownlint for documentation. ([688e8df](https://github.com/BjornMelin/docmind-ai-llm/commit/688e8dfce2b3e64dbcb7bbd71e1f6bb596d47bcb))
* **utils:** add hashing and time helpers ([5f5d6e7](https://github.com/BjornMelin/docmind-ai-llm/commit/5f5d6e79d10cd1788f286444b2c479d3f39c8536))
* wire ingestion pipeline around LlamaIndex ([b4b2e33](https://github.com/BjornMelin/docmind-ai-llm/commit/b4b2e3325520c884211bfc567e6711918c446bb2))


### Bug Fixes

* address chat_sessions.py review comments ([78d8b5b](https://github.com/BjornMelin/docmind-ai-llm/commit/78d8b5b9932523d19e446baee52bc303d1c73a3e))
* address containerization review comments ([b0961f8](https://github.com/BjornMelin/docmind-ai-llm/commit/b0961f84907aba1bdf947eb0160e309b5017a847))
* address coordinator.py review comments ([678bf48](https://github.com/BjornMelin/docmind-ai-llm/commit/678bf48fbe71aceff58972285470ba88b51fd8d6))
* address follow-up PR60 review comments ([73a54c0](https://github.com/BjornMelin/docmind-ai-llm/commit/73a54c025024fd728b8b9ccff8026bad4ba392e9))
* address keyword tool review nits ([b930922](https://github.com/BjornMelin/docmind-ai-llm/commit/b93092216a95d0b2112e9ebe3ebd851447d9c780))
* address PR 60 review comments for router_factory and chat pages ([357ae6b](https://github.com/BjornMelin/docmind-ai-llm/commit/357ae6bd1da92c787d3664d81750afebe2436f89))
* address PR review comments ([d270d25](https://github.com/BjornMelin/docmind-ai-llm/commit/d270d25b2172cb1c21bcf2c8852698e904145fb1))
* address pr review feedback ([912579d](https://github.com/BjornMelin/docmind-ai-llm/commit/912579d07dda3bafbe800a75a83a4f7e5658d90c))
* address PR review feedback ([7d8b8cc](https://github.com/BjornMelin/docmind-ai-llm/commit/7d8b8ccbf1dad648932fd79d9b7c3bcd18231ef1))
* address PR60 review comments ([5bc1a8d](https://github.com/BjornMelin/docmind-ai-llm/commit/5bc1a8d8d238ebbe8f1d522091d75b6881397623))
* address remaining PR [#60](https://github.com/BjornMelin/docmind-ai-llm/issues/60) review comment threads (8 threads) ([e9f1337](https://github.com/BjornMelin/docmind-ai-llm/commit/e9f13375bae7879fb391c7c80ccfa465c32b0ff7))
* address review comments in memory.py, models.py, snapshot_service.py ([6c301de](https://github.com/BjornMelin/docmind-ai-llm/commit/6c301de53259e8a4e6e0b10769add8954686f95a))
* address review feedback ([e071583](https://github.com/BjornMelin/docmind-ai-llm/commit/e071583bbc78df0609ccd0465bd6479b72f04519))
* **agents:** avoid fallback query double counting ([927ccb8](https://github.com/BjornMelin/docmind-ai-llm/commit/927ccb89234b00d68de1d5772ba321e4ab74af2d))
* **agents:** correct success metrics and analytics on timeouts\n\n- Do not increment successful_queries when workflow times out\n- Increment fallback_queries only when fallback path used\n- Avoid double analytics by skipping success log on timeout ([863f4ea](https://github.com/BjornMelin/docmind-ai-llm/commit/863f4eabbd0be8073b91a8f2080cbcb33928dd93))
* align env samples and ingestion fallbacks ([0f515ea](https://github.com/BjornMelin/docmind-ai-llm/commit/0f515ea791ebe523ddd689f8cbca09d6d2f62e07))
* **analytics:** address review feedback ([07b8d7f](https://github.com/BjornMelin/docmind-ai-llm/commit/07b8d7f68c49a268d1e577bdc17f1506d193fe5e))
* **analytics:** harden telemetry parsing and db lifecycle ([0c92fb2](https://github.com/BjornMelin/docmind-ai-llm/commit/0c92fb26447d2e80e9cc90df835d4fe95afa226e))
* **analytics:** harden telemetry parsing and db lifecycle ([e6bf5c8](https://github.com/BjornMelin/docmind-ai-llm/commit/e6bf5c8c897d15d15fb4e33edc03f7dc078c0045))
* **ci,retrieval:** align CI profile and GraphRAG extras ([7f9bc05](https://github.com/BjornMelin/docmind-ai-llm/commit/7f9bc05f062190db665b9a0794d14a4cb82ad1bd))
* **ci:** address container-static review feedback ([86f6585](https://github.com/BjornMelin/docmind-ai-llm/commit/86f6585a33d6ac7cdf8d7302bbe6705321a19afb))
* **ci:** handle dockerfile parser casing ([327f354](https://github.com/BjornMelin/docmind-ai-llm/commit/327f3543486ce720d04791c615cbb882d3156bbc))
* **ci:** use dependency groups for tests ([6d64c29](https://github.com/BjornMelin/docmind-ai-llm/commit/6d64c2949f3d2b263695a2dc4eb499c8ddc8173b))
* **ci:** validate ENTRYPOINT embedded shell ([d822c55](https://github.com/BjornMelin/docmind-ai-llm/commit/d822c55696eef970c110c663b244b724ab556489))
* clean rerank telemetry and normalization tests ([76564e4](https://github.com/BjornMelin/docmind-ai-llm/commit/76564e4d5067987fc7f92b26658a3bef57275cbe))
* **config:** correct URL validation logic in DocMindSettings ([2bee1f7](https://github.com/BjornMelin/docmind-ai-llm/commit/2bee1f7dda3482f41e8fef44ea6e9ce1c10f3c54))
* **config:** honor security trust_remote_code for embeddings ([c68fae0](https://github.com/BjornMelin/docmind-ai-llm/commit/c68fae0e00e8845fa1c7e0e7f8e0b4bec3219bb8))
* **coordinator:** correct fallback metadata and improve error handling ([4dd98f8](https://github.com/BjornMelin/docmind-ai-llm/commit/4dd98f8936d65d4b995e87220b0d4cc720d2af4a))
* **coordinator:** use output_mode="last_message" and add_handoff_messages=True; remove structured mode (Workstream B)\n\nRefs: agent-logs/2025-09-09/research/agents-coordinator/012-final-implementation-plan.md §Workstream B ([cca6093](https://github.com/BjornMelin/docmind-ai-llm/commit/cca6093615b9ffdabe337862151a5283b4dc7f7f))
* **coverage:** simplify HTML report not found message in check_coverage.py ([af9b6f3](https://github.com/BjornMelin/docmind-ai-llm/commit/af9b6f3d795b6e1beee35ba82cd5bae118889348))
* **cuda:** correct return type for _check_cuda_compatibility function ([6561f7f](https://github.com/BjornMelin/docmind-ai-llm/commit/6561f7f4a844fdcf944fbe0f77f94a6bd570f8ff))
* **dependencies:** update llama-index-llms-openai version to 0.6.13 ([9d15148](https://github.com/BjornMelin/docmind-ai-llm/commit/9d1514891fc44ea3ff242f45c1d2553409116057))
* **docs:** correct table formatting and remove unnecessary whitespace in API documentation ([2e4063d](https://github.com/BjornMelin/docmind-ai-llm/commit/2e4063df4ae6b6304663122e5c9ed690e29084e5))
* **document:** improve document loading type hints and cache stats ([4dd98f8](https://github.com/BjornMelin/docmind-ai-llm/commit/4dd98f8936d65d4b995e87220b0d4cc720d2af4a))
* **documents:** enhance error handling and path validation ([4dd98f8](https://github.com/BjornMelin/docmind-ai-llm/commit/4dd98f8936d65d4b995e87220b0d4cc720d2af4a))
* enforce strong hashing secret defaults ([0d741fb](https://github.com/BjornMelin/docmind-ai-llm/commit/0d741fb68fb848750e55fafe228f475980a97665))
* **eval:** address PR review items (contexts duplication; BEIR custom metric fallback; qrels subset; robust path filtering; schema/doc_mapping alignment; error messages) ([526457f](https://github.com/BjornMelin/docmind-ai-llm/commit/526457fbffcceb8c8a41c05bb1f75c8b8a70cd77))
* **eval:** correct import path for ServerHybridRetriever in BEIR runner\n\n- Import from src.retrieval.hybrid instead of removed query_engine path ([6b564eb](https://github.com/BjornMelin/docmind-ai-llm/commit/6b564ebbe074dbe6d48a416a594e997c0d19c69a))
* **eval:** update evaluation metrics to use dynamic k values\n\n- Adjusted the evaluation functions to accept a dynamic k value for metrics calculation.\n- Updated CSV leaderboard output to reflect the chosen k in the metric column names. ([a953ed1](https://github.com/BjornMelin/docmind-ai-llm/commit/a953ed1a6e92d62e0d70023a92233702504eea2e))
* **eval:** validate --sample_count argument to ensure it is non-negative ([620a67d](https://github.com/BjornMelin/docmind-ai-llm/commit/620a67d54175471f1fad75f1376fbb775d835086))
* **graph_config:** restore JSONL rendering for graph export ([4dd98f8](https://github.com/BjornMelin/docmind-ai-llm/commit/4dd98f8936d65d4b995e87220b0d4cc720d2af4a))
* harden snapshot metadata and image exports ([99f8559](https://github.com/BjornMelin/docmind-ai-llm/commit/99f8559e86aea89e98bc44305bbd99a5df5779d4))
* **hybrid:** ensure upgrade safety by invoking ensure_hybrid_collection in retriever init; include AttributeError in guard for test fakes ([373c425](https://github.com/BjornMelin/docmind-ai-llm/commit/373c42509ce5b334ac92e5b48844770ed4d582bf))
* **image_index:** add dimension validation for image collections ([4dd98f8](https://github.com/BjornMelin/docmind-ai-llm/commit/4dd98f8936d65d4b995e87220b0d4cc720d2af4a))
* **images:** refine thumbnail creation logic ([4dd98f8](https://github.com/BjornMelin/docmind-ai-llm/commit/4dd98f8936d65d4b995e87220b0d4cc720d2af4a))
* **ingestion:** allow missing embed model fallback ([9dd7998](https://github.com/BjornMelin/docmind-ai-llm/commit/9dd7998f7c6747ca25697cc66021a38a4aefce2f))
* **keyword:** improve exception handling for qdrant-client compatibility ([903b3c7](https://github.com/BjornMelin/docmind-ai-llm/commit/903b3c7a1115b04d25bbd82aae5bbb0dbcb46e4e))
* make document loading async-safe ([a619479](https://github.com/BjornMelin/docmind-ai-llm/commit/a61947998ad29b044412e7e187e7361fc964720b))
* **memory:** add error handling and logging for memory operations ([fd920e9](https://github.com/BjornMelin/docmind-ai-llm/commit/fd920e98410adeb98de7ac4711dd182b40554406))
* **memory:** add error handling for memory deletion ([4dd98f8](https://github.com/BjornMelin/docmind-ai-llm/commit/4dd98f8936d65d4b995e87220b0d4cc720d2af4a))
* **memory:** align tool errors and cleanup sqlite timing ([a3210f5](https://github.com/BjornMelin/docmind-ai-llm/commit/a3210f56ef8ddf81be74f2fadcada50d7c2dc201))
* **multimodal_fusion:** correct logging format in MultimodalFusionRetriever ([fd920e9](https://github.com/BjornMelin/docmind-ai-llm/commit/fd920e98410adeb98de7ac4711dd182b40554406))
* **multimodal_fusion:** set image query timeout from settings ([4dd98f8](https://github.com/BjornMelin/docmind-ai-llm/commit/4dd98f8936d65d4b995e87220b0d4cc720d2af4a))
* **multimodal:** harden indexing, caching, and exports ([c1b77fb](https://github.com/BjornMelin/docmind-ai-llm/commit/c1b77fb78b2a3259b2425a4341794e738224af32))
* **multimodal:** harden router tool and artifacts ([046e9d7](https://github.com/BjornMelin/docmind-ai-llm/commit/046e9d75318d305cc808e7ee9f1448cb62cb0d82))
* **pages:** derive GraphRAG export seeds from retrievers with deterministic fallback; remove placeholder-based seeding\n\n- Use get_export_seed_ids(pg_index, vector_index, cap) on Documents page\n- Respect settings.graphrag_cfg.export_seed_cap for cap\n- Update tests to assert fallback determinism via graph_config ([03e3b44](https://github.com/BjornMelin/docmind-ai-llm/commit/03e3b440df61032814e9ba73bf8c1f411b2b748a))
* **persistence:** declare sqlite store TTL support ([74cdb30](https://github.com/BjornMelin/docmind-ai-llm/commit/74cdb301db39569503e49cbc7403748b063559b2))
* **persistence:** harden snapshot finalize/cleanup and config hash; test(ui): stabilize Documents snapshot AppTest; test(ui): load pages via importlib; test(retrieval): avoid import-order coupling in selector fallback; test(storage): stub QdrantVectorStore for coverage ([277bb4b](https://github.com/BjornMelin/docmind-ai-llm/commit/277bb4bac3f12b4db488e096d82ca3feaf962c8b))
* **postprocessor_utils:** correct retriever query engine construction ([4dd98f8](https://github.com/BjornMelin/docmind-ai-llm/commit/4dd98f8936d65d4b995e87220b0d4cc720d2af4a))
* **pr-60:** address review feedback ([d9ddfb6](https://github.com/BjornMelin/docmind-ai-llm/commit/d9ddfb69a6080c54838fd63b98ff779b763179c7))
* **pr-review:** address CodeRabbit threads ([44da819](https://github.com/BjornMelin/docmind-ai-llm/commit/44da819554a16b57609cdb532e644dec4f9e2cd1))
* **pr-review:** address container checks and torch cache ([3b6d0b8](https://github.com/BjornMelin/docmind-ai-llm/commit/3b6d0b8263f91c1169b902190f087aa00af8078a))
* **pr-review:** address remaining CI and compose notes ([e392c99](https://github.com/BjornMelin/docmind-ai-llm/commit/e392c99fb62b820a87ad5f615937d512e7794d26))
* **pr60:** address CodeRabbit review comments ([e803c39](https://github.com/BjornMelin/docmind-ai-llm/commit/e803c396fee94f127dde8dddf3719783e2187e4a)), closes [#60](https://github.com/BjornMelin/docmind-ai-llm/issues/60)
* **pr:** address review comments for multimodal cleanup ([696f3d1](https://github.com/BjornMelin/docmind-ai-llm/commit/696f3d1fb7eda73000650bcbf64380b35bff4378))
* **prompts:** update release readiness and service boundary prompts ([9f8364c](https://github.com/BjornMelin/docmind-ai-llm/commit/9f8364cdb503912901eab793a8505ea18bfdc78f))
* provide unstructured reader with source file ([80e751e](https://github.com/BjornMelin/docmind-ai-llm/commit/80e751e08be22aed59e4000cfe49046d6dbfa21e))
* **pr:** resolve review feedback ([d470b3f](https://github.com/BjornMelin/docmind-ai-llm/commit/d470b3fe2d954bee7655ef0355cb27bd0776ed26))
* **pylint:** narrow exception handlers, wrap long lines, and suppress complexity in router; cleanups across retrieval and utils ([dde3e32](https://github.com/BjornMelin/docmind-ai-llm/commit/dde3e3279488d31314c1a226c2caabed9ae66c0d))
* **pylint:** narrow exception handlers, wrap long lines, and suppress Streamlit page module-name; mark unused param ([47f1f4b](https://github.com/BjornMelin/docmind-ai-llm/commit/47f1f4be399778b7aca993db269c58a82c46e0c0))
* **pytest:** ensure consistent imports when running from subpaths ([de18b14](https://github.com/BjornMelin/docmind-ai-llm/commit/de18b14743511a19d9c7078a9367e2ede03b53a7))
* **reranking:** improve node ID retrieval and score assignment ([6129c53](https://github.com/BjornMelin/docmind-ai-llm/commit/6129c53afa969a6a32baacb4ca9b19b4f9adb2f4))
* **reranking:** improve score handling and stability in reranking logic ([78d7adc](https://github.com/BjornMelin/docmind-ai-llm/commit/78d7adcfac5c7f0f4e11ff082aa584e587907ab3))
* Resolve 24 unresolved PR review comments (PR [#60](https://github.com/BjornMelin/docmind-ai-llm/issues/60)) ([f5e6974](https://github.com/BjornMelin/docmind-ai-llm/commit/f5e6974758712db17c83f86407a995782762cd1f))
* resolve PR [#60](https://github.com/BjornMelin/docmind-ai-llm/issues/60) review threads - batch 1 ([83fd61d](https://github.com/BjornMelin/docmind-ai-llm/commit/83fd61d516edf84fa06aa3da6be705476f3fab5e))
* resolve PR review comments ([c4d6bb7](https://github.com/BjornMelin/docmind-ai-llm/commit/c4d6bb79e7c2fbcfccaa971b9c65a573d397f27d))
* resolve PR review comments ([6ae999e](https://github.com/BjornMelin/docmind-ai-llm/commit/6ae999e820e6b96db0d1ec587cba050ae38222cd))
* resolve review comments ([0ec979d](https://github.com/BjornMelin/docmind-ai-llm/commit/0ec979d260cf22b2f4f5cf16e987c255f8f67386))
* resolve review comments ([2b1a56d](https://github.com/BjornMelin/docmind-ai-llm/commit/2b1a56da55d2d8647382deefce0f165b2d2e8b41))
* resolve review feedback for logs and tests ([e610fa7](https://github.com/BjornMelin/docmind-ai-llm/commit/e610fa7687dc1cc7fe09bcf64f188c461a7edc9d))
* restore app test seams and coordinator compliance ([95a3ff7](https://github.com/BjornMelin/docmind-ai-llm/commit/95a3ff704a368caf0a408d3f95135a104b2f07b6))
* restore observability dependency in uv.lock ([dcd05c6](https://github.com/BjornMelin/docmind-ai-llm/commit/dcd05c603ec1cccaffbb4943e132098ddcde46b2))
* restore router tool state lookup and rrf compatibility ([d1f4f38](https://github.com/BjornMelin/docmind-ai-llm/commit/d1f4f383e43e827d9f605285442a4d1a1436d4b9))
* restore shared LLM retries and stabilize helpers ([c4885ac](https://github.com/BjornMelin/docmind-ai-llm/commit/c4885ac8891310a857c5a81efa79b0c02c9ebf21))
* **retrieval:** avoid unbound postprocessor getter ([757c304](https://github.com/BjornMelin/docmind-ai-llm/commit/757c3046b39a90b3b8a3d72aadb36ae4304c645f))
* **retrieval:** expose ServerHybridRetriever for hybrid tool monkeypatching\n\n- Export ServerHybridRetriever at module scope in router_factory\n- Prefer module-level retriever/params with lazy import fallback\n- Unblocks unit tests expecting rf.ServerHybridRetriever\n\nfix(eval): defer optional deps for CLI imports (ragas, beir)\n\n- Wrap ragas/BEIR imports with placeholders to avoid import-time failures\n- Add on-demand imports inside main() for real runs\n- Improves testability and robustness when optional deps are absent\n\ntests: all passing (1020), coverage 76.3% ([0c620c2](https://github.com/BjornMelin/docmind-ai-llm/commit/0c620c20a9971365ca2815d09b1c7dac54ef379e))
* **retrieval:** guard KG tool creation and embedding access\n\n- Append knowledge_graph tool only when a query engine is available\n- Validate Settings.embed_model in ServerHybridRetriever and raise clear error\n\nrefactor(api): remove private _HybridParams from public exports\n\n- Keep internal usage but avoid exporting private dataclass in __all__\n\ndocs(config): clarify patching comment in integrations; router alias comment\n\n- Rephrase comments to avoid test framing in production code\n- Mark RetrieverQueryEngine alias as an internal compatibility shim\n\nci: all tests pass (1020), coverage ~76.3%, pylint &gt;=9.5, ruff clean ([1b914b9](https://github.com/BjornMelin/docmind-ai-llm/commit/1b914b906dd13aaebce43644f1dd2e254d3ee7a3))
* **retrieval:** guard node_postprocessors in query engine builders ([277d560](https://github.com/BjornMelin/docmind-ai-llm/commit/277d560a0d75b6adbf810fc9a058419685bd896d))
* **retrieval:** improve document retrieval and error logging ([4dd98f8](https://github.com/BjornMelin/docmind-ai-llm/commit/4dd98f8936d65d4b995e87220b0d4cc720d2af4a))
* **retrieval:** improve error handling in sparse encoder initialization\n\n- Added fallback logic to return None if both preferred and fallback models fail to initialize, allowing callers to handle sparse encoding gracefully.\n- Updated comments for clarity on the error handling process. ([b33118d](https://github.com/BjornMelin/docmind-ai-llm/commit/b33118d2d6c2596479d5fa5b325dad70ca49acb1))
* **retrieval:** refactor imports and enhance lazy loading in __init__.py ([fd920e9](https://github.com/BjornMelin/docmind-ai-llm/commit/fd920e98410adeb98de7ac4711dd182b40554406))
* **retrieval:** tighten routing and contextual detection ([465e7ab](https://github.com/BjornMelin/docmind-ai-llm/commit/465e7abd915fe1725b8d96db68dbb6b69f301d83))
* reuse _sha256_matches in docker_fetch_torch_wheel.py ([39f29a7](https://github.com/BjornMelin/docmind-ai-llm/commit/39f29a74db0da70afcadb6f297752aefe59c9972))
* **review:** address PR60 review threads ([c3d0ca3](https://github.com/BjornMelin/docmind-ai-llm/commit/c3d0ca3643621cc4da8432ca320aeedc058f81b1))
* **router_factory:** improve error handling for graph tools ([4dd98f8](https://github.com/BjornMelin/docmind-ai-llm/commit/4dd98f8936d65d4b995e87220b0d4cc720d2af4a))
* **router_factory:** improve readability of knowledge graph tool messages ([1f9362d](https://github.com/BjornMelin/docmind-ai-llm/commit/1f9362db3355269158dc9ad8a969b84144946f6f))
* **router_tool:** improve error handling and logging ([4dd98f8](https://github.com/BjornMelin/docmind-ai-llm/commit/4dd98f8936d65d4b995e87220b0d4cc720d2af4a))
* **router_tool:** improve type hints and exception handling ([96ca5c5](https://github.com/BjornMelin/docmind-ai-llm/commit/96ca5c5d49a8834c569f36327a8e1afd285357ba))
* **router,integration:** avoid unintended LLM resolution; add NoOpLLM and safe selector/RouterQueryEngine fallbacks; prevent Settings.llm property access during embed-only setup [autofix] ([713eebf](https://github.com/BjornMelin/docmind-ai-llm/commit/713eebf7199f5489a0ff580a60d223e0207da7ca))
* **router:** correct comments and exception handling in build_router_engine function ([68ec313](https://github.com/BjornMelin/docmind-ai-llm/commit/68ec3139f118db8b120bf3ebbbf44dad75ab7eb6))
* **scripts:** resolve PR 64 review comments in check_links.py ([a3ae2e3](https://github.com/BjornMelin/docmind-ai-llm/commit/a3ae2e358bede1b7362db896968f749269239442))
* **settings:** address PR review feedback ([8f1ce9c](https://github.com/BjornMelin/docmind-ai-llm/commit/8f1ce9c5540c62112897646f57888b3ff559cdf0))
* **settings:** address review feedback and harden GGUF path ([0814312](https://github.com/BjornMelin/docmind-ai-llm/commit/0814312aa41778090bc84ab5e4325a8d68d97b89))
* **settings:** apply validated runtime via model_copy ([1316f33](https://github.com/BjornMelin/docmind-ai-llm/commit/1316f33e66f6891fefc9b0d223cb3c0791b325b6))
* **settings:** harden GGUF path and persistence ([e1b104f](https://github.com/BjornMelin/docmind-ai-llm/commit/e1b104fef9e4e227078abaf5b718d6384b0ee2d8))
* **settings:** harden gguf path validation ([5959cd3](https://github.com/BjornMelin/docmind-ai-llm/commit/5959cd3ce766b12fb2b7d7038aa4367983654e72))
* **settings:** harden settings tests and env persistence ([1121350](https://github.com/BjornMelin/docmind-ai-llm/commit/11213501112a3933e39ee4d0e6aa9229f0d48406))
* **settings:** harden settings UI validation and env persistence ([082ada1](https://github.com/BjornMelin/docmind-ai-llm/commit/082ada141175bb8d7039e6b58c0dd2b38d385d60))
* **settings:** harden settings UI validation and env persistence ([d6d17c9](https://github.com/BjornMelin/docmind-ai-llm/commit/d6d17c98502dc39d69eec49b1cd101c5782d5ce4))
* **settings:** harden settings validation and tests ([38f0b9b](https://github.com/BjornMelin/docmind-ai-llm/commit/38f0b9b3bff69a9ef0ba2f48f86e9d11a8c788ea))
* **settings:** install integration stub after initial render to avoid import-time issues ([eb5fa43](https://github.com/BjornMelin/docmind-ai-llm/commit/eb5fa43194fb34669535af3c25a6bfb0d25ad97a))
* **settings:** reference hashing ADR and env var ([ad4b9b5](https://github.com/BjornMelin/docmind-ai-llm/commit/ad4b9b514dc8e74c41e7bbff40a4deda79d8d28b))
* **settings:** update in-place settings application for runtime validation ([903b3c7](https://github.com/BjornMelin/docmind-ai-llm/commit/903b3c7a1115b04d25bbd82aae5bbb0dbcb46e4e))
* **settings:** validate GGUF path input ([397a5da](https://github.com/BjornMelin/docmind-ai-llm/commit/397a5da1979484e6810e29a0ec57074f122aca13))
* **siglip_adapter:** add error handling for image and text feature extraction ([4dd98f8](https://github.com/BjornMelin/docmind-ai-llm/commit/4dd98f8936d65d4b995e87220b0d4cc720d2af4a))
* simplify condition for image indexing and ensure upsert waits ([f11a0ea](https://github.com/BjornMelin/docmind-ai-llm/commit/f11a0ea290fdafc2cb415876cf1e7ec13080d52e))
* **snapshot_utils:** add logging for export path collision handling ([8533fa9](https://github.com/BjornMelin/docmind-ai-llm/commit/8533fa9aa76b0e5b6a1825e94135dd531d9e2f2e))
* **snapshot.py:** optimize generator usage and fix path traversal vulnerability ([aba5b5a](https://github.com/BjornMelin/docmind-ai-llm/commit/aba5b5a391885f8f5665beec47621eb816bf6df9))
* **snapshot:** release lock on workspace init failure ([aa0aeca](https://github.com/BjornMelin/docmind-ai-llm/commit/aa0aeca6e522eedd44a1fe303ab7af1a56739ef5))
* **storage:** guard optional torch import and CUDA usage for CPU-only environments ([c01e6a3](https://github.com/BjornMelin/docmind-ai-llm/commit/c01e6a393b5ad1fddf4e1a5b41c7a4dd3eb2812c))
* **telemetry:** clarify base-dir warnings ([9d05b88](https://github.com/BjornMelin/docmind-ai-llm/commit/9d05b88c78e9b6bbfe1a2fb9e9c62e651a373abb))
* **telemetry:** clarify override rejection reason ([9989802](https://github.com/BjornMelin/docmind-ai-llm/commit/99898027996a6e6d1a0e0230b732019716d77421))
* **telemetry:** default to settings.data_dir ([f80c9cf](https://github.com/BjornMelin/docmind-ai-llm/commit/f80c9cfb1f35e26ecd5fab71fd42ed99a00a9621))
* **telemetry:** enhance telemetry sanitization logic ([4dd98f8](https://github.com/BjornMelin/docmind-ai-llm/commit/4dd98f8936d65d4b995e87220b0d4cc720d2af4a))
* **telemetry:** tighten path and retry guards ([11602b7](https://github.com/BjornMelin/docmind-ai-llm/commit/11602b77b1fac2853591ce7b1a74dfb25dedd692))
* **tests:** increase timeout and budget for Settings AppTest ([f4a91d6](https://github.com/BjornMelin/docmind-ai-llm/commit/f4a91d635bb7999aa296f12923e0d2b0cacffa58))
* **tests:** update budget for Settings AppTest timeout ([05d600b](https://github.com/BjornMelin/docmind-ai-llm/commit/05d600b8b5c400194ea876e8595e7189fd8d9a0a))
* **tests:** update mock variable names for consistency in session persistence test ([b0f8cfd](https://github.com/BjornMelin/docmind-ai-llm/commit/b0f8cfd2179f95d8a8f16de8be7091fe7ccfaf53))
* update HybridParams import in multimodal_fusion and test files ([80899e9](https://github.com/BjornMelin/docmind-ai-llm/commit/80899e98870bfdbe920887649e1f0a87b6d29c01))
* **verify_structural_parity:** improve error handling for missing src directory ([ccf7550](https://github.com/BjornMelin/docmind-ai-llm/commit/ccf7550c98ca4a1297cabf3c5c400c7ec8958273))


### Documentation

* add developer guide for adding templates; link from ADR-020 and README ([a96f329](https://github.com/BjornMelin/docmind-ai-llm/commit/a96f329d4801e48c7a530086c02a655d23a0efaa))
* add hardware policy guide for device/VRAM helpers and usage ([2362e56](https://github.com/BjornMelin/docmind-ai-llm/commit/2362e569cda844144f2b7148adcb137795e33666))
* add testing notes for patch seams and stability tips ([e708360](https://github.com/BjornMelin/docmind-ai-llm/commit/e7083607274ab030283df69a7b46b76580c3e5cb))
* address review nits ([eb1618a](https://github.com/BjornMelin/docmind-ai-llm/commit/eb1618a00fc94ccb2506ce5ededf5bfe8a66b96e))
* **adr-018:** note compatibility with SPEC-020 template system (RichPromptTemplate) ([30b74ea](https://github.com/BjornMelin/docmind-ai-llm/commit/30b74eaa24ca90c55ad56f2d6d726424dfafe257))
* **adr-058:** enhance tradeoffs, verification, and add constraint documentation ([35d7914](https://github.com/BjornMelin/docmind-ai-llm/commit/35d7914c918be13c8580c50528726d5d742fdfef))
* **adrs,specs:** GraphRAG router+persistence; add ADR-038 and SPEC-014; amend ADR-003/013/016/019/022/024/031/033/034; revise SPEC-006/004/002; update FR-009 and RTM ([8c7b1ff](https://github.com/BjornMelin/docmind-ai-llm/commit/8c7b1ff6b8969bfd03d597b8ef7789a133ceff03))
* **adrs:** add release readiness ADRs ([8193a32](https://github.com/BjornMelin/docmind-ai-llm/commit/8193a32acbcd7464f90e770cfac413a7f84bc923))
* **adrs:** finalize chat persistence ADR-057 ([300cf81](https://github.com/BjornMelin/docmind-ai-llm/commit/300cf8119792fee1f29a1dd35e7d7edc5bfb2447))
* **adrs:** format tables and lists for improved readability ([2f8d1fe](https://github.com/BjornMelin/docmind-ai-llm/commit/2f8d1fe72b9353c7e4a69798a6df342ce2a205a3))
* **adrs:** refresh ADR set and supersede stale decisions ([77bd87c](https://github.com/BjornMelin/docmind-ai-llm/commit/77bd87c4af29903e2c7a2eeaed8bdfc898e1a141))
* **adrs:** standardize decision frameworks and improve formatting across multiple ADRs ([a46c5ea](https://github.com/BjornMelin/docmind-ai-llm/commit/a46c5ea8a15e7ca46b38a4616b4a490f0cb22531))
* **adrs:** update ADR-019 for optional GraphRAG module ([aeaffa9](https://github.com/BjornMelin/docmind-ai-llm/commit/aeaffa939d2355afebe0790c4a59c908d801b9f9))
* **adrs:** update ADR-024 to clarify server-side fusion environment flags\n\n- Removed obsolete client-side fusion knobs; refined server-side fusion environment variables (fusion_mode/fused_top_k) and clarified documentation on env-only overrides and offline defaults. ([7606309](https://github.com/BjornMelin/docmind-ai-llm/commit/7606309aaa02ac1c5dd8b13df9891e198e6a5a58))
* **adr:** update supervisor flags (last_message/full_history, add_handoff_messages); add limitation note on deadline propagation ([ebf7022](https://github.com/BjornMelin/docmind-ai-llm/commit/ebf70220ba753b139c88e11fcf8951dcb2a02e70))
* **agents:** expand opensrc usage guidance ([dacb4ad](https://github.com/BjornMelin/docmind-ai-llm/commit/dacb4adfab2e511548f8a4935cfa41af52f16058))
* align ADR-037 and SPEC-005; add device policy + flags; update integration README example; changelog; .env defaults ([6340ddd](https://github.com/BjornMelin/docmind-ai-llm/commit/6340ddd44968fb2b123676605ea79ff46e904ddb))
* align developer docs ([ee84d85](https://github.com/BjornMelin/docmind-ai-llm/commit/ee84d853a7ec5877dbadfc940f1dce932e60d8f8))
* align docs with router_factory, server-side hybrid, GraphRAG snapshots/exports\n\n- README: router_factory, KG gating, Snapshots & Staleness, GraphRAG exports\n- PRD: rerank latency, sparse=BM42/BM25, SigLIP default, router paragraph\n- Overview: router example; added summaries\n- Dev: architecture/system updates; config flags; operations runbooks\n- Specs/ADRs: spec-004/006/012/014 and requirements alignment\n- User: configuration page, getting-started notes, troubleshooting updates\n- API: router_factory and export examples; CI/CD doc added\n\nBREAKING CHANGE: legacy client-side fusion knobs no longer documented; docs now reflect server-side Qdrant fusion only ([891baf5](https://github.com/BjornMelin/docmind-ai-llm/commit/891baf50339d04dd4c6a914128de25b813908780))
* align GraphRAG env vars and guide ([6044d52](https://github.com/BjornMelin/docmind-ai-llm/commit/6044d52245651e4c67bdefcef837128d53ac1d78))
* align snapshot specs and traceability with new architecture ([185ceb7](https://github.com/BjornMelin/docmind-ai-llm/commit/185ceb7218c24edbb07cea76ed182de17d418d87))
* **analysis:** finalize document analysis modes ADR ([5464821](https://github.com/BjornMelin/docmind-ai-llm/commit/5464821e3351e26801b8094c383c8c93897a7fed))
* **api:** update python example to reflect server-side fusion and router_factory usage\n\n- Enhanced example output to include server-side fusion details and router_factory integration.\n- Added pseudo-code for building a router with vector and optional graph for clarity. ([3e1e970](https://github.com/BjornMelin/docmind-ai-llm/commit/3e1e9701a6964b00a4e59d1f4461f52d99cb8f1b))
* **changelog:** add Unreleased notes for server-side hybrid (RRF default; DBSF env), schema ensure, dedup, telemetry ([021862f](https://github.com/BjornMelin/docmind-ai-llm/commit/021862fe59450a9e4a439ee09015cc105cbf31b7))
* **changelog:** include doc alignment bullets for router/hybrid/GraphRAG/snapshots ([038c989](https://github.com/BjornMelin/docmind-ai-llm/commit/038c989d56e54527988888d12baf7de4d9c79c6b))
* **changelog:** record app helper removal, allowlist hardening, clear caches feature, and new unit tests under Unreleased ([f80c258](https://github.com/BjornMelin/docmind-ai-llm/commit/f80c25859438ab311324876469ba13f19f2cd483))
* **changelog:** update changelog for recent changes ([1f349da](https://github.com/BjornMelin/docmind-ai-llm/commit/1f349da938637954b764dcf40e176dcdbec237f5))
* **chat:** finalize LangGraph chat persistence + agentic memory ([928f36d](https://github.com/BjornMelin/docmind-ai-llm/commit/928f36d2495fd387a0dac19cb842b7e3fb5b9cab))
* **chat:** remove langmem references from SPEC-041 ([d85268d](https://github.com/BjornMelin/docmind-ai-llm/commit/d85268dae8223c81f5a31f97d505cd57adea9c33))
* **ci:** align pipeline with uv/ruff/pyright ([7a3c183](https://github.com/BjornMelin/docmind-ai-llm/commit/7a3c183a4c8fbcbfc0b71af197bc18b6192c8b32))
* **config:** clean configuration guide annotations per review\n\n- Remove extraneous reranking note blockquote that appeared as noisy annotation.\n- Drop trailing inline comment from DOCMIND_RETRIEVAL__USE_RERANKING example.\n- Keeps the guide concise and avoids confusion flagged by Copilot/Sourcery.\n ([7bdd4dd](https://github.com/BjornMelin/docmind-ai-llm/commit/7bdd4dd7f36ed0460f872caac859d40748cd200a))
* **config:** recover and finalize configuration usage guide; add reranking override note ([8c1d547](https://github.com/BjornMelin/docmind-ai-llm/commit/8c1d547f500434adbeb9608772a6eeb09cf1d4a8))
* **configuration:** update header formatting in configuration usage guide ([8909478](https://github.com/BjornMelin/docmind-ai-llm/commit/8909478be5dce1c376663200f30d416c8a925c15))
* Create an index page for superseded specifications and update the main spec index to link to it. ([16f2533](https://github.com/BjornMelin/docmind-ai-llm/commit/16f2533c46c14dfcd6038ac8422f92b584063961))
* **docker:** harden containerization (cpu app + gpu profile) ([9188485](https://github.com/BjornMelin/docmind-ai-llm/commit/9188485d905e2dd4f58d77933b201299d45a1c38))
* enhance docstrings for helper functions and critical algorithms (14 threads) ([9d9b87f](https://github.com/BjornMelin/docmind-ai-llm/commit/9d9b87fe074b41f3f1e149f5fb1d62395e5cbc70))
* enhance documentation and configuration clarity; update link checker script ([cb26151](https://github.com/BjornMelin/docmind-ai-llm/commit/cb261514adab5e04f9a830699c28ebe2209c31e3))
* enhance testing documentation for clarity and debugging options ([d35be07](https://github.com/BjornMelin/docmind-ai-llm/commit/d35be070cda7c0511fbd5d5180824ae0367137b2))
* **final-plans:** add checklists and align plans with implemented router/hybrid/GraphRAG/snapshots\n\n- 003 UI: switch to router_factory; checklists added\n- 004 Analytics: checklists and status updated\n- 005 Eval: import path corrected; aligned\n- 006 Model CLI: checklist added\n- 007 GraphRAG: checklist and acceptance status updated\n- 012 RTM: statuses to Implemented; code paths updated\n- 002 Decisions: added checklists and marked completed items ([68069ed](https://github.com/BjornMelin/docmind-ai-llm/commit/68069edad79ec0f54743a7587d212e984ccb823f))
* finalize ADR/SPEC/prompt set for final release ([a24c3ff](https://github.com/BjornMelin/docmind-ai-llm/commit/a24c3ff32054610099b14a95f458f3493a46025e))
* fix formatting and consistency ([03e38e7](https://github.com/BjornMelin/docmind-ai-llm/commit/03e38e7ad31efa142195bfa68e6192463b1de8bb))
* fix punctuation and unit spacing ([cc33be5](https://github.com/BjornMelin/docmind-ai-llm/commit/cc33be59c032f477925e4c03b7a70bb62d18b238))
* **graphrag:** add 004 implementation checklist (Phase‑2 polish) ([d25461d](https://github.com/BjornMelin/docmind-ai-llm/commit/d25461d32a54435cc550f480276698044df52fa6))
* **graphrag:** migrate plan notes into guides ([2027c91](https://github.com/BjornMelin/docmind-ai-llm/commit/2027c91b4b3b28bb859b9665e4323f7b17dca1b7))
* **graphrag:** migrate plan notes into guides ([4db0565](https://github.com/BjornMelin/docmind-ai-llm/commit/4db056568c2b352b9ba65970550a1780290894f5))
* **graphrag:** remove outdated implementation checklist, spec updates, test plan, and changelog notes for Phase-2 polish ([abbea38](https://github.com/BjornMelin/docmind-ai-llm/commit/abbea3897add5ab0465d33ae431d2778e2b34f52))
* **graphrag:** SPEC/ADR update notes (004) ([d0c819b](https://github.com/BjornMelin/docmind-ai-llm/commit/d0c819bead13fc16fdbe539ca14b8f0dd0677976))
* **graphrag:** test plan diff (004) ([48f6f22](https://github.com/BjornMelin/docmind-ai-llm/commit/48f6f22839982f4f5ec199db5b36fb729367bdc5))
* https://docs.pytest.org/en/stable/explanation/pythonpath.html ([de18b14](https://github.com/BjornMelin/docmind-ai-llm/commit/de18b14743511a19d9c7078a9367e2ede03b53a7))
* improve formatting and clarity in getting-started and system-architecture documentation ([68dafb3](https://github.com/BjornMelin/docmind-ai-llm/commit/68dafb319dc9b575225a273b4bdb1a1fd845b7a3))
* **observability:** align spec and references ([a0d53b8](https://github.com/BjornMelin/docmind-ai-llm/commit/a0d53b80ae451aba3fc8958aaa46badcbd0d40f5))
* **prompts:** add ADR/SPEC read-first and official docs links ([ffade4b](https://github.com/BjornMelin/docmind-ai-llm/commit/ffade4bc2682cd39b1aedaa653793e4fd905539f))
* **prompts:** add implementation executor prompts ([01ae4dc](https://github.com/BjornMelin/docmind-ai-llm/commit/01ae4dc7187b7643f19da0efe45d5f2c8293c9f0))
* **prompts:** add MCP doc corpora block to release readiness ([b76b110](https://github.com/BjornMelin/docmind-ai-llm/commit/b76b1109a1f0aee4002f84e36dcd4f0b118b391d))
* **prompts:** add prompt-041 and refresh tooling ([e878ebc](https://github.com/BjornMelin/docmind-ai-llm/commit/e878ebc64ab84fed22fa87a93d41f11229ee3606))
* **prompts:** add tool and skill preflight guidance ([f12e677](https://github.com/BjornMelin/docmind-ai-llm/commit/f12e67735ab58afac95ab831f2c3f8afc1662163))
* **prompts:** add v1 work-package executor prompts ([02c7854](https://github.com/BjornMelin/docmind-ai-llm/commit/02c7854856cfa00d68eb5f088e6a76fc9f6a03f0))
* **prompts:** align spec-006 prompt with router+persistence plan (SPEC-014) ([556a5dc](https://github.com/BjornMelin/docmind-ai-llm/commit/556a5dc00f9d53cec92fa92db4182561876992d9))
* **prompts:** expand tool playbooks and parallelization ([b1ec7cc](https://github.com/BjornMelin/docmind-ai-llm/commit/b1ec7cc4381022852a61e7adcd1861b1ff86eb1c))
* **prompts:** expand tooling + MCP docs guidance ([2fd4765](https://github.com/BjornMelin/docmind-ai-llm/commit/2fd4765b5daf50d04a5babfe1adba7be8f2ba444))
* **prompts:** standardize executor sections and verification ([fa357d0](https://github.com/BjornMelin/docmind-ai-llm/commit/fa357d020fd3183be4774d5fb78c810b04544140))
* reconcile changelog history ([1c53753](https://github.com/BjornMelin/docmind-ai-llm/commit/1c53753facf64d47d1ca3cd01377972b0d22cd9b))
* refine time travel UX semantics in chat persistence specification ([5e4f75e](https://github.com/BjornMelin/docmind-ai-llm/commit/5e4f75e67dfb9e2f3350cedb2e004fb3850d218c))
* refresh readme and agents guide ([ef9be10](https://github.com/BjornMelin/docmind-ai-llm/commit/ef9be10d7db500dad5d2996a6cf1131193bd6e85))
* refresh readme and agents guide ([bc07749](https://github.com/BjornMelin/docmind-ai-llm/commit/bc07749747abdd7f187a5b5af9d02952cddf71a8))
* relocate last main commit into docs branch ([ba2f737](https://github.com/BjornMelin/docmind-ai-llm/commit/ba2f7375a2449c3732b3f3b3a61125ac37e46ef5))
* relocate last main commit into docs branch ([b58dc6d](https://github.com/BjornMelin/docmind-ai-llm/commit/b58dc6da57c9268ec4bac06965d79510025e68af))
* remove DEVICE_POLICY_CORE flag; adapter: narrow exception handling for imports/settings/torch fallbacks ([16f3834](https://github.com/BjornMelin/docmind-ai-llm/commit/16f3834fef093663d6bbb2e4310d468664e8619e))
* remove outdated task checklist, merge matrix, and GraphRAG prompt files for hybrid retrieval ([351ae68](https://github.com/BjornMelin/docmind-ai-llm/commit/351ae68e11c203e90c2425119d81151f640d0446))
* **reranking:** document router parity and env override; update API examples; config sample ([b00cc58](https://github.com/BjornMelin/docmind-ai-llm/commit/b00cc5852e0d33cddfa3243f93ea8cbbd7cbb87a))
* resolve PR [#54](https://github.com/BjornMelin/docmind-ai-llm/issues/54) review comments across ADRs, specs, and prompts ([cc981bc](https://github.com/BjornMelin/docmind-ai-llm/commit/cc981bc43e4c1b87708cfe8b87c2ecf4f591387a))
* resolve PR review comments ([0ed5a92](https://github.com/BjornMelin/docmind-ai-llm/commit/0ed5a9256553fcb740c9ffa95ff43361ff88aa14))
* **rtm:** update release readiness indices ([710eeb5](https://github.com/BjornMelin/docmind-ai-llm/commit/710eeb5c086833c952461ef1ba0a17debade49f7))
* **security:** safe logging with keyed fingerprints ([48d404a](https://github.com/BjornMelin/docmind-ai-llm/commit/48d404a87c661bcb2581096e256e32e2ab1c9658))
* **spec-020:** add implementation checklist and mark tasks complete ([ac16e32](https://github.com/BjornMelin/docmind-ai-llm/commit/ac16e324d9d19a1d4756a7160cddc751900a1c2f))
* **specs:** add release readiness specs ([59cc6bf](https://github.com/BjornMelin/docmind-ai-llm/commit/59cc6bf771f33a6b197b71eb9c873edd1c854c02))
* **specs:** add SPEC-041 and update RTM ([26e5821](https://github.com/BjornMelin/docmind-ai-llm/commit/26e5821d956b3a83e27a2c7b178a8592d2d03ef3))
* **specs:** mark specs as implemented for chat persistence and multimodal pipeline ([9f8364c](https://github.com/BjornMelin/docmind-ai-llm/commit/9f8364cdb503912901eab793a8505ea18bfdc78f))
* **spec:** SPEC-010 and data/eval README updates for determinism, dynamic [@k](https://github.com/k), and schema fields ([4540b51](https://github.com/BjornMelin/docmind-ai-llm/commit/4540b51547bae4e9c7245726925d70d8b56f29a7))
* **specs:** update SRS/RTM and add v1 work packages ([1f51157](https://github.com/BjornMelin/docmind-ai-llm/commit/1f51157dbc93b7e830346b8e00fdc2c4377617ce))
* **spec:** update Requirements/RTM (telemetry fields); ADR-024 (DBSF env toggle) and ADR-031 (named vectors+IDF); user config envs ([bc198fa](https://github.com/BjornMelin/docmind-ai-llm/commit/bc198fa506508db6b7dee34467f09c2097207a7b))
* **telemetry:** clarify analytics db override rules ([9680acc](https://github.com/BjornMelin/docmind-ai-llm/commit/9680acc561077f1c89559fd92b47356fd64460c3))
* **telemetry:** clarify truncation accounting ([2d2c923](https://github.com/BjornMelin/docmind-ai-llm/commit/2d2c923d5140d377560fca68867c1857421c7beb))
* update ADR documents for improved clarity and formatting ([e6e5179](https://github.com/BjornMelin/docmind-ai-llm/commit/e6e5179e5e2093557d0fe2a7755f4de670602dd6))
* update configuration references and improve documentation clarity ([20acbaf](https://github.com/BjornMelin/docmind-ai-llm/commit/20acbaf674ed6b0447ecc7ba5301127a137506f4))
* update GPU stack and tooling exclusions ([b9280c8](https://github.com/BjornMelin/docmind-ai-llm/commit/b9280c8db5881d6f96771725cb1be304a89d1000))
* update prompts for consistency and clarity ([7b1e9c9](https://github.com/BjornMelin/docmind-ai-llm/commit/7b1e9c90019e62a7cc9320bfc47e9f06a0bc293e))
* update specs/ADRs and RTM; add changelog entry for docs/specs/RTM updates ([7eb20f9](https://github.com/BjornMelin/docmind-ai-llm/commit/7eb20f957b890f43f7f6bea891b23834b9aed4a1))

## [Unreleased]

### Added
- Optional llama-index adapter with lazy availability checks and install guidance (`src/retrieval/llama_index_adapter.py`).
- Coverage for OCR policy, canonicalization, image I/O, router fallbacks, and UI ingestion flows (new unit tests under `tests/unit`).
- Canonical ingestion models and hashing helpers powering the library-first ingestion pipeline (`src/models/processing.py`, `src/persistence/hashing.py`).
- LlamaIndex-based ingestion pipeline, DuckDB-backed cache/docstore wiring, AES-GCM page-image exports, and OpenTelemetry spans (`src/processing/ingestion_pipeline.py`).
- Snapshot lock and writer modules with heartbeat/takeover metadata, atomic promotion, tri-file manifest, and timestamped graph export metadata (`src/persistence/lockfile.py`, `src/persistence/snapshot_writer.py`).
- Snapshot lock heartbeat refresher prevents TTL expiry during long-running persists (`src/persistence/lockfile.py`).
- Manifest metadata now records the active embedding model and spec-compliant `versions["llama_index"]` entry (`src/pages/02_documents.py`).
- PDF page-image exports accept explicit encryption flags to avoid global state mutation (`src/processing/pdf_pages.py`, `src/processing/ingestion_pipeline.py`).
- GraphRAG router/query helpers using native LlamaIndex retrievers and telemetry instrumentation for export counters and spans (`src/retrieval/graph_config.py`, `src/retrieval/router_factory.py`, `src/agents/tools/router_tool.py`).
- Streamlit UI integration that surfaces manifest metadata, staleness badges, GraphRAG export tooling, and ingestion orchestration (`src/pages/01_chat.py`, `src/pages/02_documents.py`, `src/ui/ingest_adapter.py`, `src/agents/coordinator.py`).
- Observability configuration and helpers for OTLP/console exporters and optional LlamaIndex instrumentation (`ObservabilityConfig`, `configure_observability`, updated `scripts/demo_metrics_console.py`).
- Quick-start demos for the overhaul: `scripts/run_ingestion_demo.py` (pipeline smoke test) and refreshed console metrics demo.
- OpenAIConfig (openai.*) with idempotent /v1 base_url normalization and api_key.
- SecurityConfig (security.*) centralizing allow_remote_endpoints, endpoint_allowlist, trust_remote_code.
- HybridConfig (hybrid.*) declarative policy (enabled/server_side/method/rrf_k/dbsf_alpha).
- Clear caches feature: Settings page button and `src/ui/cache.py` helper (bumps `settings.cache_version` and clears Streamlit caches).
- Pure prompting helper: `src/prompting/helpers.py` with `build_prompt_context` (pure; no UI/telemetry) and unit tests.
- SPEC-008: Programmatic Streamlit UI with `st.Page` + `st.navigation`.
  - New pages: `src/pages/01_chat.py`, `src/pages/02_documents.py`, `src/pages/03_analytics.py`.
  - New adapter: `src/ui/ingest_adapter.py` for form-based ingestion.
- ADR-032: Local analytics manager (`src/core/analytics.py`) with DuckDB and best-effort background writes; coordinator logs query metrics.
- SPEC-013 (ADR-040): Model pre-download CLI `tools/models/pull.py` using `huggingface_hub`.
- SPEC-010 (ADR-039): Offline evaluation CLIs:
  - `tools/eval/run_beir.py` (NDCG@10, Recall@10, MRR@10)
  - `tools/eval/run_ragas.py` (faithfulness, answer_relevancy, context_recall, context_precision)
  - `data/eval/README.md` with usage instructions.
- Evaluation harness hardening (schema + determinism):
  - Dynamic `@{k}` metric headers for BEIR (`ndcg@{k}`, `recall@{k}`, `mrr@{k}`) and explicit `k` field.
  - Leaderboard rows now include `schema_version` and `sample_count` for reproducibility.
  - JSON Schemas: `schemas/leaderboard_beir.schema.json`, `schemas/leaderboard_ragas.schema.json`, `schemas/doc_mapping.schema.json`.
  - Validator script: `scripts/validate_schemas.py` (enforces header↔k consistency; validates required fields/types).
  - Determinism utilities under `src/eval/common/`: seeds + thread caps.
  - Doc id mapping persisted to `doc_mapping.json` per run.

### Changed
- Dependency stack now adds the maintained Kùzu property-graph and DuckDB vector-store integrations, replaces the obsolete `llama-index-graph-stores-duckdb` package, and installs the `kuzu` runtime by default.
- ColPali reranker packages now live in an optional `multimodal` extra so the core install can track LlamaIndex 0.14 without version pin conflicts.
- Streamlit ingestion adapter now re-checks embeddings after ingestion and logs when vector index is skipped or built.
- Ingestion pipeline elevates embedding auto-setup failures to warnings and logs plaintext fallbacks for Unstructured reader errors.
- CI now runs base and llama profiles with optional `graph` extra to enforce optional dependency coverage.
- Snapshot manifest/schema now records `complete`, `schema_version`, `persist_format_version`, graph export metadata, and enforces `_tmp-` workspace plus `CURRENT` pointer discipline.
- Guard snapshot workspace initialization to release file locks if creation fails (`src/persistence/snapshot.py`).
- Router, UI, and telemetry layers consistently emit OpenTelemetry spans/metrics for ingestion, snapshot promotion, GraphRAG selection, and export flows.
- Packaging and CI rely on `uv` with an `observability` extra (OTLP exporters, portalocker, LlamaIndex OTEL) and run `ruff`, `pylint`, and `uv run scripts/run_tests.py --coverage` under locked environments.
- Shared fixtures/tests cover ingestion pipeline builders, snapshot locks, Streamlit AppTest interactions, and console exporter stubs.
- Enforced backend-aware OpenAI-like `/v1` normalization in LLM factory for LM Studio, vLLM (OpenAI-compatible), and llama.cpp server.
- Moved all import-time I/O from settings into explicit `startup_init(settings)` in integrations.
- Unified server-side hybrid gating to `retrieval.enable_server_hybrid` + `fusion_mode`; removed legacy flags.
- Settings UI now shows read-only policy state (server-side hybrid, fusion mode, remote endpoint allowance, allowlist size) and resolved normalized backend base URL.
- `.env.example` rewritten to use `DOCMIND_OPENAI__*`, `DOCMIND_SECURITY__*`, and `DOCMIND_VLLM__*`; removed raw `VLLM_*` keys.
- BEIR and RAGAS CLIs now call determinism setup first; BEIR CLI respects `--k` for metric computation and emits dynamic headers matching `k`.
- CI workflow: added schema validation step after tests to catch leaderboard schema drift.
- Post-ingest Qdrant indexing (hybrid) wired into ingestion adapter; Documents page builds a router engine for Chat.
- SPEC-006: GraphRAG exports (Parquet + JSONL) triggered by Documents page checkbox.
- GraphRAG (Phase 1): Library-first refactor of `src/retrieval/graph_config.py` to use only documented LlamaIndex APIs (`as_retriever`, `as_query_engine`, `get`, `get_rel_map`); removed legacy/dead code and index mutation; added portable JSONL/Parquet exports via `get_rel_map`.
- GraphRAG (Phase 2): Added `create_graph_rag_components()` factory to return (`graph_store`, `query_engine`, `retriever`) from a `PropertyGraphIndex`.
- UI wiring: Documents page stores `vector_index`, `hybrid_retriever`, and optional `graphrag_index` in `st.session_state`.
- Coordinator: best-effort analytics logging added after processing each query.
- Router toolset unified: `router_factory.build_router_engine(...)` composes `semantic_search`, `hybrid_search`, and `knowledge_graph` tools; selector policy prefers `PydanticSingleSelector` then falls back to `LLMSingleSelector`.
- GraphRAG helpers (`graph_config.py`) now emit label-preserving exports and provide `get_export_seed_ids()` for deterministic seeding.
- Snapshot manifest enriched and corpus hashing normalized to relpaths; Chat autoload/staleness detection wired to these fields.
- Docs updated: README and system-architecture examples use `MultiAgentCoordinator` directly instead of removed helpers.
- UI refactor: `src/app.py` now only defines pages and runs navigation; all monolithic UI logic moved to `src/pages/*`.
- Tests now patch real library seams (LlamaIndex, utils) instead of `src.app` re-exports.
- Removed short-lived re-exports from `src/app.py` (e.g., LlamaIndex classes, loader helpers) to maintain strict production/test separation.

### Removed

- Legacy ingestion/analytics modules (document processor, cache manager, legacy telemetry instrumentation) and associated compatibility shims/tests.
- Legacy `openai_like_*` fields from settings and corresponding env keys from `.env.example`.
- Legacy `retrieval.hybrid_enabled` and `retrieval.dbsf_enabled`; tests updated accordingly.
- Duplicate and conflicting env keys in `.env.example`.
- Compatibility shims for `DOCMIND_ALLOW_REMOTE_ENDPOINTS`; remote access policy now lives solely under `security.*`.
- Removed legacy helpers from `src/app.py`; app remains a thin multipage shell.

### Tests

- Rebuilt unit/integration suites for ingestion, snapshot locking, GraphRAG seed policy, Streamlit pages, and observability helpers (`tests/unit/processing/test_ingestion_pipeline.py`, `tests/unit/persistence/test_snapshot_*`, `tests/unit/retrieval/test_graph_seed_policy.py`, `tests/unit/ui/test_documents_snapshot_utils.py`, `tests/unit/observability/test_config.py`, integration UI tests).
- Coverage workflow consolidated under `scripts/run_tests.py --coverage` with HTML/JSON/XML artifacts.
- Updated unit and integration tests for new openai.*, security.*, and unified hybrid policy.
- Adjusted factory tests to expect /v1-normalized api_base for OpenAI-like servers.
- Removed legacy env toggle tests and added/updated allowlist and normalization tests.
- New unit tests: allowlist validation (`tests/unit/config/test_settings_allowlist.py`), prompt helper (`tests/unit/prompting/test_helpers.py`), clear caches helper (`tests/unit/ui/test_cache_clear.py`).
- Removed dependency on deleted app helpers: `tests/unit/app/test_app.py` removed.
- Added unit tests for analytics manager insert/prune.
- Added CLI smoke tests for model pull and RAGAS/BEIR harnesses.
- Added page import smoke tests for new Streamlit pages.
- Added unit test for Chat router override mapping.
- Added unit tests: GraphRAG factory (`tests/unit/retrieval/test_graph_rag_factory.py`), graph helpers (`tests/unit/retrieval/test_graph_config_utils.py`), and portable exports (`tests/integration/test_graphrag_exports.py`).
- Added unit tests for SnapshotManager (`tests/unit/persistence/test_snapshot_manager.py`) and router factory (`tests/unit/retrieval/test_router_factory.py`).
- Added integration tests for router composition (`tests/integration/test_ingest_router_flow.py`) and exports (`tests/integration/test_graphrag_exports.py`).
- Added E2E smoke test for Chat via router override (`tests/e2e/test_chat_graphrag_smoke.py`).
- Updated Chat router override test to allow additional forwarded components when present.
- New hybrid/router/graph tests:
  - `tests/unit/retrieval/test_hybrid_retriever_basic.py` (dedup determinism; sparse-unavailable dense fallback)
  - `tests/unit/retrieval/test_router_factory_hybrid.py` (vector + hybrid + knowledge_graph tools)
  - `tests/unit/retrieval/test_seed_policy.py` (retriever-first seed policy and fallbacks)
  - `tests/unit/retrieval/test_graph_helpers.py` (label preservation + `related` fallback)
  - `tests/unit/persistence/test_corpus_hash_relpaths.py` (relpath hashing determinism)
  - Updated/removed legacy integration tests; examples now use `router_factory`

### Docs

- SPEC-014, SPEC-006, SPEC-012, requirements, traceability matrix, README, overview, PRD, and ADR-031/033/019 updated to describe the new ingestion pipeline, snapshot/lock semantics, GraphRAG workflow, and observability configuration.
- ADR-024 amended with OpenAI-compatible servers and openai.* group; documented idempotent `/v1` base URL policy and linked to the canonical configuration guide.
- Configuration Reference updated with a canonical “OpenAI-Compatible Local Servers” section and a Local vs Cloud configuration matrix.
- README updated with DOCMIND_OPENAI__* examples (LM Studio, vLLM, llama.cpp) and a link to the canonical configuration section.
- SPEC-001 (LLM Runtime) updated to reflect OpenAILike usage for OpenAI-compatible backends and corrected Settings page path.
- Traceability (FR-SEC-NET-001) updated for OpenAI-like `/v1` normalization and local-first default posture.
- Requirements specification aligned with nested `openai.*`, `security.*`, and `retrieval.hybrid.*` groups, reiterated the local-first security policy, and linked to the canonical configuration guide.
- Docs updated: README and system-architecture examples use `MultiAgentCoordinator` directly instead of removed helpers.

### Tooling

- CI workflow pins `uv sync --extra observability --group test --frozen`, runs Ruff/pylint/test gates, and enforces updated formatting/lint rules.
- Developer documentation references the new extras, commands, and smoke scripts for ingestion and telemetry verification.

### Reranking/Multimodal Consolidation

- Centralized device and VRAM policy via `src/utils/core` (`select_device`, `has_cuda_vram`) and delegated usage in embeddings and multimodal helpers.
- Unified SigLIP loader (`src/utils/vision_siglip.py`) reused by adapter for consistent caching and device placement.
- Enforced minimal reranking telemetry schema (stage, topk, latency_ms, timeout) with deterministic sorting and RRF tie-breakers.

### Fixed

- Read-only settings panel simplified; no longer references removed `reranker_mode`.
- README updated with offline predownload steps and new envs.
- `validate_startup_configuration` handles Qdrant `ResponseHandlingException`/`UnexpectedResponse` as connectivity failures with structured results (production-safe, test-friendly behavior).

### Security

- Hardened endpoint allowlist validation in `src/config/settings.py` to validate parsed hostnames/IPs and block spoofed `localhost` and malformed URLs.
- Router telemetry test stability: patch `log_jsonl` on the module object via `importlib` to account for LangChain `StructuredTool` wrapper.
- Security: `validate_export_path` error messages aligned with tests and documentation (egress/traversal vs. outside project root).
- Deleted legacy model predownload script: `scripts/model_prep/predownload_models.py`.
- Removed monolithic UI blocks from `src/app.py` (chat/ingestion/analytics).
- Removed legacy/custom router code and tests; all retrieval routes via `router_factory`. No backwards compatibility retained.

## [1.3.0] - 2025-09-08

### Breaking

- Removed legacy prompts module (`src/prompts.py`) and all usages/tests. Migrate to the new file‑based prompt template system (SPEC‑020) via `src.prompting` APIs (`list_templates`, `render_prompt`, `format_messages`, `list_presets`).
- Removed deprecated retrieval modules no longer used after the factory refactor:
  - `src/retrieval/bge_m3_index.py`
  - `src/retrieval/optimization.py`

### Added

- SPEC‑020: Prompt Template System (RichPromptTemplate, file‑based)
  - New `src/prompting/` package: models, loader, registry, renderer, validators
  - Templates under `templates/prompts/*.prompt.md` (YAML front matter + Jinja body)
  - Presets under `templates/presets/{tones,roles,lengths}.yaml`
  - Public API: `list_templates`, `get_template`, `render_prompt`, `format_messages`, `list_presets`
  - Streamlit UI integration (replaces PREDEFINED_PROMPTS)
  - Developer Guide: `docs/developers/guides/adding-prompt-template.md`
- Prompt telemetry logging in app: logs `prompt.template_id`, `prompt.version`, `prompt.name` to local JSONL after render (async + sync paths); sampling/rotation controlled via existing telemetry envs.

### Changed

- Standardized SPEC‑020 document to match repository SPEC format (YAML header, related requirements/ADRs, file operations, checklist).
- ADR‑018 (DSPy Prompt Optimization): marked Implemented; added note on compatibility with SPEC‑020 (rewriter can run before template rendering or on free‑form input).
- RTM updated: FR‑020 marked Completed with code/test references; README “Choosing Prompts” updated to document templates/presets and API usage.

### Removed

- Legacy prompt constants and associated tests; replaced by file‑based templates with RichPromptTemplate.
- Deprecated retrieval modules left from pre‑factory architecture (`bge_m3_index.py`, `optimization.py`).

### Tests

- New unit/integration/E2E smokes for prompting:
  - Unit: loader, registry/renderer, validators
  - Integration: registry list + render smoke
  - E2E: prompt catalog presence
- Updated existing tests to use `src.prompting` instead of legacy prompts.

## [1.2.0] - 2025-09-08

### Breaking

- Removed all legacy client-side fusion knobs (rrf_alpha, rrf_k_constant, fusion weights) and UI reranker toggles. Server-side Qdrant Query API fusion is authoritative; RRF default, DBSF env-gated via `DOCMIND_RETRIEVAL__FUSION_MODE`. Also removed the deprecated `retrieval.reranker_mode` setting — reranker implementation is now auto‑detected (FlagEmbedding preferred when available, else LI).

### Added

- Retrieval telemetry (JSONL) with canonical keys: retrieval.*and dedup.*.
- Rerank telemetry with rerank.*: stage, latency_ms, timeout, delta_changed_count.
- Enforced BM42 sparse with IDF modifier in Qdrant schema; migration helper for existing collections.
- SIGLIP model env `DOCMIND_EMBEDDING__SIGLIP_MODEL_ID` and predownload script `scripts/predownload_models.py`.
- Imaging: DPI≈200 via PyMuPDF, EXIF-free WebP/JPEG, optional AES‑GCM at-rest encryption.
- Tests: unit tests for Query API fusion (RRF/DBSF), rerank timeout fail-open, SigLIP rescore mock, encrypted imaging round-trip, retrieval env mapping.

### Changed

- Reranking is always-on (BGE v2‑m3 text + SigLIP visual) with policy-gated ColPali; UI no longer exposes reranker knobs. Implementation selection is automatic (no env/config toggles).
- Canonical env override: `DOCMIND_RETRIEVAL__USE_RERANKING=true|false` (no UI toggle).
- Deprecated: `DOCMIND_DISABLE_RERANKING` (use `DOCMIND_RETRIEVAL__USE_RERANKING`).

- Test stability and design-for-testability:
  - Removed the last test-only seam from production code: integrations no longer expose a `ClipEmbedding` alias or accept test-only injection. Embedding setup always uses `HuggingFaceEmbedding`; tests patch the constructor via `monkeypatch` when needed.
  - Reverted/avoided test shims in `src/app.py`, `src/retrieval/router_factory.py`, and page modules; imports are production-only.
  - Stabilized import-order–sensitive UI and persistence tests by patching the consumer seams directly and, where necessary, clearing only the specific modules in test-local `conftest.py` (no global module cache hacks). AppTest-based UI tests patch the page module’s attributes (`build_router_engine`, export helpers) instead of relying on import order.
  - Snapshot roundtrip tests stub `llama_index` loaders deterministically by overriding `sys.modules` for the exact import points used by the helpers.

### Fixed

- Read-only settings panel simplified; no longer references removed `reranker_mode`.
- README updated with offline predownload steps and new envs.
- `validate_startup_configuration` handles Qdrant `ResponseHandlingException`/`UnexpectedResponse` as connectivity failures with structured results (production-safe, test-friendly behavior).

## [1.1.0] - 2025-09-07

### Added

- Hybrid retrieval (SPEC‑004): Qdrant server‑side fusion via Query API
  - Named vectors `text-dense` (BGE‑M3 1024D) and `text-sparse` (sparse index)
  - Prefetch dense≈200 / sparse≈400; default fusion RRF; DBSF experimental via env `DOCMIND_RETRIEVAL__FUSION_MODE`
  - De‑dup by `page_id` before fused cut; fused_top_k≈60; telemetry

- Reranking (SPEC‑005; ADR‑037):
  - Default visual re‑score with SigLIP text–image cosine (timeout 150ms)
  - Text BGE v2‑m3 CrossEncoder (timeout 250ms)
  - Optional ColPali policy (VRAM ≥8–12GB, small‑K ≤16, visual‑heavy); cascade SigLIP prune → ColPali final; fail‑open
  - Rank‑level RRF merge across modalities; always‑on (no UI toggles; ops env only)

- PDF page images: WebP default (q≈70, method=6), JPEG fallback; DPI≈200; long‑edge≈2000px; simple pHash for dedup hints.

### Changed

- ADR/specs alignment: ADR‑036 marked Superseded by ADR‑024 v2.7 and SPEC‑005; ADR‑002 reflects SigLIP default; SPEC‑004/005 cross‑refs updated.

## [1.0.0] - 2025-09-02

### Added

- SPEC‑003: Unified embeddings
  - BGE‑M3 text embeddings with dense+sparse outputs via LI factories
  - Tiered image embeddings (OpenCLIP ViT‑L/H, SigLIP base) with LI ClipEmbedding
  - Runtime dimension derivation; deterministic offline tests
  - Legacy wrappers removed; LI‑first wiring throughout

- Unstructured-first document chunking in the processing stack:
  - Partition via `unstructured.partition.auto.partition()`;
  - Chunk via `unstructured.chunking.title.chunk_by_title` with small-section smoothing and `multipage_sections=true`;
  - Automatic fallback to `unstructured.chunking.basic.chunk_elements` for heading‑sparse documents;
  - Table isolation (tables are never merged with narrative text);
  - Full element metadata preserved (page numbers, coordinates, HTML, image path) across chunks.
- Hybrid ingestion pipeline that integrates Unstructured with LlamaIndex `IngestionPipeline` and our `UnstructuredTransformation` for strategy-based processing (hi_res / fast / ocr_only) with caching and docstore support.
- Processing utilities: `src/processing/utils.py` with `is_unstructured_like()` to centralize element safety checks for Unstructured.
- Reranker configuration: surfaced `retrieval.reranker_normalize_scores` in settings and .env; factory respects settings with explicit override precedence.
- Performance: lightweight informational smoke test for processing throughput (kept out of CI critical path).
- Integration tests: additional chunking edge-cases (clustered titles, frequent small headings) and deterministic patches for Unstructured element detection.
- Sprint documentation and navigation:
  - `agent-logs/2025-09-01/sprint-unstructured-first-chunking/README.md` (index) and `011_current_status_and_remaining_work.md` (status)
  - `agent-logs/2025-09-02/processing/001_semantic_cache_and_reranker_ui_research_plan.md` (next sprint research plan)
- New integration tests validating:
  - By-title section boundaries and multi-page sections;
  - Basic fallback on heading-sparse inputs;
  - Table isolation invariants.
- Performance smoke test verifying throughput (informational target ≥ 1 page/sec locally).
- Robust E2E tests for Streamlit app using `st.testing.v1.AppTest` with boundary-only mocks for heavy libs (torch, spacy, FlagEmbedding, Unstructured, Qdrant, LlamaIndex, Ollama) and resilient assertions (no brittle UI string matching).
- Coverage gate scoped to changed subsystems: `src/processing` and `src/config` with `fail_under=29.71%`.

- LLM runtime (SPEC‑001: Multi‑provider runtime & UI):
  - Unified LLM factory `src/config/llm_factory.py` using LlamaIndex adapters:
    - vLLM/LM Studio/llama.cpp server via `OpenAILike` (OpenAI‑compatible)
    - Ollama via `llama_index.llms.ollama.Ollama`
    - Local llama.cpp via `LlamaCPP(model_path=…, model_kwargs={"n_gpu_layers": -1|0})`
    - Respects `model`, `context_window`, and `llm_request_timeout_seconds`
  - New Settings page `src/pages/04_settings.py` with provider dropdown, URL fields, model/path input, context window, timeout, GPU toggle
    - “Apply runtime” rebinds `Settings.llm` (force)
    - “Save” persists to `.env` via a minimal updater
  - Provider badge `src/ui/components/provider_badge.py` shows provider/model/base_url on Chat and Settings
  - Security validation: `allow_remote_endpoints` default False with `endpoint_allowlist`; LM Studio URL must end with `/v1`
  - Observability: logs provider/model/base_url on apply, simple counters (`provider_used`, `streaming_enabled`)
  - Chat wiring: `src/app.py` uses `Settings.llm` for agent system after `initialize_integrations()`
  - Docs/specs: SPEC‑001 checklist checked; RTM updated for FR‑010/FR‑012
  - Tests:
    - Factory type assertions: `tests/unit/test_llm_factory.py`
    - Extended factory behavior (overrides, /v1, llama.cpp local): `tests/unit/test_llm_factory_extended.py`
    - Runtime rebind: `tests/unit/test_integrations_runtime.py`
    - Settings Apply/Save roundtrip: `tests/integration/test_settings_page.py`
    - Provider toggle + apply (ollama/vllm/lmstudio/llamacpp server+local): `tests/integration/test_settings_page.py`

- Multimodal reranking (ADR‑037): ColPali (visual) + BGE v2‑m3 (text), auto‑gated per node modality with fusion and top‑K truncation; builders for text/visual rerankers; integration test added.
- UI controls (ADR‑036): Sidebar Reranker Mode (`auto|text|multimodal`), Normalize scores, Top N.
- Configuration (ADR‑024): global `llm_context_window_max=131072`.
- Ingestion (ADR‑009): PDF page image emission via `pdf_pages_to_image_documents()` tagging `metadata.modality="pdf_page_image"`.

- Integration test for reranker toggle parity (Quick vs Agentic): `tests/integration/test_reranker_parity.py`.
- New utils unit tests to raise coverage: document helpers, core async contexts, monitoring timers.

### Changed

- Agents tools refactor: split `src/agents/tools.py` into cohesive modules under `src/agents/tools/` (`router_tool.py`, `planning.py`, `retrieval.py`, `synthesis.py`, `validation.py`, `telemetry.py`) with `src.agents.tools` as an aggregator. Public API preserved via re-exports; targeted `cyclic-import` disables added where necessary.
- Linting: re-enabled complexity rules (`too-many-statements`, `too-many-branches`, `too-many-nested-blocks`) and fixed violations by extracting helpers. Imports organized per Ruff; helper signatures annotated to satisfy `ANN001`.

- Tools now import patch points (ToolFactory, logger, ChatMemoryBuffer, time) via aggregator for resilient tests.
- Retrieval tool hardened: explicit strategy path by default; conditional aggregator fast‑path for resilience scenarios; DSPy optional with short‑query fallback.
- Planning tweaks for list/categorize decomposition; timing via aggregator time.
- Validation thresholds tuned for source overlap (inclusive) via constants.
- Test runner `scripts/run_tests.py` updated to ASCII‑only output and corrected import validation list.

- UI/runtime (SPEC‑001): removed legacy in‑app backend selection and ad‑hoc LLM construction; centralized provider selection and LLM creation via Settings page + unified factory with strict endpoint validation.

- Retrieval & Reranking:
  - Router parity: RouterQueryEngine now passes reranking `node_postprocessors` for vector/hybrid/KG tools when `DOCMIND_RETRIEVAL__USE_RERANKING=true` (mirrors ToolFactory). Safe fallbacks keep older signatures working.
  - Tests: Added router_factory injection toggle test and KG fallback tests; added hybrid injection test behind explicit `enable_hybrid=True` with stubs.

### Docs/Specs/RTM

- Specs updated:
  - spec‑014: Added UI staleness badge exact tooltip, single‑writer lock semantics with timeout, manifest fields, atomic rename guidance, and acceptance/UX mapping.
  - spec‑004: Clarified server‑side‑only hybrid via Qdrant Query API (Prefetch + FusionQuery), named vectors + IDF, dedup, and telemetry; prohibited client‑side fusion and UI fusion toggles.
  - spec‑006: Defined GraphRAG exports and seeds policy (JSONL required; Parquet optional; deterministic deduped seeds capped at 32).
  - spec‑002/spec‑003/spec‑005: Clarified ingestion OCR fallback; embedding routing/dimensions (BGE‑M3 1024; SigLIP); always‑on reranking and ColPali gating.
  - spec‑010: Documented offline evaluation CLIs and CSV schema expectations; strict mocks/no heavy downloads in CI.
  - spec‑012: Added canonical telemetry events (router_selected, snapshot_stale_detected, export_performed, traversal_depth) and DuckDB analytics guidance.
  - spec‑013: Documented offline mode (HF_HUB_OFFLINE) and Parquet extras; JSONL fallback when pyarrow missing.
  - spec‑001/spec‑011: Settings scope & validation; offline‑first allowlist; LM Studio /v1 rule; selector policy; secrets redaction and non‑egress export requirements.
  - ADR‑011: Supervisor output_mode (last_message/full_history), add_handoff_messages rename, streaming fallback, and best‑effort analytics guidance.
  - ADR‑024: Offline defaults and endpoint allowlist policy.

- Requirements/RTM:
  - requirements.md: Added FR‑009.1–009.6 (staleness badge; SnapshotManager lock/rename; exports JSONL/Parquet; deterministic seed policy; export path security; telemetry events) and FR‑SEC‑NET‑001 (offline‑first allowlist; LM Studio /v1).
  - traceability.md: Mapped new FRs to code/tests and marked them Implemented.

### Fixed

- Stabilized supervisor shims in integration tests (compile/stream signatures) and ensured InjectedState overrides visibility.
- Addressed flaky/residual lint warnings across test suites; ensured ruff clean and pylint tests ≥ 9.8.

- Removed legacy/deprecated splitters (SentenceSplitter-based chunkers) and all backward-compatibility paths in favor of Unstructured-first approach.
- Updated `src/processing/document_processor.py` to:
  - Select strategy by file type (hi_res/fast/ocr_only),
  - Hash, cache, and error-handling improvements with tenacity-backed retry,
  - Convert chunked elements to LlamaIndex `Document` nodes, then normalize to our `DocumentElement` API.
- Modernized E2E test architecture:
  - Centralized stubs for agents, tools, and utils modules;
  - Replaced strict call-count and exact-text checks with structure/behavior checks;
  - Ensured offline determinism by stubbing external services and model backends.
- DocumentProcessor: removed test-only code paths (e.g., partition kwargs/type fallbacks, patched-chunker detection) to keep src production-pure; tests now patch compatibility in tests/.
- Reranker: `create_bge_cross_encoder_reranker()` now resolves settings first, then explicit args; enforces device-aware batch sizing; returns fresh `NodeWithScore` instances instead of mutating inputs.
- Cache stats: `SimpleCache` now derives `total_documents` from underlying KVStore instead of internal counters.
- .env.example: added missing mappings for processing (new_after_n_chars, combine_text_under_n_chars, multipage_sections, debug_chunk_flow), retrieval (reranker_normalize_scores), cache (ttl_seconds, max_size_mb), DB (qdrant_timeout), and vLLM base URLs.
- Router Query Engine integration: passed `llm=self.llm` into `RetrieverQueryEngine.from_args` to avoid accidental network calls via global settings.
- Retrieval pipeline (ADR‑003): inject reranker by mode (auto/multimodal → `MultimodalReranker`, text → text‑only builder). Updated `src/retrieval/__init__.py` docstring to ADR‑037.

### Fixed

- Stabilized pipeline initialization in performance tests by providing fake `IngestionPipeline` with `cache`/`docstore` attributes to satisfy pydantic validation.
- Prevented import-time failures by lazily importing Ollama in `setup_llamaindex()` and avoiding heavy integration side effects at module import.
- E2E and integration tests: ensured async mocks are awaited; stabilized partition side-effects to accept kwargs; improved deterministic patches for Unstructured element detection.
- Unit tests: added reranker factory properties test; tightened exception expectations and removed unused mocks where applicable.
- Tests: replaced brittle full‑app UI import with deterministic unit/integration coverage to avoid side effects.

### Removed

- Legacy chunkers and compatibility shims.
- Implicit side-effect initialization from `src/config/__init__.py` (explicit `initialize_integrations()` only where needed).
- Test-compatibility logic from production code (src/). Tests now own the compatibility shims and patches.
- ColBERT reranker integration and legacy BGECrossEncoder path; all related tests deleted. No backward compatibility retained.

### Testing & Tooling

- Ruff formatting and lint cleanup on changed files.
- Pylint score for `document_processor.py` ≥ 9.5 (≈ 9.85).
- Deterministic, offline test execution via mocks/stubs; no GPU/network required.
- Ruff: formatted and lint-clean on changed files.
- Pylint: changed modules meet quality gate; unrelated large modules earmarked for a future maintenance pass.
- Scoped pylint to `src/` for refactor; dependencies updated: replace `llama-index-postprocessor-colbert-rerank` with `llama-index-postprocessor-colpali-rerank`.

### Migration Notes

- No backwards compatibility retained for legacy chunkers. Downstream callers should rely on Unstructured-first processing and the `DocumentProcessor` API.
- Tests should import heavy integrations behind stubs and use library-first mocks (LlamaIndex `MockLLM`, `MockEmbedding`, in-memory `VectorStoreIndex`).

---

[1.0.0]: https://example.com/releases/tag/v1.0.0

### Added — Router, GraphRAG, Multimodal, Settings

- LangGraph Supervisor integration with `router_tool` wired into coordinator; InjectedState carries router engine and runtime toggles.
- Sub‑question decomposition strategy via `SubQuestionQueryEngine` exposed as `sub_question_search` in router toolset.
- GraphRAG default ON per ADR‑019; knowledge_graph tool registered when KG index present; one‑line rollback flag.
- Multimodal ingestion and `MultiModalVectorStoreIndex` built only when image/page‑image nodes exist; `multimodal_search` tool conditional.
- Settings convenience aliases and helpers:
  - Aliases: `context_window_size`, `chunk_size`, `chunk_overlap`, `enable_multi_agent` (validated and mapped to nested fields).
  - Helpers: `get_model_config()`, `get_embedding_config()`, `get_vllm_env_vars()`.
- Optional BM25 keyword tool behind `retrieval.enable_keyword_tool` (disabled by default).
- Telemetry logging tests for router creation, strategy selection, and fallback.
- New router tool unit tests (success, error, no metadata) and supervisor integration path.
- Sub‑question fallback unit test (tree_summarize) when SQE creation fails.

### Changed

- Agents tools refactor: split `src/agents/tools.py` into cohesive modules under `src/agents/tools/` (`router_tool.py`, `planning.py`, `retrieval.py`, `synthesis.py`, `validation.py`, `telemetry.py`) with `src.agents.tools` as an aggregator. Public API preserved via re-exports; targeted `cyclic-import` disables added where necessary.
- Linting: re-enabled complexity rules (`too-many-statements`, `too-many-branches`, `too-many-nested-blocks`) and fixed violations by extracting helpers. Imports organized per Ruff; helper signatures annotated to satisfy `ANN001`.

- Replaced all legacy `multi_query_search` references with `sub_question_search`; updated selector descriptions and constants.
- System and integration tests migrated to nested settings and final defaults (embedding, database.sqlite path, monitoring limits, GraphRAG default ON); system workflows updated for async behavior.
- Documentation: added developer guide `docs/developers/configuration-usage-guide.md`; linked from developers README.

### (Test-related items grouped under Added/Changed)

### Removed

- Legacy `multi_query_search` references and assumptions in code and tests.

- LlamaIndex integration packages added as first‑class dependencies:
  - `llama-index-embeddings-clip` for CLIP multimodal embeddings
  - `llama-index-postprocessor-colbert-rerank` for ColBERT reranking
  - `llama-index-llms-ollama` for local Ollama backend
- New cache unit test: DuckDBKV lifecycle (create/stats/clear) under `tests/unit/cache/test_ingestion_cache.py`.
- Shared backlog tracker: `agent-logs/2025-09-02/next-tasks/001-next-tasks-and-research-backlog.md`.
- Caching architecture: replaced custom `SimpleCache` with LlamaIndex `IngestionCache` backed by `DuckDBKVStore` (single local file at `cache/docmind.duckdb`). No back‑compat.
- DocumentProcessor: wired DuckDBKV‐backed `IngestionCache`; added safe JSON coercion for Unstructured element metadata to support KV persistence.
- Utils: cache clear/stats now operate on the DuckDB cache file (delete/inspect path).
- Tests: modernized app and perf test patches to target `src.app` functions; ensured deterministic performance tests and removed brittle timing checks.
- DI: container creation logs a standard message for test capture (`Created ApplicationContainer`).
- Resolved import errors by adding LlamaIndex split packages and using correct namespaced imports (CLIP, ColBERT, Ollama).
- Prevented JSON serialization errors when persisting pipeline outputs by coercing metadata to JSON‑safe types.
- Stabilized performance tests by patching the correct targets and relaxing environment‑sensitive assertions.
- `src/cache/simple_cache.py` and all references.
- `tests/unit/cache/test_simple_cache.py` and docs/spec references to SimpleCache; specs now reflect ADR‑030.
- Agents tools refactor: split `src/agents/tools.py` into cohesive modules under `src/agents/tools/` (`router_tool.py`, `planning.py`, `retrieval.py`, `synthesis.py`, `validation.py`, `telemetry.py`) with `src.agents.tools` as an aggregator. Public API preserved via re-exports; targeted `cyclic-import` disables added where necessary.
- Linting: re-enabled complexity rules (`too-many-statements`, `too-many-branches`, `too-many-nested-blocks`) and fixed violations by extracting helpers. Imports organized per Ruff; helper signatures annotated to satisfy `ANN001`.
- Ingestion (SPEC‑002): Completed Unstructured‑first ingestion with LlamaIndex IngestionPipeline
  - Strategy mapping (hi_res / fast / ocr_only) with OCR fallback
  - Deterministic IDs for text and page‑image nodes via `sha256_id`
  - PDF page‑image emission with stable filenames (`__page-<n>.png`) and bbox
  - DuckDBKV‑backed `IngestionCache` at `./cache/docmind.duckdb`; surfaced cache stats
  - Tests: unit and integration for page images, deterministic IDs, chunking heuristics

- Config: `llm_backend` is a strict literal ("ollama"|"llamacpp"|"vllm"|"lmstudio"); tests updated accordingly
- Models: `ErrorResponse` enriched with optional `traceback` and `suggestion`; `PdfPageImageNode` no longer carries error fields
- Tests: PDF rendering patched/stubbed in unit/integration paths to avoid heavy I/O while preserving behavior under test
- Docs: SPEC‑002 updated to reflect actual implementation (Unstructured chunking, IngestionPipeline transform, caching, page‑image emission, validation commands)
- Server-side hybrid toggle and UI
  - Added `retrieval.enable_server_hybrid` in `src/config/settings.py` (default off) and wired precedence in `src/retrieval/router_factory.py` (explicit param > settings > default).
  - Added Settings UI toggle in `src/pages/04_settings.py` with `.env` persistence via `DOCMIND_RETRIEVAL__ENABLE_SERVER_HYBRID`.
- Ingestion analytics
  - Added best-effort analytics logging inside `src/processing/document_processor.py` using `AnalyticsManager` (DuckDB prepared statements). Logs ingest latency and element counts behind `analytics_enabled`.
- Telemetry/security completeness
  - Enriched reranker telemetry with `rerank.path` and `rerank.total_timeout_budget_ms` in `src/retrieval/reranking.py`.
  - Added endpoint allowlist tests under `tests/unit/config/test_endpoint_allowlist.py`.
- Hook robustness
  - Hardened LangGraph pre/post hooks in `src/agents/coordinator.py` to annotate state on failures (`hook_error`, `hook_name`) without crashing.
- Micro-tests for stability/coverage
  - Added tests: router settings fallback, hooks resilience, security utils, sparse query encoding, and settings round‑trip.

### Changed

- Minor router_factory docs/comments; no behavior change beyond settings fallback tests.

### Tests

- `tests/unit/retrieval/test_router_factory_settings_fallback.py` to validate settings‑driven hybrid tool registration.
- `tests/unit/agents/test_hooks_resilience.py` to ensure hook exceptions are non‑fatal.
- `tests/unit/utils/test_security.py` for PII redaction/egress checks.
- `tests/unit/retrieval/test_sparse_query_encode.py` for sparse query encoding path.
- `tests/unit/config/test_settings_roundtrip.py` for retrieval flag presence.
- BREAKING: Removed `src/agents` public re-exports. All imports must use explicit module paths, e.g. `from src.agents.coordinator import MultiAgentCoordinator` and `from src.agents.tool_factory import ToolFactory`.
- Docs updated to reflect explicit import guidance and to avoid aggregator/app re-exports in tests.
- Deprecated/legacy aggregator import examples were removed from documentation.
