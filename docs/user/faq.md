# Frequently Asked Questions

Quick answers to common questions about DocMind AI.

## General Questions

### What is DocMind AI?
DocMind AI is a local document analysis system that uses AI to understand and analyze your documents. It runs entirely on your machine with zero cloud dependencies, ensuring complete privacy.

### What makes DocMind AI different from other document tools?
- **100% Local**: Your documents never leave your machine
- **AI-Powered**: Uses advanced language models for intelligent analysis
- **Multi-Agent System**: 5 specialized AI agents coordinate to handle complex queries
- **No API Keys**: Works offline after initial setup
- **Multiple Formats**: Supports PDFs, Word docs, spreadsheets, and 15+ other formats

### Do I need an internet connection?
After initial setup, DocMind AI works completely offline. You only need internet for:
- Initial installation and model downloads
- Software updates (optional)

## Hardware and Performance

### What hardware do I need?
**Minimum**: 8GB RAM, any modern CPU  
**Recommended**: 16GB RAM, RTX 4060 (12GB VRAM)  
**Optimal**: 32GB RAM, RTX 4090 (24GB VRAM)

### Can I run DocMind AI without a GPU?
Yes! DocMind AI works on CPU-only systems, though it will be slower (10-20 seconds per query vs 1-3 seconds with GPU).

### How much storage space do I need?
- Application: ~50MB
- Models: 5-10GB (downloaded automatically)
- Document cache: 1GB (configurable)
- Total: ~15-20GB

### What kind of performance should I expect?
- **RTX 4060**: 3-7 seconds per query
- **RTX 4090**: 1-3 seconds per query
- **CPU-only**: 10-20 seconds per query

### Which NVIDIA GPUs are supported?
Any modern NVIDIA GPU with 8GB+ VRAM works. Tested configurations:
- RTX 4060/4070: Good performance
- RTX 4080/4090: Excellent performance
- RTX 3080/3090: Compatible with some optimizations disabled

## Privacy and Security

### Is my data private?
Absolutely. DocMind AI:
- Processes everything locally on your machine
- Never sends data to external servers
- Uses local AI models through Ollama
- Stores documents and results only on your system

### Can DocMind AI access my other files?
No. DocMind AI only accesses files you explicitly upload through the interface. It cannot read files from other locations on your system.

### Where are my documents stored?
Documents are temporarily processed in memory and can be cached locally for faster re-analysis. You can disable caching if desired. Nothing is sent to external servers.

### Is DocMind AI compliant with privacy regulations?
Since all processing happens locally on your machine, DocMind AI is inherently compliant with privacy regulations like GDPR, HIPAA, and others that require data locality.

## Documents and File Formats

### What file formats are supported?
**Documents**: PDF, DOCX, TXT, RTF, MD  
**Spreadsheets**: XLSX, CSV  
**Presentations**: PPTX  
**Web**: HTML, XML, JSON  
**Other**: MSG (email), EPUB, ODT, and code files

### What's the maximum file size?
Default limit is 100MB per file, configurable up to 200MB+ depending on your hardware.

### Can I upload multiple documents at once?
Yes! You can upload multiple related documents and analyze them together or separately.

### Does DocMind AI work with scanned PDFs?
Yes, DocMind AI uses OCR (Optical Character Recognition) to extract text from scanned documents and images.

## Usage and Features

### What types of questions can I ask?
DocMind AI handles various query types:
- **Simple lookups**: "What is the project budget?"
- **Summaries**: "Summarize the key findings"
- **Comparisons**: "Compare Q1 vs Q2 performance"
- **Analysis**: "What risks are identified and how severe are they?"
- **Multi-document**: "How do these reports relate to each other?"

### What is the multi-agent system?
DocMind AI uses 5 specialized AI agents that coordinate automatically:
1. **Query Router**: Analyzes your question
2. **Query Planner**: Breaks down complex queries
3. **Retrieval Expert**: Finds relevant information
4. **Result Synthesizer**: Combines findings
5. **Response Validator**: Ensures quality

### Can I ask follow-up questions?
Yes! DocMind AI maintains conversation history, so you can build on previous questions and dive deeper into topics.

### How accurate are the responses?
DocMind AI aims for high accuracy by:
- Using advanced language models
- Validating responses with multiple agents
- Providing source citations
- Including confidence scores

Typical accuracy for factual questions is 85-95%, depending on document quality and query complexity.

## Installation and Setup

### How long does installation take?
- Basic installation: 5-10 minutes
- Model download: 10-15 minutes (one-time)
- Total first-time setup: ~20 minutes

### Do I need technical expertise?
No! The installation process uses simple copy-paste commands. If you can follow a recipe, you can install DocMind AI.

### What is Ollama and do I need it?
Ollama is the local AI backend that runs language models on your machine. Yes, you need it - it's what makes DocMind AI work offline and privately.

### Can I use different AI models?
Yes! DocMind AI supports any model available through Ollama. The recommended model (qwen3-4b-instruct-2507) provides the best balance of performance and accuracy.

## Configuration and Troubleshooting

### How do I optimize performance for my hardware?
DocMind AI includes pre-configured profiles:
- Use "student" profile for older hardware
- Use "gaming" profile for RTX 4060/4070
- Use "research" profile for RTX 4090
- See [configuration.md](configuration.md) for details

### Why is DocMind AI slow?
Common causes and solutions:
- **CPU-only**: Enable GPU acceleration if available
- **Insufficient VRAM**: Reduce context window size
- **Model not optimized**: Enable performance optimizations
- **System overload**: Close other applications

### How do I check if everything is working correctly?
Look for these indicators in the sidebar:
- ✅ GPU Status: Shows your hardware
- ✅ Model Status: "Ready"
- ✅ Multi-Agent: "5 Agents Active"
- ✅ Ollama: "Connected"

## Limitations and Capabilities

### What are DocMind AI's limitations?
- **Language**: Primarily designed for English documents
- **Model Knowledge**: Limited to training data (no real-time information)
- **Processing Time**: Complex queries take longer than simple web searches
- **Hardware Dependent**: Performance varies significantly with hardware

### Can DocMind AI generate new content?
DocMind AI focuses on analyzing and extracting information from your documents rather than generating new content. It can summarize, compare, and explain, but it's not designed for creative writing.

### Does DocMind AI learn from my documents?
No, DocMind AI doesn't permanently learn or update from your documents. Each session starts fresh, ensuring privacy and preventing data leakage between different document sets.

### How many documents can I analyze at once?
There's no strict limit, but practical limits depend on your hardware:
- **8GB RAM**: 10-50 documents
- **16GB RAM**: 50-200 documents  
- **32GB+ RAM**: 200+ documents

## Getting Help

### Where can I get help if something isn't working?
1. Check [troubleshooting.md](troubleshooting.md) for common issues
2. Review [getting-started.md](getting-started.md) for setup problems
3. Search [GitHub Issues](https://github.com/BjornMelin/docmind-ai-llm/issues)
4. Open a new issue with your system details and error logs

### How do I report a bug or request a feature?
Use GitHub Issues with:
- Clear description of the problem or requested feature
- Your system specifications (GPU, RAM, OS)
- Error logs if applicable
- Steps to reproduce the issue

### Is there a community for DocMind AI users?
Yes! Join the discussions on GitHub for:
- Tips and best practices
- Community support
- Feature discussions
- Use case examples

---

**Still have questions?** Check the [getting-started guide](getting-started.md) for installation help, [configuration guide](configuration.md) for optimization tips, or [troubleshooting guide](troubleshooting.md) for common issues.