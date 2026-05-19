# MMO Open Source

Verified code, evaluation scripts, and benchmark reports for AI research projects.

> ⚠️ All code published here has been tested and security-reviewed.  
> No sensitive information (API keys, server IPs, credentials) is included.

## Projects

### [cot/](cot/) — Chain-of-Thought Compression
Evaluation framework and method implementations for reducing CoT token usage in LLM reasoning.

Includes implementations of:
- **DEER** — Dynamic Early Exit in Reasoning
- **TALE** — Token-Budget-Aware LLM Reasoning  
- **Don't Overthink** — Short-m@k Majority Voting

## Quick Start

Each project has its own README with setup instructions. Generally you'll need:

1. A running vLLM server (or compatible OpenAI API endpoint)
2. Set environment variables:
   ```bash
   export VLLM_URL="http://your-server:8000"
   export VLLM_API_KEY="your-api-key"
   export MODEL_NAME="your-model-name"
   ```
3. Install dependencies: `pip install httpx aiohttp pyyaml`
