# NeoServe — 90-Second Video Demo Script

> **Target Length:** 90 seconds (under 3 minutes max)  
> **Goal:** Show judges the live dashboard, ledger verification, Arm Performix PMU hardware proof, and Docker recipe in action.

---

## 🎬 Video Recording Blueprint

### [0:00 - 0:20] The Problem & Billboard Metric
- **Screen:** Open [http://localhost:3010](http://localhost:3010) (NeoServe Pitch-Black Dark Dashboard).
- **Voiceover:**  
  *"Hi everyone! Welcome to NeoServe — our cost and SLO-aware LLM serving optimizer built for AWS Graviton4 for the Arm Create AI Optimization Challenge.*  
  *Most entries in Track 2 focus on single-stream llama.cpp speedups. But production cloud serving asks a different question: at real traffic concurrency and a strict latency SLO, which configuration serves tokens **cheapest** on Arm?*  
  *On AWS Graviton4, NeoServe slashes serving costs from **$1.44 down to $0.745 per 1M tokens** — delivering a **1.94× throughput speedup** and saving **$3,505 a month** at 5 billion tokens per month scale."*

### [0:20 - 0:45] The Pareto Frontier & Performix Top-Down PMU
- **Screen:** Hover over the Pareto Frontier SVG chart, switch between `Qwen2.5-1.5B` and `Llama-3.1-8B` tabs, then scroll to the Arm Performix Top-Down PMU section.
- **Voiceover:**  
  *"NeoServe searches memory allocators like `mimalloc`, physical thread binding, continuous batching, and KleidiAI INT4 micro-kernels.*  
  *We prove *why* the winner is faster using Arm Performix PMU top-down hardware profiling: shifting execution from backend memory stalls into **retiring instructions (from 27% up to 51.8%)** and raising IPC from **1.42 to 1.49**."*

### [0:45 - 1:05] Quality Guard & Cryptographic Ledger
- **Screen:** Terminal window showing `python scripts/verify_ledger.py results/canonical`.
- **Voiceover:**  
  *"A speedup is only a win if quality holds. NeoServe evaluates `lm_eval` wikitext perplexity to ensure quantized models pass their quality budget (+2.37% PPL).*  
  *Every artifact, report, and cost card is cryptographically hashed in a SHA-256 `ledger.json`, verified right here with `verify_ledger.py`."*

### [1:05 - 1:30] Docker Recipe & Interactive MCP Playground
- **Screen:** Click through the **Interactive MCP Agent Playground** in the dashboard (`recommend_config()`, `get_serving_recipe()`), then show `serving_recipe/qwen25-1p5b/compose.yaml`.
- **Voiceover:**  
  *"NeoServe outputs reusable production artifacts: a tuned `Dockerfile.arm64` + `compose.yaml` recipe ready for Docker Compose, and a Model Context Protocol (MCP) tool allowing AI agents like Cursor or Claude to ask `recommend_config` automatically.*  
  *NeoServe turns Arm Neoverse cloud CPUs into high-efficiency production serving engines. Thank you!"*
