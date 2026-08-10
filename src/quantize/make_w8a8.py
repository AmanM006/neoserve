"""Produce an INT8 W8A8 (weight+activation) quantized model for Arm serving.

On Graviton4, W8A8 matmul runs through oneDNN JIT SMMLA (i8mm) kernels, which
roughly double INT8 matmul throughput vs NEON dot-product. W8A8 is typically near
lossless (<1% perplexity delta), making it the safe default win over bf16.

Uses llm-compressor's GPTQ/SmoothQuant W8A8 recipe. Run on the Arm host (or any box
with enough RAM); publish the result and serve it with vLLM.

Example:
    python -m quantize.make_w8a8 \
        --model meta-llama/Llama-3.1-8B-Instruct \
        --out models/llama31-8b-w8a8 \
        --calib-samples 512
"""
from __future__ import annotations

import argparse
from pathlib import Path


W8A8_RECIPE = """
quant_stage:
  quant_modifiers:
    SmoothQuantModifier:
      smoothing_strength: 0.8
    GPTQModifier:
      sequential_update: true
      dampening_frac: 0.01
      config_groups:
        group_0:
          weights:   {num_bits: 8, type: int, symmetric: true, strategy: channel}
          input_activations: {num_bits: 8, type: int, symmetric: true, strategy: token, dynamic: true}
          targets: [Linear]
"""


def build(model: str, out: str, calib_samples: int, seq_len: int,
          dataset: str) -> None:
    # Imports are local so the harness can be imported on machines without the
    # heavy ML stack (mock mode). These run on the Arm host per deploy/ec2-setup.sh.
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from llmcompressor import oneshot

    tok = AutoTokenizer.from_pretrained(model)
    mdl = AutoModelForCausalLM.from_pretrained(model, torch_dtype="auto", device_map="cpu")

    if dataset == "wikitext":
        ds = load_dataset("wikitext", "wikitext-2-raw-v1",
                          split=f"train[:{calib_samples}]")
    else:
        ds = load_dataset(dataset, split=f"train[:{calib_samples}]")
    text_col = "text" if "text" in ds.column_names else ds.column_names[0]

    def tokenize(sample):
        return tok(sample[text_col], truncation=True, max_length=seq_len)

    ds = ds.map(tokenize, remove_columns=ds.column_names)

    recipe_path = Path(out).with_suffix(".recipe.yaml")
    recipe_path.parent.mkdir(parents=True, exist_ok=True)
    recipe_path.write_text(W8A8_RECIPE, encoding="utf-8")

    oneshot(model=mdl, dataset=ds, recipe=str(recipe_path),
            max_seq_length=seq_len, num_calibration_samples=calib_samples,
            output_dir=out)
    tok.save_pretrained(out)
    print(f"[w8a8] wrote quantized model to {out}")
    print("[w8a8] serve with: vllm serve", out, "--device cpu --dtype bfloat16")


def main() -> None:
    ap = argparse.ArgumentParser(description="Build a W8A8 model for Arm serving")
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--calib-samples", type=int, default=512)
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--dataset", default="wikitext")
    args = ap.parse_args()
    build(args.model, args.out, args.calib_samples, args.seq_len, args.dataset)


if __name__ == "__main__":
    main()
