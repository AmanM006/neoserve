"""Produce an INT8-activation / INT4-weight (W4A8) quantized model for Arm serving.

W4A8 pairs 4-bit weights with 8-bit activations. On Graviton4 with vLLM, the INT4
weight path is accelerated by Arm KleidiAI INT4 micro-kernels, giving the best
throughput/$ of the three precisions (~+29% over W8A8 in Arm's measurements) at a
small, guarded quality cost. NeoServe's quality guard rejects it if perplexity
degrades past the model's budget.

Uses llm-compressor's GPTQ W4A8 recipe (group-wise 4-bit weights + dynamic per-token
8-bit activations).

Example:
    python -m quantize.make_w4a8 \
        --model meta-llama/Llama-3.1-8B-Instruct \
        --out models/llama31-8b-w4a8 \
        --group-size 128 --calib-samples 512
"""
from __future__ import annotations

import argparse
from pathlib import Path


def _recipe(group_size: int) -> str:
    return f"""
quant_stage:
  quant_modifiers:
    GPTQModifier:
      sequential_update: true
      dampening_frac: 0.01
      config_groups:
        group_0:
          weights: {{num_bits: 4, type: int, symmetric: true, strategy: group, group_size: {group_size}}}
          input_activations: {{num_bits: 8, type: int, symmetric: true, strategy: token, dynamic: true}}
          targets: [Linear]
"""


def build(model: str, out: str, group_size: int, calib_samples: int,
          seq_len: int, dataset: str) -> None:
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from llmcompressor import oneshot

    tok = AutoTokenizer.from_pretrained(model)
    mdl = AutoModelForCausalLM.from_pretrained(model, torch_dtype="auto", device_map="cpu")

    # Plain "wikitext" is not a namespace/name repo id on current HF; pin the config.
    if dataset == "wikitext":
        ds = load_dataset("wikitext", "wikitext-2-raw-v1",
                          split=f"train[:{calib_samples}]")
    else:
        ds = load_dataset(dataset, split=f"train[:{calib_samples}]")
    text_col = "text" if "text" in ds.column_names else ds.column_names[0]
    ds = ds.map(lambda s: tok(s[text_col], truncation=True, max_length=seq_len),
                remove_columns=ds.column_names)

    recipe_path = Path(out).with_suffix(".recipe.yaml")
    recipe_path.parent.mkdir(parents=True, exist_ok=True)
    recipe_path.write_text(_recipe(group_size), encoding="utf-8")

    oneshot(model=mdl, dataset=ds, recipe=str(recipe_path),
            max_seq_length=seq_len, num_calibration_samples=calib_samples,
            output_dir=out)
    tok.save_pretrained(out)
    print(f"[w4a8] wrote quantized model to {out} (group_size={group_size})")
    print("[w4a8] serve with KleidiAI-enabled vLLM: vllm serve", out,
          "--device cpu --dtype bfloat16")
    print("[w4a8] REMEMBER: run the NeoServe quality guard before publishing.")


def main() -> None:
    ap = argparse.ArgumentParser(description="Build a W4A8 model for Arm serving")
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--group-size", type=int, default=128)
    ap.add_argument("--calib-samples", type=int, default=512)
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--dataset", default="wikitext",
                    help="HF dataset name; wikitext uses config wikitext-2-raw-v1")
    args = ap.parse_args()
    build(args.model, args.out, args.group_size, args.calib_samples,
          args.seq_len, args.dataset)


if __name__ == "__main__":
    main()
