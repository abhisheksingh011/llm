# Which LLM will run on your laptop?

**[→ Try it](https://abhisheksingh011.github.io/llm/)** · one HTML file · no build,
no dependencies, no network requests, nothing uploaded

---

## The problem

You want to run a language model locally. The first question is the only one that
matters — *will it actually run on this machine?* — and the available answers are
all bad.

**Rules of thumb are wrong in the direction that wastes your time.** "8 GB of VRAM,
so a 7B model" ignores the single largest variable. Download 40 GB, wait an hour,
watch it fail to load, guess again.

**VRAM calculators ask you what you're trying to find out.** Layer count, hidden
dimension, KV heads, head dim, GQA ratio. If you knew those, you wouldn't need the
calculator.

**Almost none of them model the KV cache** — the memory the model needs to hold
your conversation. It grows linearly with context length, and it is not a rounding
error:

| Model | Weights | KV cache at 128k |
|---|---|---|
| Gemma 4 12B | 7.0 GB | **10.2 GB** |
| Llama 3.3 70B | 42.5 GB | **42.2 GB** |

Llama 3.3 70B needs as much memory for your context as for the model itself. This
is why a model that fits comfortably for chat dies on a long document — same
model, same laptop, different job — and why a calculator that only looks at file
size will confidently tell you it fits.

**The ones that do exist tell you *whether*, not *how fast*.** Those are different
questions and only one of them decides if you keep using it. 2 tokens/sec fits
perfectly and you will never open it again.

**And most treat mixture-of-experts models as dense**, which is wrong twice over.
Qwen3.6 35B-A3B occupies 21 GB but reads 1.8 GB per token. Score it as a 35B and
you'll rule out a model that would have run well.

## What this does instead

Seven questions, each with click-by-click instructions for finding the answer —
which Apple menu, which Device Manager entry. **No terminal.** Then:

- **Fit and speed together**, as a range, per model, at *your* context length
- **KV cache modelled explicitly**, which is why it asks how much text you use at once
- **Mixture-of-experts handled on both axes** — memory follows total parameters,
  speed follows active ones
- **Partial offload as a continuous split**, not a cliff, because "70% on the GPU"
  is the situation most laptops are actually in
- **Can you *train* it, not just run it** — per model, whether LoRA or 4-bit LoRA
  fits, with the GPU, RAM and disk each would take, and which cloud tier to use
  when the laptop can't
- **Every number checkable.** The methodology section labels each input published,
  calculated or editorial, and publishes the measured-vs-predicted table so you can
  see the residuals instead of taking the constants on faith

## Fine-tuning, too

Running a model and training one are different questions with different arithmetic,
so the page answers both. Training can't use the Q4_K_M file at all — it's an
inference format — so this works from parameter count:

```
full FT = p × 16 bytes   (bf16 weights + grads + fp32 Adam + master)
LoRA    = p × 2  + adapter + activations
QLoRA   = p × 0.55 + adapter + activations
disk    = p × 2 GB       for BOTH paths
```

Three things fall out that people consistently get wrong:

- **Full fine-tuning is impossible on any laptop, for any model.** Even a 1.7B needs
  ~27 GB. A 12B needs ~190 GB.
- **The disk figure is the trap.** A 4-bit run still downloads the bf16 weights
  first, so fine-tuning a 27B needs **54 GB of free disk** even though it only
  occupies 19 GB of VRAM.
- **QLoRA is CUDA-only.** bitsandbytes has no Metal backend, so on Apple Silicon
  the 4-bit path is MLX's own implementation — named separately, because it's a
  different stack. Intel Macs, AMD laptops and machines with no dedicated card get
  told there's no local path rather than given a fake number.

Unlike inference, training doesn't split across GPU and system memory, so the
verdict is binary and depends heavily on which machine you're on. These figures are
**calculated, not measured** — the bytes-per-parameter constants are standard, but
the activation heuristic has no validation anchors behind it, and the page shows
them in a quieter visual register to say so.

## How it works

Single-stream decoding is bound by memory bandwidth, not compute: producing one
token means reading every active parameter out of memory, once. A faster processor
barely helps. So the whole estimate is arithmetic you can check:

```
need      = weights + (kv_per_1k × context_k) + overhead

sec/token = (f × W) / (BW_gpu × E_gpu × M_gpu)
          + ((1 − f) × W) / (BW_cpu × 0.55 × M_cpu)
```

`f` is the fraction of weights resident in VRAM; `W` is the bandwidth-relevant
weight bytes — active parameters for MoE. `E` and `M` are the fraction of
theoretical peak an inference runtime actually reaches, calibrated against measured
throughput.

**Mixture-of-experts needs a different constant on each side of that split, and
they point in opposite directions.** Resident on a GPU, MoE runs *below* the
bandwidth ceiling: too few bytes per token to saturate the bus, so expert gathers
and launch overhead dominate. Offloaded to the CPU it runs *above* what a uniform
split predicts, because llama.cpp moves expert layers specifically and keeps
attention on the card. An earlier version of this page used one constant for both,
fitted only on offload measurements — and overstated a fully-resident MoE on an
M4 Max at 227 tokens/sec against a reported 70.

That kind of error is exactly what the published anchor table exists to catch.

## What it deliberately does not model

Prompt processing (compute-bound, different rules, and for long documents it often
dominates the wait), thermal throttling, quantisation quality, layer granularity,
training, speculative decoding, batching, multi-GPU.

The reported range is ±30%. **Prefer the range to the midpoint** — the midpoint is
the least interesting part of the answer.

## Status

Model data current as of 25 July 2026, estimator version 0.6.0. New models ship
continually; if that date is more than a few months old, treat the list as stale.

Not every figure is held to the same standard, and the page says which is which.
The validation table marks the one anchor that is a reported figure rather than a
first-hand measurement — it is also the only thing constraining MoE speed on
machines where the model fits entirely in fast memory, which is every Apple Silicon
result. Check provenance before trusting a specific number.

**Measurements are the most useful contribution.** Open an issue with machine
specification, runtime version, model build and context length. The gaps that
matter most:

- anything on AMD
- anything at long context, where the cache dominates
- any sustained rather than burst figure
- a first-hand fully-resident mixture-of-experts measurement

## Running it

It is one file. Open `index.html` in a browser, or serve the directory statically.
There is no build step and nothing to install.
