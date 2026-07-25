# Which LLM will run on your laptop?

A single-page estimator for local language models. Answer a few questions about
your machine — each with instructions for finding the answer, no terminal needed —
and get an estimate of which models fit, how fast they'll generate, and what each
is suited to.

**No build step, no dependencies, no external requests.** The whole thing is one
`index.html`: open it locally or serve it statically. Your answers are never
transmitted, stored or logged — there is no server to receive them.

## How it works

Single-stream decoding is bound by memory bandwidth, not compute: generating one
token means reading every active parameter out of memory once. So the estimate is
arithmetic you can check.

```
need      = weights + (kv_per_1k × context_k) + overhead
sec/token = (f × W) / (BW_gpu × E_gpu × M_gpu)
          + ((1 − f) × W) / (BW_cpu × 0.55 × M_cpu)
```

where `f` is the fraction of weights resident in VRAM and `W` is the
bandwidth-relevant weight bytes (active parameters for mixture-of-experts models).
The efficiency constants are calibrated against measured throughput; the
methodology section in the page publishes every anchor alongside what the page
predicts for it, so the residuals are visible rather than asserted.

Mixture-of-experts models need a different constant on each side of the split —
resident they run well below the bandwidth ceiling, offloaded they beat a uniform
split — and the page explains why.

## What it does not model

Prompt processing, thermal throttling, quantisation quality, layer granularity,
training, speculative decoding, batching and multi-GPU. The reported range is ±30%
around the central figure. Prefer the range to the midpoint.

## Status

Model data current as of 25 July 2026, estimator version 0.6.0. New models are
released continually — if that date is more than a few months old, the list is
stale.

Not every figure is held to the same standard, and the page says which is which:
the "Where each figure comes from" table labels each input as published,
calculated or editorial, and the validation table marks the one anchor that is a
reported figure rather than a first-hand measurement. Check the provenance before
trusting a specific number.

Measurements are welcome as issues — with machine specification, runtime version,
model build and context length. The gaps that matter most: anything on AMD,
anything at long context, any sustained rather than burst figure, and a first-hand
fully-resident mixture-of-experts measurement.
