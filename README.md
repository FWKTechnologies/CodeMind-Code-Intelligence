# CodeMind — System Architecture & Engineering Reference

> **Scope.** This is a concise technical reference for how CodeMind is actually built —
> representation, architecture, training objectives, and runtime behaviour — with
> concrete parameter values pulled directly from the current source, not a marketing
> summary. No source code is reproduced verbatim; every mechanism is described in
> prose, with short illustrative pseudocode written for this document. Numbers in this
> revision were re-checked against the current `CodeMindConfig` (v75) — a few figures
> in earlier drafts of this document (hidden size, decoder depth, FP8) had drifted from
> the code and are corrected here.

---

## 1. What CodeMind Is

CodeMind is a **trained-from-scratch neural system for code understanding and code
generation**, built on a **graph representation of source code** — not a pretrained LLM
fine-tuned for code, and not a prompt wrapper around one. Its core is a **Heterogeneous
Graph Transformer (HGT)**, following Hu et al. (WWW 2020), operating over a unified
schema that represents a program simultaneously as its AST, Control-Flow Graph, Data-Flow
Graph, Program Dependence Graph, Call Graph, and System Dependence Graph — together a
**Code Property Graph (CPG)**.

It pairs this custom code-model with a second, independent, off-the-shelf system —
**Qwen2.5** — used strictly for fluent natural-language interaction. The two networks are
never merged; they stay architecturally separate and are joined only through small,
learned **bridge modules** that translate between their two vector spaces. This
separation — a graph-native *code* brain and a text-native *language* model, joined by
projection bridges instead of prompt concatenation — is the organising idea behind
almost every other design choice below.

The system also includes a **curiosity-driven, calibration-aware PPO** reinforcement
loop shaping generation beyond supervised imitation; a **streaming training pipeline**
now tuned for a single H100-class GPU (see §2); and a **local, OpenAI-compatible HTTP
server** for IDE integration.

---

## 2. Model Size — Exact Numbers

Only the graph brain, the dual-brain extension, the generation decoder, and the
cross-model bridges are *trained* by this project. Qwen2.5 is loaded off-the-shelf and
frozen — its parameters are separate and not part of the figure below.

**Custom-trained core ≈ 3.29 B parameters**, given the current default config
(`hidden_dim=1664`, `num_heads=64`):

| Component | Layers | Approx. params | Notes |
|---|---|---|---|
| Brain A (understanding HGT) | 8 | ~3.10 B\* | includes token encoder, fusion, PPO head, Qwen bridges — see breakdown below |
| Brain B (editing HGT, dual-brain) | 7 | +~156 M | one HGT layer ≈ `4·d² (qkvo, 14 node types) + w_rel (21 edge types) + LayerNorm` at `d=1664` |
| Generation decoder | 9 | +~33 M | 1664-dim, 16 heads, 2,048-token action sequence ceiling |
| **Total** | | **≈ 3.29 B** | |

\*The 3.10 B figure is the documented anchor for an 8-layer Brain A at `hidden_dim=1664`
(cross-checked two ways in the source: a Chinchilla-style token-budget estimate against
the actual training corpus size, and a VRAM-ceiling estimate for an 80 GB H100 under
BF16-weights + BF16-grad + FP32-master-weight + FP32 Adam(m,v)). It is **not** a value
baked into this document from a guess — the code computes and logs the *real* number
every time a `CodeMind` instance is constructed:

```
CodeMind total parameters (encoder+brain(s)+fusion+decoder+bridges) =
  X,XXX,XXX,XXX (X.XXXB) | hidden_dim=1664 brain_a_layers=8 brain_b_layers=7
```

Treat that log line — not this document, and not the design-time estimate above — as
the ground truth for a specific run, since it sums every module actually constructed
for that config (brain B, dual-brain bridge/router, and both Qwen bridges are only
added to the total when enabled).

### 2.1 Sizing knobs, if you want to change the model size

Every field below lives in `CodeMindConfig` — change it in exactly one place, nowhere
else:

| Field | Default | Effect on size | Effect on VRAM/step |
|---|---|---|---|
| `hidden_dim` | 1664 | **Quadratic** — dominates total params (must stay divisible by `num_heads`) | Quadratic |
| `num_hgt_layers` (Brain A) | 8 | Linear, ~390M/layer at d=1664 | Linear |
| `brain_b_layers` | 7 | Linear, ~156M/layer | Linear |
| `decoder_layers` | 9 | Linear, ~33M/9-layer-equivalent slice | Linear |
| `num_heads` | 64 | None directly (must divide `hidden_dim` exactly) | Minor |
| `max_nodes` | 4096 | None | **Roughly linear** — this is the real lever for VRAM-per-step, independent of parameter count |
| `flyprompt_experts` | 6 | Small, one linear expert head each | Small |

If you only have VRAM headroom to spare (not parameter-count headroom), raise
`max_nodes` or batch size before touching `hidden_dim` — it is the only knob here that
changes memory *without* changing what the model itself can represent structurally.
If you want a smaller model for a smaller GPU, `hidden_dim` must drop in steps that
still divide evenly by `num_heads` (e.g. 1664 → 1600 at 25 heads, or reduce
`num_heads` first) — the source comments record the actual two anchor points used to
extrapolate this project's own sizing (`1664 → ~3.10B`, `1792 → ~3.6B`), which is a
reasonable way to interpolate/extrapolate your own target size before verifying it
against the real logged number.

### 2.2 The off-the-shelf language model, for scale

| Tier | Model | Params | Role |
|---|---|---|---|
| `light` | Qwen2.5-1.5B-Instruct | ~1.5 B | fast dev/testing |
| `14b` | Qwen2.5-14B-Instruct | ~14.7 B | current primary conversational tier |
| `122b-moe` | Qwen3.5-122B-A10B | 122 B total / ~10 B active | documented placeholder, not runnable on the current single-GPU target |

These are downloaded, quantised, and never trained by this project — they're listed
here only so the ~3.29 B figure above isn't misread as the *whole* system's footprint.
The two systems don't even share memory at the same time during training (§12.1).

---

## 3. The Codebase

| File | Role |
|---|---|
| `CodeMind.py` | The core brain: graph representation/parsers, the HGT network, generation decoder, static/integrity validation, curiosity-PPO loop, unified loss, streaming training pipeline, hardware/precision/cache management, large-file/multi-project handling, and a standalone CLI. Trainable and usable without Qwen. |
| `codemind_brain_dual.py` | Extensions on top of the core brain: Brain B, `FlyPromptRouter` (MoE-style routing), `AntiForgetBridge`, `GraphCollapser`, `DualBrainOrchestrator`, and the two neural bridges to Qwen's space (`QwenBrainBridge`, `QwenInstructionBridge`). |
| `CodeMind-Qwen2_5-14B.py`\*\* | The language-model side in isolation: the model-tier registry, quantised/offloaded loading for Qwen2.5, and the two training entry points (`train-language`, `train-codemind`) — deliberately never loading both large models at once. |
| `CodeMind-Run.py`\*\* | The inference-time integration layer: `CodeMindRunner`, `StructuredEmbeddingBridge`, `TokenBudgetManager`, IDE-context handling, in-stream safety scanning, and `serve_api()`. |

\*\*These two files weren't part of the most recent code review pass — their
description here is carried over unchanged from the prior revision of this document
and should be re-verified against source before being treated as current, the same way
§2's numbers were.

A fifth module, `codemind_paths.py`, supplies filesystem locations to all four files
and isn't covered separately here.

---

## 4. Layered Architecture

```
 Layer 6 — Serving & CLI        CodeMindRunner · serve_api() · CodeMind.py CLI
 Layer 5 — Cross-Model Bridges  QwenBrainBridge · QwenInstructionBridge · StructuredEmbeddingBridge
 Layer 4 — Language Model       Qwen14BModel (tiered, quantised, offload-aware)
 Layer 3 — Learning Loop        CodeMindLoss · CuriosityPPOReward · PPOAgent · TrainingPipeline · QualityEngine
 Layer 2 — The Brain            HGTBrain (A+B) · FlyPromptRouter · AntiForgetBridge · GraphCollapser · CodeDecoder
 Layer 1 — Code Representation  MultiLanguageParser · CPGBuilder / GenericASTBuilder · PolyglotLinker → CodeGraph
 Layer 0 — Platform             PrecisionManager · SSDCache (safetensors) · hardware detection · MindLog
```

Each layer consumes only the layer(s) beneath it. The rest of this document proceeds
bottom-up, following the same path one piece of source code actually travels end to end.

---

## 5. Layer 0 — Platform

**Precision.** `detect_hardware()` builds a `HardwareProfile`; `PrecisionManager` then
locks compute at **BF16** for the run, with TF32 matmul/cuDNN enabled on Ampere-or-newer
GPUs. An earlier FP8-storage code path (`fp8_locked`) was removed entirely — it never
actually engaged during training or checkpointing, so BF16 autocast was always the real
compute path in practice; removing the dead branch changes nothing about run behaviour.
Precision is fixed for the whole run rather than dynamically downgraded, so a run either
completes at one known, reproducible precision or fails loudly.

**`SSDCache`.** Under RAM/VRAM pressure, large intermediates spill to a local SSD-backed
cache. This cache used to serialise objects with `pickle` (via `torch.save`) — a genuine
arbitrary-code-execution risk if a cache file is ever tampered with or shared. It now
uses `safetensors` for tensor data and plain JSON for scalars, so no path in this layer
deserialises pickle at all.

**`MindLog`.** One logger threaded through the whole system, writing every notable event
to the console, a 5,000-event in-memory ring buffer, and an append-only JSONL file per
run, tagged by `LogCategory` for downstream filtering.

---

## 6. Layer 1 — Code Representation

Nothing downstream reasons about code as a flat token sequence; everything is first
converted to a typed, heterogeneous graph.

**`CodeGraph` schema.** 14 node types (`ast, cfg, dfg, pdg, call, sdg, token, func,
block, expr, stmt`, plus polyglot markers `lang, api, config`). Edges are typed
`(source, relation, target)` triples, including **cross-view edges** — `ast→cfg`,
`cfg→pdg`, `dfg→pdg`, `token→ast`, `func→ast` — that let attention move between "what the
syntax says" and "what actually happens" in a single message-passing step, plus a
polyglot set (`cross_lang`, `hosts`, `binds`, `wires`, `same_family`, `coexists`) for
multi-file/multi-language merging.

**Two builder strategies.** `CPGBuilder` gives Python a full, compiler-grade CPG via the
stdlib `ast` module. `GenericASTBuilder` handles every other language via family-aware
heuristic parsing (`c_like`, `js_like`, `functional`, `data`, `markup`, `config`,
`shell`, `jvm`, `script`, `other`) into the same node/edge vocabulary — a deliberate
accuracy/coverage trade-off: Python gets structural precision, everything else gets
"enough structure for a GNN to learn from."

**`LanguageRegistry`.** ~60 registered languages, each with a stable integer slot id and
family slot encoded directly on every node it produces — this is what lets one shared
set of weights condition on language identity instead of needing one encoder per
language.

**Multi-file/multi-language projects.** `MultiLanguageParser.parse_project()` builds one
graph per file; `PolyglotLinker` merges them, adding cross-file edges for real
relationships (an HTTP route invoked from a client `fetch`, an import dependency, a
config key wiring two files together) so a polyglot project is reasoned about as one
connected structure.

---

## 7. Layer 2 — The Brain

**Token encoding.** `TokenEfficientEncoder` turns each node label into a vector via
hashing + learned embedding tables rather than a subword tokenizer, avoiding the
sequence-length blow-up a conventional tokenizer would cause on graph-structured input.

**`HGTBrain`.** Stacked `HGTLayer`s with attention parameters specific to each
node/edge *type* rather than shared uniformly — the actual mechanism (Hu et al. 2020)
letting one network condition its reasoning on "this edge is a `dfg→pdg` cross-view
edge" versus "this edge is plain AST `child`" without separate per-type sub-networks.

**Generation.** `GraphActionCodec` deliberately routes *every* language, Python
included, through `GenericASTBuilder` for the generation path — even though Python's
*understanding* path (`HGTBrain`) still uses the precise `CPGBuilder`. The reason: using
Python's fine-grained AST representation for generation while every other language used
the coarser statement-level one would mean the shared decoder weights are learning two
structurally incompatible granularities from the same weights. The trade-off: Python's
generation loses `ast.unparse()`'s automatic well-formedness guarantee, in exchange for
one decoder that learns generation uniformly, with correctness enforced downstream by
the validator (§9) and PPO reward instead of by construction.

`CodeDecoder` is the autoregressive Transformer doing the actual decoding, conditioned
only on a small number of *memory tokens* (default 8) derived from the brain's single
semantic embedding of the prompt — the prompt text itself is never re-tokenized and fed
to this decoder. Current sizing: 1664-dim, 9 layers, 16 heads, 2,048-token action-sequence
ceiling, with a hard 2,600-action generation ceiling so the model **abstains** rather
than continuing to guess once a generation has clearly run too long.

```
seed    = brain.understand(prompt)             # one semantic vector, computed once
state   = decoder.init_from(seed)
actions = []
while not finished and len(actions) < action_ceiling:
    candidate = decoder.step(state, actions)
    if not codec.is_well_formed(actions + [candidate]):
        candidate = codec.best_valid_alternative(actions)
    actions.append(candidate)
code   = renderer.render(actions, language)    # not guaranteed syntactically valid
result = validator.validate(code, language)    # always re-checked, see §9
```

**Batching.** `GraphTensorizer` concatenates several graphs into one block-diagonal
heterograph per batch (the same principle as PyG's `Batch.from_data_list`), which is
what lets `HGTBrain` process several samples per forward pass instead of one at a time —
previously the main cause of low GPU utilisation.

---

## 8. Layer 2 (continued) — Dual-Brain Architecture

**Brain A / Brain B.** Brain A (8 HGT layers) specialises in understanding; Brain B (7
layers, intentionally lighter) specialises in editing. They train **sequentially**, not
jointly — Brain A's phase completes first, and its representation is carried forward
into Brain B's phase rather than discarded. `DualBrainOrchestrator` owns this two-phase
lifecycle.

**`AntiForgetBridge`** guards against catastrophic forgetting when Brain B starts
training: it fuses A's and B's embeddings via a sigmoid-gated mixture (`gate·a +
(1-gate)·out(cat)`, LayerNorm-stabilised), and adds an **EWC-lite penalty**
(`ewc_lambda = 0.12`) pulling the fused embedding back toward a running anchor of
Brain A's own past embeddings, weighted by a diagonal importance term. It also computes
a **`bridge_reward`** (cosine similarity between A's embedding and the fused output,
EMA-smoothed) that feeds directly into the PPO reward — poor knowledge preservation is
actively penalised during training, not just logged.

**`FlyPromptRouter`** — a small MoE-style router: top-2-of-6 experts selected by
softmax over gated, `task_id`-biased logits. Each `TemporalEnsembleExpert` keeps its own
EMA of past outputs (decay 0.95, 12% blended back in) as a second, expert-local
anti-forgetting mechanism. A **load-balance loss** (`0.01 × Σ(prob − mean_prob)²`)
discourages routing collapse onto one or two experts and is wired as a live,
non-detached term added into the total loss (an earlier version computed it but
discarded the gradient, so it never actually constrained anything).

**`GraphCollapser`** bounds graph size to `max_nodes` (4,096) in three linear-time
stages before the brain sees it: exact-duplicate merge (protecting structurally
important node types), linear-chain collapse (removing pass-through single-in/single-out
nodes that aren't structurally important), and score-and-trim (keeping only the
top-scoring nodes by a structural/keyword/degree score if still over budget).

---

## 9. Layer 3, Part 1 — Static Analysis, Integrity, Validation

Every generated (or user-supplied) piece of code goes through a deterministic,
non-neural pipeline before it's trusted, shown to a user, or scored for RL reward.

- **`StaticAnalyzer`** — genuinely semantic for Python (real `ast.parse`, tracks
  imports/defs/argument bindings/`except...as` bindings to flag undefined names and
  unused imports), regex/balance-based for other languages. A shared `_check_safety()`
  flags `os.system`/`subprocess.*`/`eval`/`exec`/`rm -rf`/`DROP TABLE` as **critical**
  and `__import__`/unconditional `DELETE`/`pickle.loads`/`ctypes.CDLL` as **warnings** —
  intentionally surfaced rather than blocked outright, since training data legitimately
  contains these patterns.
- **`IntegrityChecker`** — flags copied-prompt output, hollow placeholder code (`pass`,
  `...`, bare `TODO`), prose bleeding into code (English and Thai hedging markers), and
  comment-only fake solutions. Produces a `[0,1]` integrity score.
- **`CodePurityChecker`** — a lighter, independent heuristic for what fraction of lines
  "look like code" vs. prose, used as a multiplicative floor on the overall score.
- **`CodeValidator`** combines all three (plus optional sandboxed execution for Python)
  into one composite score: `1.0 − 0.4·(errors) − 0.1·(warnings) − 0.3·(safety issues)`,
  then scaled by the integrity score and the purity floor.

---

## 10. Layer 3, Part 2 — Reinforcement Learning

`CuriosityPPOReward` combines several signals into one scalar reward per generation:
**novelty** (unseen-by-content-hash patterns), **calibration** (penalising
confident-but-wrong output more than honest uncertainty), **integrity/validation**
(from §9), and the `bridge_reward` from §8. `PPOAgent` is a standard clipped-objective
PPO implementation with GAE, an entropy bonus, and a KL penalty against a frozen
reference policy, updated over **batched tensors only** — metrics are gathered into a
single tensor and synchronised to Python scalars once per PPO-update epoch rather than
per metric, since each `.item()` call forces a CUDA device→host sync that stalls the
launch queue.

`ExperienceBank` tracks a running **mastery** score per distinct sample (by SHA-256 of
its text): `new = 0.9·old + 0.1·max(0, reward)`. Once mastery passes 0.85, that sample is
skipped or de-prioritised in later passes, concentrating training time on
not-yet-mastered material.

---

## 11. Layer 3, Part 3 — The Unified Loss and Training Pipeline

```
L_total = 0.30·L_contrastive + 0.25·L_semantic_align + 0.20·L_validate
        + 0.10·L_graph_reg  + 0.15·L_ppo_policy
```

- **`L_contrastive`** — InfoNCE-style, cosine similarity at temperature 0.07 against a
  FIFO bank of 64 recent (detached, CPU-resident) target embeddings used as in-batch
  negatives. Worth recording precisely: an earlier version trained on one
  `(source, target)` pair per step with **zero negatives** — softmax of a single logit
  is always exactly 1.0, so this term's gradient was provably zero at every step,
  despite being the single largest weight (0.30) in the loss. The negative bank is the
  fix.
- **`L_semantic_align`** = `1 − cos(source_embedding, target_embedding)`.
- **`L_validate`** — MSE between the brain's own predicted validation score and the
  actual `CodeValidator` score for that sample.
- **`L_graph_reg`** — a small L2 penalty (×0.001) on per-node-type embedding magnitude.
- **`L_ppo_policy`** — the §10 PPO loss, folded into the same backward pass.

**`QualityEngine`** scores 0–100 across six weighted dimensions (syntax 25, static 20,
graph coverage 15, semantic similarity 20, execution 10, mastery 10) — training targets
**quality ≥ 85** with patience 3, rather than any fixed loss threshold.

**Streaming pipeline.** `DataConnector` streams from `.jsonl`/`.json`/plain-text sources
through a 20,000-sample shuffle buffer, backed by an SSD shard cache (20 GB budget,
LRU-by-mtime eviction) so nothing requires the full dataset resident in memory.
`_ParsePrefetcher` runs graph parsing in background worker processes ahead of the
training loop, guarded by a 20-second per-batch timeout (reduced from 120s — the longer
timeout was the main cause of the GPU sitting idle waiting on CPU-side parsing).

**Checkpointing.** Weights save via `safetensors`, metadata as plain JSON — the same
pickle→safetensors migration as `SSDCache` (§5), for the same reason.

---

## 12. Scaling Beyond a Single File

- **`LargeCodeHandler`** — for 10K–500K+ line files: splits on function/class
  boundaries, processes chunks independently, and blends 0.7×current + 0.3×previous
  chunk embedding into each next chunk's conditioning, so cross-chunk coherence lives in
  the vector representation rather than in visible prompt text.
- **`MultiProjectConnector`** — same principle at project scale (~100–2,000 files):
  indexes files without reading contents first, batches by a RAM budget, merges
  incrementally via `PolyglotLinker`, and applies `GraphCollapser` before the brain ever
  sees the accumulated graph.

---

## 13. Layer 4 & 5 — The Language Model and Cross-Model Bridges

CodeMind's own decoder produces code from a semantic vector; it does not handle fluent
natural language and was never meant to. That's Qwen2.5's job, in its own file,
specifically so training never needs both large models resident at once: **three modes,
never combined** — `train-language`, `train-codemind`, and `run/chat` (loads both, but
lazily, and never trains either while doing so).

`TIER_REGISTRY` declares `light` (1.5B, dev/testing), `14b` (current primary,
8-bit-quantised on-GPU by default, 4-bit as fallback, layer-sharded SSD offload as a
last resort via `accelerate`), and `122b-moe` (Qwen3.5-122B-A10B — a documented
placeholder architected for external vLLM/SGLang serving, not runnable on the current
single-GPU target).

**Bridging CodeMind → Qwen.** Rather than turning CodeMind's understanding into a text
description and pasting it into Qwen's prompt (lossy, token-expensive), `QwenBrainBridge`
/ `StructuredEmbeddingBridge` project CodeMind's semantic embedding directly into Qwen's
input-embedding space as a short sequence of virtual **soft-prompt tokens** — the same
principle as prefix-tuning. `QwenInstructionBridge` runs the opposite direction, feeding
a cached Qwen instruction embedding into CodeMind.

**`TokenBudgetManager`** caches recent "brain packs" (a code snippet's full
understanding output) by content hash in a 64-entry LRU, and adaptively sizes Qwen's
`max_new_tokens` request by how much code/context is actually present in the turn.

---

## 14. Layer 6 — Serving and CLI

**`CodeMindRunner`** lazily loads CodeMind and Qwen only when first needed, can unload
Qwen after an idle timeout (default 300s) to free VRAM, and exposes `explain()`,
`chat()`, `code()`, `code_and_explain()`, `status()`. It performs **in-stream safety
scanning** — each streamed code block is validated as it closes, surfacing a warning
immediately rather than only after the full response is shown. If no Qwen weights are
available locally, `DemoResponder` produces clearly labelled demo output instead of
failing outright.

**`serve_api()`** — a `ThreadingHTTPServer` exposing an OpenAI-compatible
`POST /v1/chat/completions`, `GET /v1/models`, `GET /v1/stats`, with an optional API-key
check, a 2 MB request cap, and a 90-req/60s rate limit.

**`CodeMind.py` CLI** — `status`, `understand`, `generate` (`--large` for chunked
generation, `--force` to bypass the abstain guard), `validate`, `project`, `train`,
`demo`, `cache`, and `repl` (also the default action with no subcommand, so a first run
lands in something interactive).

---

## 15. Design Philosophy

1. **Abstain over bluff.** An explicit abstain path exists in generation, validation,
   and — quantitatively — as a calibration/honesty term in the PPO reward that
   penalises confident-but-wrong output harder than honest uncertainty.
2. **Structure over text, wherever affordable.** Graph representation over token
   streams; projected vectors over paraphrased text between models; shape-constrained
   generation over free sampling.
3. **One learned system across languages**, not one model per language — reworked at
   the cost of a Python-specific guarantee (§7), specifically to avoid two languages
   being represented at incompatible granularities within shared weights.
4. **Security and reproducibility as first-class engineering.** The pickle→safetensors
   migration and the precision-locking policy both respond to concrete named failure
   modes (arbitrary code execution; irreproducible numerics), not generic boilerplate.
5. **Every sizing decision ties to a named hardware budget** (§2) — model dimension,
   layer counts, and graph node ceiling are all justified against a specific target GPU,
   and the large models are kept in separate files so no training path ever needs both
   in memory.
6. **The "why," including past mistakes, stays in the documentation.** Several
   components carry inline notes on a bug that was found, why it mattered, and the fix —
   the zero-gradient contrastive loss, the load-balance loss that was computed but never
   backpropagated, the `dir(__builtins__)` bug, a checkpoint path used as a directory by
   mistake. Keeping this history alongside the fix, not just the corrected end state, is
   itself a deliberate practice worth continuing.

---

## 16. Terminology Glossary

| Term | Meaning here |
|---|---|
| **CPG** | A single graph unifying AST, CFG, DFG, PDG, and call-graph views |
| **HGT** | Heterogeneous Graph Transformer (Hu et al., WWW 2020) — type-specific attention |
| **Brain A / B** | Understanding vs. editing HGT sub-networks |
| **AntiForgetBridge** | Fuses A+B and applies an EWC-lite penalty to prevent B overwriting A |
| **FlyPromptRouter** | MoE-style router with a load-balance auxiliary loss |
| **GraphCollapser** | Reduces an oversized graph toward `max_nodes` in three stages |
| **CodeDecoder** | The autoregressive Transformer producing the action sequence |
| **Soft prompt** | Dense vectors injected directly into an LM's input-embedding space |
| **Curiosity / novelty** | Reward term for not-previously-seen (by hash) code patterns |
| **Calibration** | How well stated confidence tracks actual correctness |
| **Mastery** | Running per-sample score deciding whether to keep training on it |
| **PPO** | Clipped-objective RL with GAE, entropy bonus, KL penalty vs. a frozen reference |
| **EWC** | Elastic Weight Consolidation — penalising drift from an earlier learned state |
| **Block-diagonal batching** | Combining graphs into one heterograph per forward pass |

---

## 17. Quick File Map

| To understand or change… | Look in |
|---|---|
| Language parsing into a graph | `CodeMind.py` — `LanguageRegistry`, `GenericASTBuilder`, `CPGBuilder` |
| The core network | `CodeMind.py` — `HGTLayer`, `HGTBrain` |
| Generation / decoding | `CodeMind.py` — `GraphActionCodec`, `CodeDecoder`, `GraphRenderer` |
| Safety/correctness checks | `CodeMind.py` — `StaticAnalyzer`, `IntegrityChecker`, `CodePurityChecker`, `CodeValidator` |
| RL reward | `CodeMind.py` — `CuriosityPPOReward`, `PPOAgent` |
| The unified loss | `CodeMind.py` — `CodeMindLoss` |
| Model size / hardware sizing | `CodeMind.py` — `CodeMindConfig` (see §2) |
| Training loop / streaming | `CodeMind.py` — `TrainingPipeline`, `DataConnector`, `_ParsePrefetcher` |
| Two-brain order, anti-forgetting, routing, graph size | `codemind_brain_dual.py` — `DualBrainOrchestrator`, `AntiForgetBridge`, `FlyPromptRouter`, `GraphCollapser` |
| CodeMind → Qwen bridging | `codemind_brain_dual.py` — `QwenBrainBridge` |
| Loading/quantising Qwen | `CodeMind-Qwen2_5-14B.py` — `Qwen14BModel`, `TIER_REGISTRY` |
| Runtime chat / local API server | `CodeMind-Run.py` — `CodeMindRunner`, `serve_api` |

---

*End of document.*
