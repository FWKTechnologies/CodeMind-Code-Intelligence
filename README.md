# CodeMind
### A Graph-Native Neural Architecture for Code Understanding and Generation

---

## Overview

CodeMind is a neural system for understanding and generating source code, built and
trained from the ground up around a **graph representation of code** rather than raw
text. Instead of treating a program as a sequence of tokens, CodeMind represents it as
a unified graph combining its syntax tree, control flow, data flow, program dependence,
and call structure — and learns directly over that structure using a **Heterogeneous
Graph Transformer (HGT)**.

This is not a fine-tuned language model and not a prompt layer on top of one. The code
understanding and code generation components are trained end-to-end on graph-structured
data.

Natural-language interaction is handled separately. CodeMind's own conversational
language model is still under development, so the system currently pairs its
graph-native code model with **Qwen2.5**, an existing pretrained model used only for
fluent dialogue. The two remain fully independent — connected only through small
trained bridge modules that translate between their vector spaces — so Qwen can be
swapped out for CodeMind's own language model later without any change to the code
brain itself.

---

## Key Capabilities

- **Multi-language code understanding** across roughly 60 languages, using one shared
  set of trained weights rather than a separate model per language.
- **Structural code generation** — code is produced as a sequence of structured
  actions rather than free-form text, then rendered and validated.
- **Dual-brain design** — a dedicated understanding network and a dedicated editing
  network, trained in sequence with safeguards against the second overwriting what the
  first learned.
- **Automated validation and reward** — every generated code sample is checked for
  syntax, safety, and quality before being trusted or scored.
- **Reinforcement learning beyond imitation** — generation quality is shaped by a
  reward signal that accounts for novelty, correctness, and calibrated confidence.
- **Scales from a single function to full multi-file, multi-language projects.**

---

## Model Size

Only the graph-based components below are trained by this project. Qwen2.5 is loaded
pretrained and frozen, and is not included in this figure.

| Component | Layers | Approx. Parameters |
|---|---|---|
| Understanding network (Brain A) | 8 | ~3.10B |
| Editing network (Brain B) | 7 | +~156M |
| Generation decoder | 9 | +~33M |
| **Total** | | **≈ 3.29B** |

Model capacity is controlled by a small number of configuration values — hidden
dimension, layer depth per component, and the maximum graph size processed per
sample — each chosen against a specific hardware budget rather than picked
arbitrarily. When memory, not parameter count, is the constraint, the per-sample graph
size cap is the more efficient lever, since it scales memory roughly linearly without
changing what the model can represent.

Qwen2.5 is available at several size tiers, from a lightweight variant suited to
development hardware up to a larger mixture-of-experts tier reserved for future
multi-GPU deployment.

---

## Architecture

```
Layer 6   Serving              OpenAI-compatible API + command-line interface
Layer 5   Cross-Model Bridge   Vector-space translation between the code brain and the language model
Layer 4   Language Model       Qwen2.5 (interim — pending CodeMind's own language model)
Layer 3   Learning Loop        Unified training loss + reinforcement learning (PPO)
Layer 2   The Brain            Dual HGT networks (understanding + editing) and the generation decoder
Layer 1   Code Representation  Source code → one unified graph, across languages
Layer 0   Platform             Precision management, caching, logging
```

Each layer depends only on the layer(s) beneath it.

### Layer 0 — Platform

Training runs at a fixed numeric precision (BF16) throughout, so results are
reproducible rather than varying between runs. Data that doesn't fit in memory spills
to disk using a serialization format chosen specifically to avoid the arbitrary-code
risks of naive object pickling. A single logging system records every significant
event to console, memory, and a persistent per-run log file.

### Layer 1 — Code Representation

Every supported language is converted into the same graph schema — roughly a dozen
node types, with typed edges that connect *what the code says* (syntax) to *what it
does* (control and data flow), so a single attention step can reason across both.
Python receives a fully precise, compiler-grade conversion; other languages are
converted through language-family-aware heuristics into the same schema. This is a
deliberate trade-off — full precision where it's cheap to get, broad coverage
everywhere else, all learned by one shared model.

Multi-file and multi-language projects are parsed file by file and then linked into
one connected graph using real cross-file relationships: imports, API calls between
services, shared configuration.

### Layer 2 — The Brain

Code elements are encoded through hashing and learned embedding tables rather than a
conventional subword tokenizer, which keeps large graphs from causing an unmanageable
blow-up in sequence length. The graph transformer applies attention parameters
specific to each node and edge type, letting one network distinguish a data-flow
relationship from a plain syntactic one without a separate sub-network for each.

The **dual-brain** design trains an understanding network first, then trains a
lighter, dedicated editing network on top of its learned representation — with a
bridging mechanism that actively guards against the editing network overwriting what
the understanding network already knows, measured by a knowledge-retention score that
feeds directly into training reward.

Generation-time routing uses a small mixture-of-experts layer, with an auxiliary loss
that keeps usage spread across experts instead of collapsing onto just one or two.
Oversized graphs are reduced toward a fixed size budget through a fast, multi-stage
process before reaching the network, so processing cost stays predictable regardless
of input size.

Code generation itself is autoregressive: a single semantic vector representing the
model's understanding of the target seeds a decoder that produces a structured
sequence of actions rather than raw text, rendered into code and checked. Generation
has a hard length ceiling — past that point, the model **abstains** rather than
continuing to guess.

### Layer 3 — Learning Loop

A unified training loss combines several objectives at once: contrastive
representation learning, semantic alignment, self-predicted quality accuracy,
embedding regularization, and the reinforcement-learning objective below — each with
its own weight based on how much it should shape training.

Reinforcement learning (PPO) shapes generation quality beyond supervised imitation.
Reward comes from several signals together: novelty of the generated pattern,
calibration (confidently wrong output is penalized harder than honest uncertainty),
automated validation of the generated code, and the dual-brain knowledge-retention
score. A per-sample mastery tracker shifts training time away from material the model
has already learned well and toward material it hasn't.

Training data streams from disk rather than requiring the full dataset in memory, and
graph parsing runs ahead of the training loop in background workers, so GPU time isn't
lost waiting on CPU-side preprocessing.

### Layer 3 — Validation

Before any generated code is trusted, shown to a user, or scored for reward, it passes
through a deterministic, rule-based validation pipeline: static syntax and safety
analysis (flagging risky patterns like system-command execution without necessarily
blocking them, since real training data legitimately contains such patterns),
placeholder/non-functional-output detection, and a composite 0–100 quality score
spanning syntax correctness, static analysis, structural coverage, semantic
similarity, execution outcome, and mastery.

### Layer 4–5 — Language Model and Cross-Model Bridge

CodeMind's own language model is still in development. In the interim, the system
loads Qwen2.5 — pretrained and frozen, never fine-tuned by this project — solely to
produce fluent conversational output. The two models are never loaded together during
training; training runs in strictly separated modes (train the code brain, train the
language model, or run both together at inference time only).

To hand understanding from the code brain to Qwen, the system projects the code
brain's semantic embedding directly into Qwen's input-embedding space as a short
sequence of vectors, rather than converting it into a text description and inserting
that into a prompt. This avoids both the extra token cost and the information loss a
text-based handoff would introduce — and it's the same interface point where
CodeMind's own language model will plug in once ready.

### Layer 6 — Serving

The system exposes an OpenAI-compatible HTTP API — chat completions, model listing,
usage stats — with request-size limits and rate limiting, alongside a command-line
interface for direct use: checking status, understanding a file, generating code,
validating code, analyzing a full project, or running training. Streamed code output
is safety-checked as it's produced, so issues surface immediately rather than only
after a full response completes.

---

## Design Principles

1. **Abstain rather than bluff.** Explicit uncertainty and stopping paths exist in
   generation, and are directly rewarded during training rather than left implicit.
2. **Structure over raw text, wherever it's affordable.** Graph representation instead
   of token streams; projected vectors instead of paraphrased text between models.
3. **One shared model across languages**, not one model per language — at the cost of
   a small, deliberate precision trade-off for the highest-coverage language.
4. **Security and reproducibility are engineering requirements, not afterthoughts.**
   Both the caching format and the fixed-precision training policy were chosen to
   close specific, identified failure modes.
5. **Every capacity decision is tied to a real hardware budget**, not chosen
   arbitrarily.
6. **Known issues and their fixes stay documented alongside the fix itself** — not
   just the corrected end state — as part of maintaining the system over time.
