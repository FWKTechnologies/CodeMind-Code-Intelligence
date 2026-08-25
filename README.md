# CodeMind — System Architecture & Engineering Reference

> **Scope and intent.** This document is a rigorous, prose-and-diagram description of
> how the CodeMind system is actually built: its representations, its neural
> architecture, its training objectives, its runtime behaviour, and the specific
> engineering trade-offs recorded directly in the implementation. No source code is
> reproduced anywhere in this document — every mechanism is described in technical
> prose, with concrete parameter values, formulas, and short illustrative pseudocode
> written specifically for this document. The goal is that a reader with no prior
> exposure to the implementation can reconstruct an accurate, detailed mental model of
> the system — not a marketing summary of it — and that a reader who *does* know the
> implementation can use this as a precise index back into it.

---

## 1. System Identity

CodeMind is a **purpose-built, trained-from-scratch neural system for code
understanding and code generation**, built on a **graph representation of source
code** — not a pretrained large language model fine-tuned for code, and not a prompted
wrapper around one. Its computational core is a **Heterogeneous Graph Transformer
(HGT)**, following Hu et al. (WWW 2020), operating over a unified graph schema that
represents a program simultaneously as its Abstract Syntax Tree (AST), Control-Flow
Graph (CFG), Data-Flow Graph (DFG), Program Dependence Graph (PDG), Call Graph, and
System Dependence Graph (SDG) — a construction generally known in program-analysis
literature as a **Code Property Graph (CPG)**.

The project pairs this custom code-model with a second, independent, off-the-shelf
system: **Qwen2.5**, a conventional decoder-only transformer language model, used
strictly for fluent natural-language interaction. The two systems are never merged
into one network; they remain architecturally separate and are connected only through
small, explicitly engineered **bridge modules** that translate between the two models'
vector spaces. This separation — a graph-native *code* model and a text-native
*language* model, joined by learned projection bridges rather than by prompt
concatenation — is the organising principle of the entire codebase and explains the
shape of almost every other design decision described below.

The system also includes: a full **curiosity-driven, calibration-aware Proximal
Policy Optimisation (PPO) reinforcement-learning loop** used to shape generation
behaviour beyond supervised imitation; a **streaming training pipeline** designed to
run on a single consumer GPU (the code repeatedly targets a 12 GB-class Ampere card,
e.g. an RTX 3060, as its baseline hardware budget); and a **local HTTP serving layer**
with an OpenAI-compatible chat endpoint intended for IDE integration.

---

## 2. The Four Files Under Review

| File | Lines (approx.) | Role |
|---|---|---|
| `CodeMind.py` | ~9,800 | The core brain. Owns the graph representation and parsers, the HGT network, the generation decoder, static/integrity validation, the curiosity-PPO reinforcement loop, the unified training loss and streaming training pipeline, hardware/precision/cache management, large-file and multi-project handling, and a standalone `argparse` CLI. Fully self-contained — trainable and usable without Qwen at all. |
| `codemind_brain_dual.py` | ~1,000 | Extensions layered on top of the core brain: a second specialised brain ("Brain B"), a mixture-of-experts-style router (`FlyPromptRouter`), an anti-catastrophic-forgetting bridge (`AntiForgetBridge`) between the two brains, a graph-size reduction algorithm (`GraphCollapser`), the two-phase training orchestrator (`DualBrainOrchestrator`), and the two neural bridge modules connecting CodeMind's embedding space to Qwen's (`QwenBrainBridge`, `QwenInstructionBridge`). |
| `CodeMind-Qwen2_5-14B.py` | ~1,430 | Everything concerning the language model in isolation: a forward-looking model-tier registry, quantised/offloaded loading strategies for Qwen2.5 (1.5B and 14B), and the two training entry points (`train-language`, `train-codemind`) — deliberately never loading both large models into memory at the same time. |
| `CodeMind-Run.py` | ~1,480 | The inference-time integration layer: `CodeMindRunner` (lazy-loads and coordinates both models), `StructuredEmbeddingBridge` and `TokenBudgetManager`, IDE-context sanitisation helpers, in-stream code-safety scanning of model output, and `serve_api()` — a local, OpenAI-compatible HTTP server. |

A fifth module, `codemind_paths.py`, is imported by all four files for filesystem
locations (data directories, checkpoint directories, offload directories) but was not
included in this review; it functions as a small path-configuration module rather than
an architectural component.

---

## 3. Layered Architecture

```
 Layer 6 — Serving & CLI
   CodeMindRunner · serve_api() (OpenAI-style local HTTP) · CodeMind.py argparse CLI
                                     │
 Layer 5 — Cross-Model Bridges       │
   QwenBrainBridge · QwenInstructionBridge · StructuredEmbeddingBridge · TokenBudgetManager
                                     │
 Layer 4 — Language Model            │
   Qwen14BModel (tiered, quantised, offload-aware load/generate/embed)
                                     │
 Layer 3 — Learning Loop             │
   CodeMindLoss · CuriosityPPOReward · PPOAgent · TrainingPipeline ·
   DualBrainOrchestrator · ExperienceBank · QualityEngine
                                     │
 Layer 2 — The Brain                 │
   HGTLayer/HGTBrain · TokenEfficientEncoder · SemanticFusion ·
   FlyPromptRouter · AntiForgetBridge · GraphCollapser · CodeDecoder
                                     │
 Layer 1 — Code Representation       │
   MultiLanguageParser · LanguageRegistry · CPGBuilder / GenericASTBuilder ·
   PolyglotLinker  →  CodeGraph (typed nodes + typed edges)
                                     │
 Layer 0 — Platform                  │
   PrecisionManager (FP8/BF16 locking) · SSDCache (safetensors) ·
   hardware detection · MindLog structured logging
```

Each layer consumes only the layer(s) beneath it; nothing in Layer 1 knows Layer 4
exists, for example. The remainder of this document proceeds bottom-up, since that is
also the order in which one piece of source code actually travels through the system
end to end.

---

## 4. Layer 0 — Platform Infrastructure

### 4.1 Hardware detection and precision policy

At start-up, `detect_hardware()` inspects the actual machine (GPU name, total VRAM,
whether the GPU is Ampere-generation or newer) and produces a `HardwareProfile`.
`PrecisionManager` then makes one binding decision for the run: **FP8 storage precision
is locked, with BF16 used for compute**, and — on Ampere-class or newer hardware —
TF32 matmul/cuDNN paths are enabled globally, since they are effectively free
throughput on that hardware generation. The key design choice is that this precision
decision is **locked for the duration of training** rather than dynamically
downgraded if instability is detected mid-run; the system prefers a run that either
completes at a known, fixed numeric precision or fails loudly, over one that silently
changes numerical behaviour partway through and produces results that are hard to
reproduce or compare against an earlier run.

### 4.2 Disk-backed spill cache: `SSDCache`

Under RAM/VRAM pressure, large or long-lived intermediate objects — cached
embeddings, per-sample curiosity/novelty markers, per-sample mastery scores — spill to
a local SSD-backed cache rather than staying resident in memory indefinitely.

A specific, documented security remediation is worth recording precisely: the cache
originally serialised these objects with Python's `pickle` protocol (via
`torch.save`/`torch.load`). Pickle is not a data-only format — deserialising a pickle
stream can execute arbitrary code embedded in it, so a cache file that has been
tampered with (e.g. corrupted in transit, synced from an untrusted machine, or shared
across a team) is a genuine arbitrary-code-execution vector, not merely a theoretical
concern. The fix replaces this with **`safetensors`** for any cached object containing
tensors — a format with no code-execution surface at all, the same format used to
distribute model weights on the Hugging Face Hub — and plain **JSON** for the scalar
values (booleans, floats) that the curiosity/trust/mastery caches actually need. After
the migration, no path in this caching layer deserialises pickle data at all, which
means a cache directory can be treated as untrusted input (e.g. safely shared or
synced) without that risk.

### 4.3 Structured logging: `MindLog`

A single logger instance is threaded through the whole system. Every notable event —
a training step, a PPO abstain decision, a hardware warning — is recorded to three
destinations simultaneously: the console (human-readable), an in-memory ring buffer
capped at 5,000 events (fast, no-disk-I/O access to recent history), and an
append-only JSONL file on disk per run (for later offline analysis of a full training
session). Events carry a `LogCategory` (e.g. `SYSTEM`, `DATA`, `TRAIN`, `PPO`) so a
downstream tool can filter by subsystem without string-parsing free-text log lines.

---

## 5. Layer 1 — Code Representation

Nothing downstream of this layer ever reasons about code as a flat token sequence.
Every input is first converted into a typed, heterogeneous graph.

### 5.1 The unified graph schema: `CodeGraph`

A `CodeGraph` holds a list of typed **nodes** and typed, directed **edges**. The same
fixed vocabulary of node and edge types is used *regardless of source language*, which
is the specific mechanism that allows one neural network to learn from every supported
language jointly instead of requiring a model per language.

**Node types** (14 total): `ast`, `cfg`, `dfg`, `pdg`, `call`, `sdg`, `token`, `func`,
`block`, `expr`, `stmt` for structural/behavioural code content, plus three
polyglot-only marker types — `lang`, `api`, `config` — used only when merging multiple
files/languages into one project graph (§5.4).

**Edge (relation) types** are 3-tuples of `(source node type, relation, target node
type)`. The schema includes intra-view edges — AST `child`/`next` structure, CFG
`flow`/`branch`, DFG `dataflow`, PDG `control_dep`/`data_dep`, call-site
`invokes`/`returns`, inter-procedural `inter_proc` — and, importantly, **cross-view
edges** that connect one analysis view to another on the same underlying code element:
`ast → maps_to → cfg`, `cfg → maps_to → pdg`, `dfg → maps_to → pdg`, `token → belongs →
ast`, `func → contains → ast`. These cross-view edges are what let the graph
transformer's attention mechanism move between "what the syntax literally says" and
"what actually happens at runtime" within a single message-passing step, rather than
learning each view in isolation. A final small set of edges exists purely for
multi-file/multi-language merging: `func → cross_lang → func`, `lang → hosts → func`,
`api → binds → call`, `config → wires → stmt`, `lang → same_family → lang`, `lang →
coexists → lang`.

### 5.2 Two graph-building strategies, chosen per language

- **`CPGBuilder` / `_PythonCPGBuilder`** builds a *complete, precise* CPG for Python by
  visiting the real AST produced by Python's own standard-library `ast` module,
  deriving a real control-flow graph, a real data/def-use flow graph, and real
  inter-procedural call edges from it. This path is only possible because Python ships
  a production-grade parser as part of the language runtime itself; it is treated as
  the reference-quality path in the system.
- **`GenericASTBuilder`** handles every other supported language, for which no bundled
  formal grammar exists. It is **family-aware**: each `LanguageSpec` (§5.3) declares
  which syntactic family (`c_like`, `js_like`, `functional`, `data`, `markup`,
  `config`, `shell`, `jvm`, `script`, `other`) a language belongs to, along with
  family-appropriate lexical hints (comment syntax, brace vs. indentation
  conventions), and the builder applies heuristic, line/brace/keyword-based parsing to
  still emit output in the same node/edge vocabulary as the Python builder. This is a
  deliberate, explicit accuracy/coverage trade-off: Python is understood with
  compiler-grade structural precision; every other language is understood with
  "sufficient statistical structure for a graph neural network to learn from,"
  without a correctness guarantee equivalent to a real parser.

### 5.3 Language identity and cross-language slots: `LanguageRegistry`

`LanguageRegistry` is a static table of **roughly 60 registered languages**, each
entry (`LanguageSpec`) declaring an id, display name, family, recognised file
extensions and aliases, and family-appropriate syntax hints. Language *detection* for
a piece of source without a known filename falls back to lightweight heuristic
matching against these hints. Two further pieces of registry state are what actually
make cross-language learning possible: every language and every family is assigned a
stable integer **slot id**, and every graph node carries its originating language's
slot (and family slot) as a small numeric feature — so "this call-site node came from
Rust, a c-like-family language" is encoded directly on the node, letting one shared set
of network weights condition its behaviour on language identity rather than needing a
separate encoder per language.

### 5.4 Multi-file, multi-language projects: `PolyglotLinker` / `MultiLanguageParser`

`MultiLanguageParser.parse_project()` parses each file in a project into its own
`CodeGraph`, then `PolyglotLinker` merges all of them into one connected graph, adding
the polyglot-only edge types from §5.1 to represent real cross-file, cross-language
relationships: an HTTP route defined server-side being invoked via a client-side
`fetch`/HTTP call, an import/require dependency between files, a configuration key
that wires two files' behaviour together, or a looser "these two languages belong to
the same family" relatedness signal. This is the mechanism that lets the brain reason
about, say, a Python backend, a JavaScript frontend, and a YAML configuration file
together as one connected structure, rather than as three unrelated graphs.

---

## 6. Layer 2 — The Brain: Representation and Reasoning

### 6.1 From node label to vector: `TokenEfficientEncoder`

Before the graph transformer runs, every node's textual label must become a fixed-size
vector. Rather than a conventional subword tokenizer — which would multiply sequence
length considerably for graph-structured input — `TokenEfficientEncoder` uses
**hashing combined with learned embedding tables**: each label is reduced to a small
set of hashed features (also incorporating the node's language and family slot from
§5.3), which are then looked up in learned embedding tables and combined. The stated
design target is roughly a **3–5× reduction** in effective sequence length compared to
raw byte-pair-encoding of equivalent graph content — a meaningful saving given a single
file's graph can contain thousands of nodes. `encode_nodes_batch` performs this for an
entire graph's nodes in a single GPU call, which is a prerequisite for the
whole-graph batching described in §6.4.

### 6.2 The reasoning core: `HGTLayer` / `HGTBrain`

`HGTLayer` implements the **Heterogeneous Graph Transformer** operator (Hu, Dong,
Wang, Sun; *WWW 2020*): because a `CodeGraph` contains many distinct node types and
many distinct relation types, the attention computation is **type-specific** —
query/key/value projections, and the attention weighting itself, are parameterised
per node-type and per relation-type, rather than one shared transformer block applied
uniformly regardless of what kind of graph element is being attended to or what kind
of edge connects two nodes. `HGTBrain` stacks multiple `HGTLayer`s, applies a
`semantic_head` pooling projection to produce one embedding per graph, and exposes
three task-specific output heads sharing the same backbone: `understand`, `generate`,
and `validate` — so the same learned structural representation serves several
downstream consumers rather than being recomputed per task. In the current default
configuration (`CodeMindConfig`, §10.1) Brain A uses **8 HGT layers, 64 attention
heads, and a 1024-dimensional hidden state**; Brain B (§7.1) uses a lighter 6-layer
variant.

`SemanticFusion` is a small learned gate blending the HGT's structural embedding with
the (cheaper, token-level) `TokenEfficientEncoder` signal — letting the network fall
back toward lexical similarity when purely structural signal is ambiguous, via a
sigmoid gate over the concatenation of both representations.

### 6.3 Constrained generation: two codecs, one deliberate trade-off

CodeMind does **not** generate code as free next-subword-token sampling from an open
vocabulary; generation is over a constrained action sequence describing graph
structure. The system contains two codecs reflecting a documented design evolution:

- **`ASTGrammarCodec`** (earlier approach): encodes and decodes **Python specifically**,
  using the exact field-by-field schema of every node class in Python's `ast` module.
  Every AST node class and every field of every class has a fixed, precomputed
  vocabulary slot (id layout computed once, before ever seeing training data, because
  the node-class list and constant-character set are properties of the Python grammar
  itself, not of the corpus). This lets a decoded sequence be constrained to always be
  grammatically well-formed Python, by construction.
- **`GraphActionCodec`** (current, generalised approach): intentionally calls
  `GenericASTBuilder` **directly**, bypassing `MultiLanguageParser.parse()`'s normal
  behaviour of routing Python through the fine-grained `_PythonCPGBuilder`. The
  documented rationale is specific and important: if generation used the very
  fine-grained Python-only representation (thousands of AST-level nodes per function,
  with no dedicated `stmt`-type node at all) while every other language used the
  coarser statement-level representation from `GenericASTBuilder`, the model would be
  learning two structurally incompatible granularities from the same shared weights,
  and any pattern-matching step that specifically looks for a `"stmt"` node type would
  silently find nothing for Python. Forcing **every** language, Python included,
  through the same builder for the *generation* path removes this inconsistency. This
  is an explicit trade-off: Python's generation path gives up the automatic
  well-formedness guarantee that `ast.unparse()` used to provide, in exchange for one
  system that learns generation uniformly across all languages, with output quality
  enforced downstream by the validator and the PPO reward loop instead of by
  construction. Understanding (`HGTBrain`) is unaffected by this change and continues
  to use the precise `CPGBuilder` for Python.

**`GraphRenderer`** performs the final text-rendering step, converting a decoded
sequence of `(node_type, label)` pairs into source code using family-aware heuristics
— again because most supported languages have no bundled formal grammar to render
against — and explicitly does **not** guarantee syntactic correctness, which is exactly
why every generation result is re-validated (§8) rather than trusted directly.

**`CodeDecoder`** is the autoregressive Transformer network performing the actual
decoding. Its context is a small number of *memory tokens* (default: 8) derived
entirely from the brain's semantic embedding of the prompt — the prompt text itself is
never re-tokenized and fed to this decoder, because "understanding the prompt" has
already happened once, via the HGT brain (§6.2), and that single embedding is what
conditions generation. Default sizing (`CodeDecoderConfig`): 512-dimensional model,
6 layers, 8 heads, a maximum action-sequence length of 768 in the base config (the
active `CodeMindConfig` overrides this to 1024-dim / 8 layers / 16 heads / a 2,048
action-length ceiling, since a graph-action sequence runs roughly 2–3× longer than the
equivalent code's character count); a hard generation ceiling of ~2,600 actions exists
specifically so the model **abstains rather than continuing to guess** once a
generation has clearly run past a reasonable length.

*Illustrative sketch of the generation control flow (not the real implementation):*

```
seed   = brain.understand(prompt)                  # one semantic vector, computed once
state  = decoder.init_from(seed)                    # decoder's only conditioning signal
actions = []
while not finished and len(actions) < action_ceiling:
    candidate = decoder.step(state, actions)         # e.g. "open a call node"
    if not codec.is_well_formed(actions + [candidate]):
        candidate = codec.best_valid_alternative(actions)
    actions.append(candidate)
code = renderer.render(actions, language)            # not guaranteed syntactically valid
result = validator.validate(code, language)          # always re-checked, see §8
```

### 6.4 Batched graph processing

A single `CodeGraph` is turned into per-node-type input tensors and per-edge-type
index tensors by `GraphTensorizer.graph_to_hgt_input`, which encodes an entire graph's
nodes in one GPU call and then redistributes the results back into per-type buckets by
tracked local index — avoiding a Python-level loop calling the encoder once per node.
`GraphTensorizer.graphs_to_hgt_batch` extends this to *multiple graphs at once*: it
concatenates several graphs into a single **block-diagonal heterograph** (the same
principle as PyTorch Geometric's `Batch.from_data_list`), offsetting each graph's edge
indices per node-type so that message passing never crosses between graphs, and
returning a `batch_idx` tensor recording which graph each node row belongs to (used at
final pooling time to recover one embedding per graph). This batching is specifically
what allows `HGTBrain` to process several training samples in a single forward pass
rather than one at a time — the latter being identified directly in the code as the
main reason GPU utilisation was previously low (small single-graph forward passes
followed by idle waiting).

---

## 7. Layer 2 (continued) — Dual-Brain Architecture

Rather than one network handling both "understanding" and "editing" equally well, the
system splits into two specialised sub-networks and adds explicit machinery to prevent
them from destructively interfering with each other.

### 7.1 Brain A and Brain B

**Brain A** (default: 8 HGT layers) specialises in understanding — analysis, embedding,
and validation support. **Brain B** (default: 6 HGT layers, intentionally lighter)
specialises in editing tasks. They are trained **sequentially, not jointly**: Brain A's
phase runs to completion first; Brain B's phase begins only afterward, with Brain A's
representation carried forward rather than discarded. `DualBrainOrchestrator` owns
this two-phase lifecycle, and `DualBrainState.phase` records whether the system is
currently in phase `"A"`, `"B"`, or `"unified"`.

### 7.2 Preventing catastrophic forgetting: `AntiForgetBridge`

When Brain B's training phase begins, the real risk is that gradient updates drive its
representation far enough from Brain A's that Brain A's earlier learned structure is
effectively erased — the well-known **catastrophic forgetting** problem in continual
learning. `AntiForgetBridge` addresses this with two complementary mechanisms:

1. **Learned fusion.** Brain A's and Brain B's embeddings are each linearly projected,
   concatenated, and combined through a sigmoid-gated mixture (`gate * a + (1-gate) *
   out(cat)`), followed by a `LayerNorm` — normalisation added specifically to stabilise
   training when `emb_a` and `emb_b` have different natural scales.
2. **EWC-lite penalty.** A lightweight version of *Elastic Weight Consolidation*: a
   running "anchor" (the mean of recent Brain-A embeddings) and a diagonal
   Fisher-information-like importance weighting (the mean squared value per dimension)
   are maintained as buffers, and the fused embedding is penalised — scaled by
   `ewc_lambda = 0.12` — for drifting from that anchor along dimensions that were
   historically important.

The bridge also computes a **`bridge_reward`**: the cosine similarity between Brain A's
(detached) embedding and the fused output, mapped from `[-1, 1]` to `[0, 1]` and
smoothed with an exponential moving average. This is not a passive diagnostic — it
feeds directly into the reinforcement-learning reward function (§9.3): if fusion is
*not* preserving Brain A's knowledge well, that is treated as something to actively
penalise during training, not merely something to log.

### 7.3 Sparse expert routing: `FlyPromptRouter`

`FlyPromptRouter` is a small mixture-of-experts-style router (its name references
insect-brain-inspired sparse coding/routing research): a `LayerNorm`-stabilised gate
projects the pooled input to per-expert logits, `task_id`-specific bias terms are
added, Gaussian exploration noise is optionally injected during training, and the
top-*k* (default *k*=2 of 6) experts are selected by softmax probability and combined
by weight. Two mechanisms worth understanding individually:

- Each expert (`TemporalEnsembleExpert`) is itself pre-normalised (`LayerNorm`) and
  keeps its own **exponential moving average of past outputs** (decay = 0.95),
  blending 12% of that EMA back into its current output — a second, expert-local
  anti-forgetting mechanism, distinct from and complementary to `AntiForgetBridge`.
- A **load-balance auxiliary loss** (`0.01 × Σ(prob − mean_prob)²`) discourages the
  gate from collapsing routing onto only one or two experts, a documented common
  failure mode for mixture-of-experts models. Notably, an earlier version of this
  router *computed* this auxiliary loss but discarded it as a detached value used only
  for logging — meaning the expert-collapse guard the docstring claimed to provide
  never actually influenced gradients. The fix exposes the loss as a live,
  non-detached tensor (`router._lb_loss`) that the training loop explicitly adds into
  the total loss, which is the only way an auxiliary loss can actually constrain
  behaviour during backpropagation.

### 7.4 Bounding graph size before the brain: `GraphCollapser`

Real code graphs routinely exceed the network's configured node budget
(`max_nodes`, default 4,096 in the current configuration — see §10.1). Rather than
naive truncation, `GraphCollapser` reduces graph size in three ordered, roughly
linear-time stages:

1. **Exact-duplicate merge** — nodes sharing identical `(node_type, label[:80])` are
   merged via a dictionary lookup (O(N)), except for a protected set of structurally
   important types (`func`, `call`, `cfg`, `dfg`, `pdg`, `ast`, `stmt`, `sdg`), which
   are never merged away even if duplicated.
2. **Linear-chain collapse** — a node with exactly one incoming and one outgoing edge
   that is *not* one of the four hard "anchor" types (`func`, `call`, `dfg`, `sdg`) and
   whose label does not match an "important" keyword pattern (`main`, `init`, `start`,
   `run`, `execute`, `handler`, `router`, `dispatch`, `entry`) is removed, and its
   neighbours are reconnected directly around it — collapsing "pass-through"
   structure that contributes little independent information.
3. **Score-and-trim** — if the graph is still over budget, every remaining node
   receives an importance score (a structural-type bonus, an important-keyword bonus,
   and a connectivity-degree term capped at 15), and only the top-scoring nodes up to
   the node budget are retained, with all edges remapped accordingly.

Deduplication and edge remapping are dictionary/set-based throughout (O(N) / O(E)
rather than pairwise O(N²) comparison), which matters because this routine can run
once per training sample in the hot path.

---

## 8. Layer 3 (Part 1) — Static Analysis, Integrity, and Validation

Every generated (or user-supplied) piece of code is checked by a deterministic,
non-neural pipeline before it is trusted, shown to a user, or used to compute a
reinforcement-learning reward. This subsystem exists specifically as a safety net
around a generative model, and its output score is not just advisory — it is the
system's primary source of ground truth for reward computation.

### 8.1 `StaticAnalyzer`

For Python, analysis is genuinely semantic, not just pattern-based: the source is
parsed with `ast.parse`, and every `Import`/`ImportFrom`, function/class definition,
function-argument binding (`ast.arg` nodes — handled explicitly because parameters are
*not* represented as `ast.Name` nodes and were previously mis-flagged as undefined),
and `except ... as name` binding is collected, alongside every `Name` node's load/store
context, to detect **possibly-undefined names** (used but never defined, imported, or
a recognised builtin) and **unused imports**. One specific, documented correctness fix:
determining the set of Python builtins must use `import builtins; dir(builtins)`
rather than `dir(__builtins__)` — when this module is *imported* normally (as opposed
to run directly), Python rebinds `__builtins__` to a **dict**, and `dir()` on a dict
returns dict methods (`keys`, `items`, `clear`, …) rather than builtin names, which
silently made the builtins set near-empty and caused ordinary calls like `print`,
`len`, `range`, `isinstance` to be flagged as undefined in real usage. For non-Python
languages, analysis falls back to generic checks: balanced parentheses/braces, and
regex-based detection of `eval(`/`exec(` usage.

A shared `_check_safety()` routine (used for every language) scans for a fixed set of
dangerous patterns and severities: `os.system(...)`, `subprocess.call/run/Popen(...)`,
`eval(`, `exec(`, and `rm -rf` and `DROP TABLE` are flagged as **critical**; dynamic
`__import__(...)`, unconditional `DELETE FROM ... ;`, `pickle.loads(...)`, and
`ctypes.CDLL` are flagged as **warnings**. The stated intent is explicitly *not* to
block the model from ever learning about such patterns — training data legitimately
contains them — but to make their presence visible to every downstream consumer of the
analysis (the validator, the reward function, the user-facing safety scanner in the
runner).

### 8.2 `IntegrityChecker` — detecting "cheating"

`IntegrityChecker.check()` looks specifically for output that resembles a real
solution without being one:

- **Copied-prompt detection** — if the (whitespace-normalised) prompt text appears
  verbatim inside a generation that is not meaningfully longer than the prompt itself,
  this is flagged as copying without implementing.
- **Hollow-code detection** — if every non-comment code line matches a small set of
  placeholder patterns (`pass`, `...`, `raise NotImplementedError`, `TODO`, `FIXME`,
  or a trivial `return None`/`0`/`""`/`[]`), the output is flagged as a hollow
  implementation.
- **Mixed prose/code detection** — lines matching a set of characteristic
  natural-language hedging markers, in both English (*"as an AI", "I think", "maybe",
  "here is", "note that"*) and Thai (*คือ, น่าจะ, อาจจะ, ลองดู, ในความเห็น, ขออภัย,
  ไม่แน่ใจ*), are counted; two or more such lines (or even one, outside markdown/plain
  text output) flags mixed natural-language content bleeding into what should be pure
  code.
- **Comment-only fake solutions** — output consisting entirely of comment lines (no
  real code lines at all) is flagged outright as cheating.

Each detected issue subtracts from a running integrity score in `[0, 1]`; a score
below 0.5 marks the output `is_cheating`. This score directly weights the *integrity*
term of the PPO reward (§9.3) and is one of the two multiplicative penalty factors
applied to the overall `CodeValidator` score (§8.4).

### 8.3 `CodePurityChecker`

A separate, lighter heuristic (independent of correctness) estimating what fraction of
non-blank, non-comment lines "look like code" — matched against a small set of code-like
patterns (keyword-leading lines such as `def `, `class `, `import `, `const `, `SELECT
`, etc., or lines composed purely of code-punctuation characters) versus "look like
prose" (long lines, more than 8 words, containing none of the typical code-structuring
characters `{ } ( ) ; =`). This produces a purity ratio in `[0, 1]`, used as a
multiplicative floor (`max(0.3, purity)`) on the overall validation score — heavily
prose-laden output cannot score well even if it happens to contain no outright syntax
errors.

### 8.4 `CodeValidator` — the composite verdict

`CodeValidator.validate()` combines the three checks above plus, optionally, real
sandboxed **execution** (Python only, `run_test=True`) into one `ValidationResult`.
The composite score starts at `1.0` and is adjusted as:

```
score  = 1.0
       − 0.4 × (# error-severity issues)
       − 0.1 × (# warning-severity issues)
       − 0.3 × (# safety-flagged issues)
score *= integrity_report.score
score *= max(0.3, code_purity)
score += 0.3  if executed successfully (when run_test=True)
score −= 0.3  if execution failed        (when run_test=True)
score  = clip(score, −1.0, 1.0)
```

A composite `should_abstain` recommendation is set when the output is flagged as
cheating, has more than two error-severity issues, or is hollow code that also failed
execution — cases judged too unreliable to present as a genuine answer. This score
(rescaled to `[0, 1]`) is what `CuriosityPPOReward.compute()` treats as ground-truth
*correctness* (§9.3), and is also what `CodeMindLoss.validate_loss` trains a dedicated
`validate` prediction head to approximate directly, so the model can eventually predict
its own likely validation outcome without running the full check.

---

## 9. Layer 3 (Part 2) — Curiosity-Driven Reinforcement Learning

Rather than training generation purely by supervised imitation of a corpus, or by a
reward that only measures whether output "worked," the system layers a
**multi-component intrinsic-plus-extrinsic PPO reward** on top of validation, built
around four explicitly named objectives preserved directly in the source: to be
**curious** (seek out genuinely new patterns), **self-aware** (be confident in
proportion to actual correctness, not more), **honest** (prefer declining to answer
over confidently producing something wrong), and **safe**.

### 9.1 Novelty memory: `CuriosityMemory`

A SHA-256 hash of whitespace-normalised source (truncated to 512 characters) is used
as a novelty key. Code the system has not seen (by this hash, checked both against an
in-memory set and the persistent `SSDCache`) scores full novelty; code it has already
seen scores zero — an explicit decay mechanism that stops the reward from repeatedly
favouring the same trivial pattern.

### 9.2 A slow-moving trust score: `TrustTracker`

A single running score in `[0, 100]`, adjusted per generation by the `trust_delta`
computed in §9.3: it decreases sharply on detected cheating or confident-and-wrong
output, and increases on honest abstention that turns out to have been warranted, or
on genuinely valid, high-integrity output. This is deliberately a *slower*, longer-horizon
signal about overall reliability, distinct from the per-generation reward.

### 9.3 Reward decomposition: `CuriosityPPOReward.compute()`

Given a `ValidationResult`, the model's own stated confidence for that generation,
whether it chose to abstain, and the `bridge_reward` from `AntiForgetBridge` (§7.2,
defaulting to 0.5 when not applicable), the reward is assembled from the following
weighted terms (all values as configured in the source):

| Term | Formula (informal) | What it rewards / penalises |
|---|---|---|
| Curiosity | `0.15 × novelty × (0.5 + 0.5 × integrity)` | genuinely new patterns, discounted if integrity is questionable |
| Correctness | `0.35 × (validation_score + 1) / 2` | the validator's own assessment, rescaled to `[0,1]` |
| Calibration | `−0.25 × gap` if overconfident by >0.1; `+0.05` if usefully under-confident on strong output; else `0.1 × (1 − |gap|)` | confidence matching actual correctness — punishes overconfidence more than it rewards under-confidence |
| Honesty | `+0.15` abstain-when-wrong; `−0.30` detected cheating; `+0.10` valid & not abstaining; `+0.05` low-confidence-and-wrong (at least not overconfident) | preferring truthful behaviour, especially abstaining appropriately |
| Integrity | `0.15 × integrity × purity`, further `−0.20` if cheating | combined integrity/purity signal from §8.2–8.3 |
| Safety | `0.10 × validator_safety_score` | absence of dangerous patterns |
| Bridge-retention bonus/penalty | up to `±0.05`, scaled once `bridge_reward` exceeds 0.7 (bonus) or falls below 0.3 (penalty) | whether Brain B is preserving Brain A's knowledge (§7.2) |
| Trust delta (separate, not summed into reward) | `−5.0` cheating/overconfident-wrong; `+0.5` warranted abstain; `+2.0` valid & high-integrity; `−2.0` otherwise invalid; `0.0` neutral | feeds `TrustTracker`, not the PPO scalar reward itself |

All reward terms (excluding the separate trust delta) are summed and clipped to `[-1,
1]` as the final scalar reward passed to PPO. A model that is *correct but recklessly
overconfident*, or correct in a way flagged as likely gaming the validator, is scored
*worse* here than a model that is honestly uncertain about a harder case — this is a
direct, quantitative expression of the "abstain over bluff" design principle discussed
in §15.

### 9.4 The policy network and PPO update: `CodePolicyHead` / `PPOAgent`

`CodePolicyHead` sits on top of the brain's embedding and produces a policy
distribution, a scalar value estimate, and a **confidence** score explicitly intended
to be *calibrated* (matched to actual likely correctness) rather than raw softmax
peakiness.

`PPOAgent` implements a standard, carefully engineered PPO update over this policy:

- **Optimiser**: AdamW, learning rate `3e-4`, weight decay `0.01`, betas `(0.9, 0.95)`
  — the lower β2 (versus the common default 0.999) is chosen specifically for more
  stable convergence in an RL setting.
- **Learning-rate schedule**: cosine annealing with warm restarts (`T_0 = max(warmup
  steps, 10)`, floor `1e-5`), stepped only after an initial 50-step warm-up.
- **Advantage estimation**: Generalised Advantage Estimation (GAE), `gamma = 0.99`,
  `gae_lambda = 0.95`, computed by a standard backward recursion over the
  reward/value buffer, followed by per-batch normalisation (zero mean, unit
  variance) of both advantages and rewards when the buffer holds more than a couple
  of samples.
- **Clipped surrogate objective**: standard PPO clipping with `clip_eps = 0.2`.
- **Value loss**: MSE between the predicted value and the (normalised) realised
  reward, weighted `value_coef = 0.5`.
- **Entropy bonus**: `entropy_coef = 0.02`, encouraging continued exploration rather
  than early policy collapse.
- **KL penalty against a frozen reference policy** (`kl_coef = 0.05`): after
  `set_reference()` snapshots the current policy (frozen, gradient-disabled), each
  update additionally penalises the current policy's KL divergence from that
  reference — a second, independent anti-forgetting mechanism, this time specific to
  the RL policy itself (distinct from `AntiForgetBridge`, which protects the
  embedding brains, and from each expert's own EMA blending in `FlyPromptRouter`).
- **Update cadence**: updates only fire once the experience buffer holds at least
  `min_buffer_size` (default 8) transitions, avoiding noisy single-sample policy
  updates; a `flush()` path forces an update on whatever remains buffered at, e.g.,
  the end of an epoch.
- **Gradient clipping**: global-norm clipped to 1.0 before the optimiser step.

The update loop is written to run over **batched tensors only** (no per-sample Python
loop), with logging metrics gathered into a single tensor and synchronised to Python
scalars once per epoch of PPO update rather than once per individual metric — a
deliberate performance choice, since a `.item()`/`.tolist()` call forces a CUDA
device→host synchronisation that stalls the GPU launch queue, and this cost compounds
significantly at millions of training steps.

### 9.5 Skipping mastered examples: `ExperienceBank`

For every distinct piece of training source (identified by a SHA-256 hash of the raw
text), a running **mastery** score in `[0, 1]` is tracked as an exponentially smoothed
average of past rewards for that exact sample (`new = 0.9 × old + 0.1 ×
max(0, reward)`). Once mastery reaches a configurable threshold (default 0.85), that
sample can be skipped or de-prioritised (`should_skip`, `priority`) in subsequent
training passes, so training time is spent disproportionately on samples the model has
not yet mastered rather than being spread evenly over material that is already solved.

---

## 10. Layer 3 (Part 3) — The Unified Loss and the Training Pipeline

### 10.1 Model configuration: `CodeMindConfig`

The active default configuration, with the reasoning for several non-obvious choices
preserved directly as comments in the source, is:

| Field | Default | Note |
|---|---|---|
| `hidden_dim` | 1024 | Deliberately reduced from an earlier 2048 specifically to fit training on a 12 GB-class GPU — at 2048, optimiser state alone exceeded 12 GB |
| `num_hgt_layers` | 8 | Doubled from an earlier 4, to capture dependencies across more hops |
| `num_heads` | 64 | Increased from an earlier 8, for much finer per-head attention granularity (head_dim = 1024/64 = 16... — chosen so hidden_dim divides evenly) |
| `max_nodes` | 4096 | Halved from an earlier 8192 specifically to reduce peak VRAM per training step — explicitly documented as a real, structural trade-off (more aggressive/frequent subsampling of large graphs, i.e. real loss of structural detail) rather than a pure efficiency win |
| `enable_ppo` / `ppo_buffer_size` | `True` / 16 | Increased from 8, for a larger, more stable PPO replay window |
| `enable_dual_brain` / `brain_b_layers` | `True` / 6 | Brain B scaled in proportion to Brain A's 4→8 increase, while staying lighter |
| `fp8_locked` | `True` | See §4.1 |
| `flyprompt_experts` | 6 | Increased from 4, for more routing diversity across 55+ language families |
| `decoder_dim` / `decoder_layers` / `decoder_heads` | 1024 / 8 / 16 | Generation-decoder sizing, tuned to still fit the same GPU budget |
| `decoder_max_seq_len` | 2048 | Action sequences run roughly 2–3× longer than the equivalent code's character length |
| `decoder_max_gen_actions` | 2600 | Hard ceiling — beyond this, generation must abstain rather than keep guessing |
| `qwen_embed_dim` | 5120 | Must match whichever Qwen tier produced the cached instruction embeddings (14B tier = 5120; light tier = 1536) |

### 10.2 `CodeMindLoss` — the unified multi-objective loss

Training combines five weighted sub-losses into a single scalar:

```
L_total = w_c · L_contrastive + w_s · L_semantic_align + w_v · L_validate
        + w_g · L_graph_reg   + w_p · L_ppo_policy
```

with default weights `w_c = 0.30`, `w_s = 0.25`, `w_v = 0.20`, `w_g = 0.10`,
`w_p = 0.15`.

- **`L_contrastive`** — an InfoNCE-style objective: a source graph's embedding should
  align with its paired target code's embedding, and be distinguishable from
  unrelated ("negative") embeddings, via cosine similarity scaled by a temperature of
  `0.07`. A specific, high-impact bug is documented here and worth understanding
  precisely: the pipeline trains on **one** `(source, target)` pair per step (not a
  batch), and an earlier version of this loss computed InfoNCE cross-entropy over a
  1×1 similarity "matrix" — one query, one positive, **zero negatives**. Softmax of a
  single logit is mathematically always exactly `1.0`, meaning that loss term — and
  its gradient — was provably zero for every input at every step, regardless of data
  or hyperparameters. Since `w_c = 0.30` is the single largest weight in the unified
  loss, roughly 30% of the intended training signal was inert for the entirety of any
  run using the unfixed version. The fix maintains a small FIFO bank (`deque`, default
  size 64) of recent target embeddings — detached, so no extra forward passes are
  needed — used as in-batch negatives, giving the similarity row real alternatives to
  discriminate against so the gradient is no longer identically zero. The bank persists
  across steps for the lifetime of one `CodeMindLoss` instance and is kept on CPU
  specifically to avoid extra VRAM pressure.
- **`L_semantic_align`** — simply `1 − cosine_similarity(source_embedding,
  target_embedding)`, a direct alignment pressure independent of the contrastive
  negatives.
- **`L_validate`** (`validate_loss`) — MSE between the brain's own predicted
  `validate` head output and the *actual* validation score computed by `CodeValidator`
  for that sample — training the model to predict its own likely validation outcome.
- **`L_graph_reg`** (`graph_regularizer`) — a small L2 penalty (`× 0.001`) on
  per-node-type embedding magnitudes, guarding against exploding activations rather
  than encoding any semantic objective.
- **`L_ppo_policy`** — the PPO policy-gradient loss described in §9.4, folded into the
  same unified backward pass when available.

A documented performance note: every component tensor is computed **exactly once**
per forward call (an earlier version computed `contrastive`/`semantic_align` twice
each — once for the returned loss, once again for a logged scalar — doubling GPU work
for no benefit), and all per-step logging scalars are gathered into a single tensor
and converted to Python floats with **one** `.tolist()` call rather than up to six
separate `.item()` calls — each of which forces a CUDA device→host synchronisation
that stalls the kernel launch queue. An explicit `sync_breakdown=False` fast path
skips this synchronisation entirely on steps where only the loss tensor (for
backward) is needed and no per-step metric breakdown will be read, deferring detailed
logging to every *N* steps instead of every step.

### 10.3 `QualityEngine` — a multi-dimensional quality score (0–100)

Model quality is explicitly **not** measured by loss alone. `QualityEngine.score()`
computes a weighted composite across six dimensions:

| Dimension | Max points | Basis |
|---|---|---|
| Syntax | 25 | `25 − 8×(error count)`, floored at 0 |
| Static | 20 | `20 − 4×(warning count)`, floored at 0 |
| Graph | 15 | `15 × (0.6 × node/edge coverage + 0.4 × node-type diversity)` against reference thresholds of 50 nodes / 80 edges and the full `NODE_TYPES` vocabulary |
| Semantic | 20 | `20 ×` cosine similarity to a target embedding when available, else a Jaccard word-overlap fallback against a target string, else a neutral 10 |
| Execution | 10 | 10 if the code actually ran successfully in a sandboxed test, 5 if merely `valid` without execution, 0 otherwise |
| Mastery | 10 | `10 ×` the `ExperienceBank` mastery score for that exact sample |

The six components sum to a 0–100 total. Training targets a **quality ≥ 85** stopping
criterion (§10.4) rather than any fixed loss threshold, and — as with validation —
this function accepts a precomputed `ValidationResult` when the caller already has one
(the training hot path does), specifically to avoid redundantly re-running validation
on the same code a second time within the same step.

### 10.4 `TrainConfig` and the streaming `TrainingPipeline`

Default training configuration includes: `epochs = 3`, `batch_size = 32` with gradient
accumulation `grad_accum = 4`, learning rate `2e-4`, a validation split of 10%, a
**quality-target early stop at 85%** with patience 3, and a per-batch data-availability
timeout of **20 seconds** (deliberately reduced from an earlier 120 seconds — the
longer timeout was identified as the main cause of the GPU sitting idle for up to two
minutes per batch whenever the CPU-side graph-parsing worker fell behind the training
loop's consumption rate).

Training itself is explicitly implemented as a **streaming pipe**, not a
batch-everything-upfront process:

```
Data (streamed) → Parse → CPG → HGT (Brain A / Brain B) → Unified Loss
  → Backprop → Validate → PPO Reward → Policy Update → Quality Report
```

`DataConnector` streams samples from a configured data directory (supporting `.jsonl`,
`.json`, plain `.py`/`.txt`/`.md` files, and paired prompt/code records), without ever
requiring the entire dataset to be resident in memory: a windowed **shuffle buffer**
(default capacity 20,000 samples) bounds memory to roughly that many samples' worth of
data at once, and an SSD-backed **shard cache** persists pre-processed shards between
runs — subject to a total size budget (default 20 GB) enforced by pruning the
least-recently-modified shard files first (an LRU-by-mtime eviction policy), which was
added specifically because an earlier version of the shard cache had no size ceiling
at all and grew without bound. `_ParsePrefetcher` performs graph parsing in background
worker processes (`ProcessPoolExecutor`) ahead of when the training loop needs each
sample, which is what the 20-second batch timeout is guarding against falling behind.

Per-epoch and per-run reporting (`TrainingReport`) is deliberately granular: beyond
aggregate loss and quality history, it records a **per-component loss/quality
breakdown history** (contrastive, semantic, validate, graph, PPO, individually, per
epoch) and a computed list of **stagnant components** — sub-losses whose per-epoch
average did not improve between the last two epochs even while the total loss/quality
did improve. The documented rationale is direct: a descending total-loss curve alone
cannot tell you *which* part of the model is actually improving versus which part is
flat or regressing while being outweighed by the other terms; this per-component
history is what lets that question be answered after a run instead of only being
inferred by eye from one aggregate curve.

### 10.5 Checkpointing

As with `SSDCache` (§4.2), model weights are persisted with `safetensors` rather than
raw pickle, with non-tensor metadata (training phase, epoch counters, sample counts)
stored separately as plain JSON. `DualBrainOrchestrator.save_dual()`'s history contains
two more documented, concrete fixes worth recording: an earlier version resolved its
save directory from a **file** path constant that was actually meant to identify the
unified-checkpoint *file*, but was being used as though it were a directory — corrected
to reference the dedicated checkpoint-directory constant instead; and an earlier
version additionally wrote each brain/bridge component to its own separate file *as
well as* to the combined `unified.safetensors` file, even though the load path
(`load_dual()`) was confirmed to read only from the unified file and never from the
per-component files — meaning the per-component files were redundant writes,
consuming steadily growing disk space every training run for data that was never
actually read back. The per-component writes were removed, leaving only the one file
that is genuinely used. A defensive legacy loader remains for pre-migration pickle
checkpoints, which get re-saved in the safe format the next time `save_dual()` runs.

---

## 11. Scaling Beyond a Single Small File

- **`LargeCodeHandler`** processes source files in the 10K–500K+ line range by
  splitting on function/class boundaries (never mid-function), running
  `understand`/`generate` independently per chunk, and reassembling output while
  preserving indentation and structure — so the whole file is never required in
  memory, or within the graph node budget, at once. For multi-chunk *generation*
  specifically, the semantic embedding produced for the previous chunk is blended
  (`0.7 × current + 0.3 × previous`) into the next chunk's conditioning embedding, so
  cross-chunk coherence is carried through the vector representation itself and not
  only through whatever textual "tail" happens to be visible in the prompt string.
- **`MultiProjectConnector`** applies the same principle at project scale (roughly
  100–2,000 files): it first scans and indexes every file (path, size, detected
  language, modified time) without reading file contents, groups files into batches
  sized to a configured RAM budget, parses and merges each batch through
  `PolyglotLinker` incrementally, and applies `GraphCollapser` to the accumulated
  graph before it reaches the HGT brain — so a large project is processed as a bounded
  sequence of manageable batches, never as one unbounded in-memory graph.

---

## 12. Layer 4 & 5 — The Language Model and the Cross-Model Bridges

### 12.1 Why a genuinely separate model

CodeMind's own decoder (§6.3) produces code conditioned on a semantic vector; it does
not read or produce fluent natural language and was never intended to. Free-form
conversation about code — explanations, rationale, general discussion — is handled
entirely by a separate, conventional large language model, **Qwen2.5**, kept in its
own file specifically so training never requires holding both large models in memory
at once (a 12 GB-class GPU cannot hold both simultaneously in any of the configured
tiers). The module's own documentation states this directly: **three modes exist and
are never combined** — `train-language` (Qwen only, no CodeMind loaded),
`train-codemind` (CodeMind only, no Qwen loaded), and `run`/`chat` (handled entirely
by the separate runner file, `CodeMind-Run.py`, which loads both — but lazily, and
never training either while doing so).

### 12.2 A forward-looking model-tier registry

`TIER_REGISTRY` declares three tiers with explicit hardware requirements, including
one not currently runnable on the target hardware, specifically so a future
contributor does not have to rediscover *why*:

| Tier | Model | Architecture | Min. VRAM | Loader | Feasible on this hardware today |
|---|---|---|---|---|---|
| `light` | Qwen2.5-1.5B-Instruct | dense | 2 GB | in-process (`hf_bnb`) | Yes — fast dev/testing tier |
| `14b` | Qwen2.5-14B-Instruct | dense | 10 GB | in-process (`hf_bnb`) | Yes — current primary tier, 8-bit on a 12 GB GPU |
| `122b-moe` | Qwen3.5-122B-A10B | Mixture-of-Experts, 256 experts / 10B active per token | ~74 GB at 4-bit | external `vllm_server` (tensor-parallel) | No — not a "bigger version" of the same loading strategy; a genuinely different architecture family requiring multi-GPU tensor-parallel serving, e.g. 4× H100/A100-class hardware, which a single consumer GPU cannot approximate at any quantisation level |

The `122b-moe` entry is a documented placeholder for future scaling: it is architected
to be swapped in as an HTTP client (`_load_moe_via_server`) against a self-hosted
vLLM/SGLang deployment rather than an in-process load, precisely because
`from_pretrained` + `bitsandbytes` quantisation — the strategy that works for the
dense 14B model — is not a practical loading strategy for a sparse MoE model at that
scale.

### 12.3 Memory-constrained loading strategies for the 14B tier

`Qwen14BModel` selects among several loading strategies depending on available VRAM,
in order of preference: **8-bit quantised, entirely on GPU** (the CodeMind default —
described in the source as noticeably sharper output quality than 4-bit NF4
quantisation, via `bitsandbytes`' `BitsAndBytesConfig(load_in_8bit=True)`, all weights
pinned to device 0 with no CPU offload); **4-bit quantised, entirely on GPU** (a
fallback when 8-bit does not fit, using roughly 8–9 GB of VRAM on the target
hardware); and, if neither GPU-resident option fits, **layer-sharded SSD offload**
using Hugging Face `accelerate`'s `init_empty_weights` / `infer_auto_device_map` /
`load_checkpoint_and_dispatch` machinery, which computes a balanced per-layer
device map across GPU, CPU RAM, and a dedicated SSD offload directory, and spills
whatever does not fit in GPU+RAM to disk — explicitly engineered to keep total RAM
usage around a small fixed budget (documented as ~1 GB, with the rest spilling to the
M.2 SSD) rather than requiring the full model footprint in system RAM.

### 12.4 Bridging *from* CodeMind *to* Qwen: structured soft prompts

Rather than converting CodeMind's understanding of a piece of code into a long text
description and inserting that into Qwen's prompt (expensive in tokens and inherently
lossy), two closely related bridge modules project CodeMind's semantic embedding
directly into Qwen's own input-embedding space as a short sequence of **virtual
("soft") prompt tokens**:

- **`QwenBrainBridge`** (in the dual-brain file) takes the brain's semantic embedding
  plus a 16-dimensional metadata vector (language-family id, language confidence,
  file count, average line count, presence of imports/classes, a cyclomatic-complexity
  proxy, edit-mode flag, a quality signal, a token-budget hint, and reserved slots for
  future metadata) and projects it, through an intermediate layer sized to avoid a
  direct bottleneck, into `num_tokens` (default 8) vectors of Qwen's hidden dimension
  (5,120 for the 14B tier).
- **`StructuredEmbeddingBridge`** (in the runner file) is functionally the same idea
  with slightly different internal plumbing, used specifically at inference time by
  `CodeMindRunner`.

Either way, the resulting 8 dense vectors are prepended to the conversation as
`inputs_embeds` — functioning as extra, invisible "tokens" carrying dense structured
context. The documented framing is explicit: this replaces what would otherwise be a
roughly 50–100 token free-text summary with 8 dense vectors, at both lower token cost
and without the information loss of compressing a structured understanding into
prose.

*Illustrative sketch (not the real tensor shapes):*

```
brain_vector = codemind.understand(code).semantic            # 1024-dim (Brain A / config default)
meta_vector  = [trust, quality, validate_score, node_count, edge_count, language_id, ...]
soft_prompt  = bridge.project(brain_vector, meta_vector)      # 8 × 5120-dim Qwen-space vectors
response     = qwen.generate(prefix_embeds=soft_prompt, messages=chat_history)
```

### 12.5 Bridging *from* Qwen *to* CodeMind: instruction understanding

**`QwenInstructionBridge`** is the deliberate mirror image, and its documentation
states the motivating problem precisely: before this bridge existed, CodeMind's own
graph-based understanding pipeline was effectively being asked to parse raw
natural-language *instructions* directly — something a model built around
code-syntax graphs has no genuine capacity to do. This bridge instead consumes Qwen's
own **pooled hidden state** for a natural-language instruction and projects it down
into CodeMind's embedding space (through an intermediate layer and a final
`LayerNorm`, specifically to prevent scale mismatches from exploding when it is added
to the structural embedding). Crucially, that Qwen embedding is computed **offline**,
ahead of CodeMind training — via a dedicated `embed-instructions` command in the Qwen
file — and cached to disk per training-sample id, rather than run live during CodeMind
training, again because a 12 GB-class GPU cannot hold both large models simultaneously.
The resulting vector is **added to** (not substituted for) whatever structural
embedding CodeMind derives from any code attached to the instruction, via
`MultiLanguageParser`'s own instruction-parsing path; if no cached Qwen embedding
exists for a given sample, the system falls back cleanly to the structural signal
alone rather than failing. This is the specific mechanism that lets "Qwen understands
language" and "CodeMind understands and writes code" combine in the intended
direction, without CodeMind ever needing to become a language-understanding model
itself.

### 12.6 Runtime cost control: `TokenBudgetManager`

At inference time, `TokenBudgetManager` caches recently computed "brain packs" (a
piece of code's full understanding output) keyed by a SHA-256 hash of normalised code
text, in a bounded LRU cache (default capacity 64 entries) — so identical or repeated
code appearing across a conversation is not re-analysed from scratch. It also
adaptively sizes the `max_new_tokens` budget requested from Qwen based on the actual
amount of code/context present: a short chat message with no code gets a small budget
(as little as 32–160 tokens depending on message length), while a large pasted
function scales the budget roughly linearly with line count and character count, up to
a capped ceiling — a direct, measurable cost/latency control rather than a fixed
budget for every request regardless of size.

---

## 13. Layer 6 — Serving and CLI

### 13.1 `CodeMindRunner` — the interactive integration object

`CodeMindRunner` is the object that actually drives day-to-day interactive use after
training: it lazily loads the CodeMind brain and the Qwen model only when first
needed, can unload either after a configurable idle period (`auto_unload_qwen_sec`,
default 300 s) to free VRAM, retrieves or computes a brain pack for any code involved
in the current turn (via `TokenBudgetManager`, §12.6), builds the soft-prompt prefix
via `StructuredEmbeddingBridge` (§12.4), and exposes the high-level operations
`explain(code)`, `chat(message, code=...)`, `code(prompt)` (generation),
`code_and_explain(...)`, and `status()`.

It also implements several defensive, IDE-integration-specific text-processing
routines: stripping legacy metadata markers that may be embedded in IDE-supplied
context, separating a "workspace file snapshot" block from a "running conversational
memory" block within a combined payload, and trimming an oversized context block down
to a character budget while attempting to preserve the most relevant parts (e.g.
collapsing a bulk file listing rather than truncating it arbitrarily). Notably, it also
performs **in-stream safety scanning**: as the model's streamed response produces
closed code blocks, each one is run through `CodeValidator` as it closes, and a safety
warning is surfaced inline immediately if a dangerous pattern is detected — rather than
only being caught by a post-hoc check after the full response has already been shown
to the user.

If neither Qwen weight tier is available locally, a **`DemoResponder`** produces
clearly labelled canned demo output instead of failing outright, letting the system be
tried without first committing to a large model download.

### 13.2 `serve_api()` — a local, OpenAI-compatible HTTP server

A `ThreadingHTTPServer` exposes `POST /v1/chat/completions` (OpenAI-Chat-Completions
compatible request/response shape), plus `GET /v1/models` and `GET /v1/stats`,
intended as the local integration point for an external editor/IDE plugin. Basic
production-mindedness is present even for a strictly local tool: an optional API key
check (`CODEMIND_API_KEY` environment variable — the server prints an explicit warning
if it is bound to a non-localhost host without a key configured), a request body size
cap (2 MB), and a simple per-client sliding-window rate limit (default: 90 requests per
60-second window). The server also exposes a background **warm-up** path
(`warmup_qwen()`) that preloads the Qwen 14B tier ahead of the first real request
(documented as taking roughly 5–7 minutes on first load), reporting warm-up phase and
any failure (including detecting the common Windows "page file too small" failure
mode and surfacing that specifically) via `GET /v1/stats`.

### 13.3 The standalone `CodeMind.py` CLI

Independent of Qwen and the runner entirely, `CodeMind.py` exposes its own
`argparse`-based command-line interface for developing and testing the brain in
isolation:

| Command | Purpose |
|---|---|
| `status` | Report model/data/checkpoint status |
| `understand <input>` | Structural analysis and issue report for a file or inline snippet |
| `generate <prompt>` | Generate code from a prompt; `--large` switches to chunked, boundary-aware generation past a single token budget (see §11); `--force` bypasses the abstain guard |
| `validate <input>` | Report the validation score/issues for a file or snippet |
| `project <paths...>` | Recursively scan and understand an entire directory tree |
| `train` | Connect a data directory and run training, optionally saving a checkpoint |
| `demo` | Auto-detection + polyglot + a short training smoke test |
| `cache` | Inspect, or clear, the SSD shard cache |
| `repl` | An interactive loop that loads the model once and stays resident — also the **default** action when no subcommand at all is given, a deliberate choice so a first-time user running the bare script lands in something interactive rather than a bare usage error |

---

## 14. End-to-End Data Flows

### 14.1 Training flow, one sample

```
raw source text (any supported language)
   │  LanguageRegistry.detect()
   ▼
CodeGraph                              (CPGBuilder for Python / GenericASTBuilder otherwise)
   │  PolyglotLinker                    (only for multi-file / multi-language input)
   ▼
GraphCollapser.collapse()              (only if graph exceeds max_nodes = 4096)
   ▼
GraphTensorizer → (x_dict, edge_index_dict[, batch_idx_dict])   per node-type tensors
   ▼
Brain A (HGT, 8 layers) → FlyPromptRouter → embedding_a
   │  (only once Brain A's training phase has completed)
   ▼
Brain B (HGT, 6 layers) → FlyPromptRouter → AntiForgetBridge(embedding_a, ·) → fused embedding
   ▼
CodeMindLoss = w_c·L_contrastive + w_s·L_semantic + w_v·L_validate + w_g·L_graph + w_p·L_ppo
   ├── backward() / optimizer.step()
   ▼
CodeValidator.validate(model output)                (static + integrity + purity [+ execution])
   ▼
CuriosityPPOReward.compute(...)  →  PPOAgent.store()/update()
                                  →  TrustTracker.update()
                                  →  ExperienceBank.update(mastery)
                                  →  QualityEngine.score()  (0–100, six weighted dimensions)
```

### 14.2 Inference / chat flow, via the runner

```
user message (+ optional pasted code)
   ▼
CodeMindRunner
   │
   ├─ if code present ─► TokenBudgetManager.get_pack(code)
   │                        │ cache miss
   │                        ▼
   │                   CodeMind.understand(code) → UnderstandResult
   │                        (semantic embedding, structural issues, validate score)
   │                        ▼
   │                   StructuredEmbeddingBridge(semantic, meta) → 8 soft-prompt vectors
   │
   ▼
_build_chat_messages(...)     — assembles system/user turns + the soft-prompt prefix
   ▼
Qwen14BModel.generate_with_prefix_embeds(...)     — streamed token generation
   ▼
_scan_response_code_blocks():
   as each fenced code block in the stream closes, run CodeValidator on it and
   surface an inline safety warning immediately if a dangerous pattern is found
```

---

## 15. Design Philosophy — Recurring Engineering Principles

Several themes recur across essentially every layer, and understanding them explains
*why* many individual decisions look the way they do, not just *what* the decisions
were:

1. **Abstain over bluff.** An explicit abstain path exists in generation
   (`should_abstain`, the hard action-length ceiling), in validation
   (`ValidationResult.should_abstain`), and — most quantitatively — as a dedicated
   *calibration* and *honesty* term inside the PPO reward function, which specifically
   penalises confident-but-wrong output more heavily than it penalises honest
   uncertainty. This is treated as a training objective, not only a runtime safeguard.
2. **Structure over text, wherever the cost of doing so is affordable.** Code is
   represented as a typed graph rather than a token stream; cross-model context is
   passed between CodeMind and Qwen as dense projected vectors rather than paraphrased
   text; generation is shape-constrained rather than freely sampled. Free text is
   consistently treated as the expensive, lossy fallback rather than the default
   representation.
3. **One learned system across languages, rather than one model per language.** Both
   the graph schema (§5.1) and the generation codec (§6.3) were explicitly reworked, at
   the documented cost of losing a Python-specific guarantee (`ast.unparse()`'s
   always-valid syntax), specifically to avoid two languages being represented at
   structurally incompatible granularities within the same shared network.
4. **Security and reproducibility treated as first-class engineering, not
   afterthoughts.** The pickle→safetensors migration (§4.2, §10.5) and the
   precision-locking policy (§4.1) both read as considered responses to concrete,
   named failure modes — an arbitrary-code-execution vector in one case, irreproducible
   numerics in the other — rather than generic defensive boilerplate.
5. **Every sizing decision is tied to a specific, named hardware budget.** Model
   dimension, layer count, head count, graph node ceiling, and even whether a given
   model tier can be loaded at all, are repeatedly justified in terms of a specific
   target GPU class (12 GB-class Ampere, e.g. RTX 3060), and the two large models are
   kept in physically separate files specifically so neither training path ever
   requires holding both in memory simultaneously.
6. **The "why," including past mistakes, is treated as part of the documentation, not
   something to delete once fixed.** Numerous components carry inline notes recording
   a bug that was found, precisely why it mattered, and what the fix was — the
   contrastive loss with a provably-zero gradient, the load-balance loss that was
   computed but never actually wired into backpropagation, the `dir(__builtins__)`
   bug that made ordinary builtins register as undefined, a checkpoint path constant
   used incorrectly as a directory, an unbounded shard cache. Retaining this history
   directly alongside the code it explains — rather than only the corrected final
   state — is itself a deliberate practice, and one worth continuing in any future
   documentation of this system.

---

## 16. Terminology Glossary

| Term | Meaning in this system |
|---|---|
| **CPG (Code Property Graph)** | A single graph unifying AST, CFG, DFG, PDG, and call-graph views of a program |
| **HGT (Heterogeneous Graph Transformer)** | Hu et al. (WWW 2020); a graph transformer whose attention parameters are specific to node/edge *types* rather than shared uniformly |
| **Brain A / Brain B** | The two specialised HGT sub-networks — understanding (A) vs. editing (B) |
| **AntiForgetBridge** | Fuses Brain A and Brain B outputs and applies an EWC-lite penalty so Brain B does not overwrite Brain A's learned representation |
| **FlyPromptRouter** | A small mixture-of-experts-style router selecting which internal expert sub-network(s) process a given embedding, with a load-balance auxiliary loss against expert collapse |
| **GraphCollapser** | Reduces an oversized CodeGraph toward a node budget while preserving structurally important nodes, in three linear-time stages |
| **GraphActionCodec / ASTGrammarCodec** | The two encode/decode schemes converting between a CodeGraph and a generation-ready action sequence |
| **CodeDecoder** | The autoregressive Transformer producing the action sequence during generation, conditioned only on the brain's semantic embedding |
| **GraphRenderer** | Converts a decoded action sequence back into source-code text, heuristically, per language family |
| **Soft prompt / prefix embedding** | A short sequence of dense vectors injected directly into a language model's input-embedding space in place of a text description |
| **BrainExportPack** | The compact structured summary (embedding + metadata) exported from CodeMind for the Qwen bridge |
| **Curiosity / novelty** | A reward term for encountering not-previously-seen (by content hash) code patterns |
| **Calibration** | How closely the model's stated confidence tracks its actual correctness, specifically penalised for overconfidence |
| **Mastery** | A per-sample running score (`ExperienceBank`) used to decide whether continued training on that exact sample is still worthwhile |
| **PPO (Proximal Policy Optimisation)** | The clipped-objective reinforcement-learning algorithm updating the policy/confidence head from the composite reward, with GAE, entropy bonus, and a KL penalty against a frozen reference policy |
| **EWC (Elastic Weight Consolidation)** | A continual-learning technique penalising drift from an earlier learned state, scaled by a per-dimension importance weighting; used here in a lightweight form inside `AntiForgetBridge` |
| **FP8 (locked) / BF16 / TF32** | The numeric precision modes used for storage / compute / matmul during training, fixed for the duration of a run |
| **Polyglot graph / PolyglotLinker** | The merged multi-language, multi-file project graph, and the component that constructs it |
| **Language family** | A coarse grouping of syntactically similar languages (`c_like`, `js_like`, `functional`, `markup`, `config`, `shell`, `jvm`, `script`, `data`, `other`) used for heuristic parsing and as an embedding feature |
| **Block-diagonal batching** | Combining several graphs into one heterograph for a single forward pass, with per-graph edge indices offset so message passing never crosses between graphs (analogous to PyTorch Geometric's `Batch.from_data_list`) |

---

## 17. Quick File Map — "Where do I look for X?"

| If you need to understand or change… | Look in |
|---|---|
| How a specific language is parsed into a graph | `CodeMind.py` — `LanguageRegistry`, `GenericASTBuilder`, `CPGBuilder` |
| The core neural network / attention mechanism | `CodeMind.py` — `HGTLayer`, `HGTBrain` |
| How code is generated (decoding strategy) | `CodeMind.py` — `GraphActionCodec`, `CodeDecoder`, `GraphRenderer` |
| Whether generated code is deemed safe/real | `CodeMind.py` — `StaticAnalyzer`, `IntegrityChecker`, `CodePurityChecker`, `CodeValidator` |
| The reward function / RL behaviour | `CodeMind.py` — `CuriosityMemory`, `TrustTracker`, `CuriosityPPOReward`, `PPOAgent`, `CodePolicyHead` |
| The unified training loss | `CodeMind.py` — `CodeMindLoss` |
| Model quality measurement (0–100) | `CodeMind.py` — `QualityEngine` |
| The training loop and data streaming | `CodeMind.py` — `TrainingPipeline`, `DataConnector`, `_ParsePrefetcher` |
| Batched graph tensorisation | `CodeMind.py` — `GraphTensorizer` |
| Precision / VRAM / disk-cache behaviour | `CodeMind.py` — `PrecisionManager`, `SSDCache`, `detect_hardware` |
| Large-file / large-project handling | `CodeMind.py` — `LargeCodeHandler`, `MultiProjectConnector` |
| Two-brain training order, anti-forgetting, MoE routing, graph size control | `codemind_brain_dual.py` — `DualBrainOrchestrator`, `AntiForgetBridge`, `FlyPromptRouter`, `GraphCollapser` |
| How CodeMind's understanding reaches Qwen | `codemind_brain_dual.py` — `QwenBrainBridge`; `CodeMind-Run.py` — `StructuredEmbeddingBridge` |
| How natural-language instructions reach CodeMind | `codemind_brain_dual.py` — `QwenInstructionBridge` |
| Downloading/loading/quantising the Qwen model, and the tier registry | `CodeMind-Qwen2_5-14B.py` — `Qwen14BModel`, `TIER_REGISTRY`, `download_model` |
| Training Qwen or CodeMind separately (never simultaneously) | `CodeMind-Qwen2_5-14B.py` — `train_language_only`, `train_codemind_only` |
| Chat/explain/generate at runtime, and the local API server | `CodeMind-Run.py` — `CodeMindRunner`, `serve_api` |
| Runtime token/cost budgeting | `CodeMind-Run.py` — `TokenBudgetManager` |
| Command-line usage of the core brain alone | `CodeMind.py` — the `argparse` CLI at the bottom of the file |

---

*End of document.*
