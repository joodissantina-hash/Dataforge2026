"""Research evidence layer.

Every research-facing statement the UI shows carries a 'tier' so a reader can tell
what kind of thing it is:

formal_result       - a theorem / derivation stated in the cited paper
reported_experiment - an empirical result the cited paper reports
interpretation      - our reading of the paper; the paper does not say this verbatim
amplification       - an educational abstraction in THIS artifact that is NOT in any paper

Nothing here claims the toy simulator reproduces the BDN architecture. It does not.
"""

from typing import Any, Dict, List

PAPERS: List[Dict[str, str]] = [
    {
        "key": "bdh2025",
        "title": "The Dragon Hatchling: The Missing Link between the Transformer and "
                 "Models of the Brain",
        "authors": "A. Koskvi, P. Uzanski, J. Choronski, Z. Starowiczka, M. Bartoszkiewicz (Pathway)",
        "year": "2025",
        "venue": "arXiv:2509.26507",
        "url": "https://arxiv.org/abs/2509.26507",
        "code": "https://github.com/pathwaycom/bdh",
        "why": "Primary source for BDH and its tensorized BDH-GPU variant. Verified against "
              "the arXiv listing and the official Pathway repository before any claim on "
              "this page was written.",
    },
    {
        "key": "mamba2023",
        "title": "Mamba: Linear-Time Sequence Modeling with Selective State Spaces",
        "authors": "A. Gu, T. Dao",
        "year": "2023",
        "venue": "arXiv:2312.00752",
        "url": "https://arxiv.org/abs/2312.00752",
        "code": "https://github.com/state-spaces/mamba",
        "why": "Establishes that input-dependent selectivity (deciding what to keep vs. "
              "forget) is what makes a fixed-size recurrent state usable on "
              "content-based recall.",
    },
    {
        "key": "zoology2024",
        "title": "Simple Linear Attention Language Models Balance the "
                 "Recall-Throughput Tradeoff",
        "authors": "S. Arora, S. Eyuboglu, M. Zhang, A. Timalsina, S. Alberti, D. Zinsley, "
                   "J. Zou, A. Rudra, C. Ré",
        "year": "2024",
        "venue": "ICML 2024 [PMLR v235]",
        "url": "https://proceedings.mlr.press/v235/arora24a.html",
        "code": "https://github.com/HazyResearch/zoology",
        "why": "The quantitative statement of the tradeoff: this artifact teaches recall "
              "quality is bounded by recurrent state size, measured on multi-query "
              "associative recall.",
    },
    {
        "key": "gateddelta2024",
        "title": "Gated Delta Networks: Improving Mamba2 with Delta Rule",
        "authors": "S. Yang, J. Kautz, A. Hatamizadeh",
        "year": "2024",
        "venue": "arXiv:2412.06464 (ICLR 2025)",
        "url": "https://arxiv.org/abs/2412.06464",
        "code": "https://github.com/NVlabs/GatedDeltaNet",
        "why": "Shows that how you write to a fixed state (delta-rule replacement plus "
              "gated erasure vs. plain decay) determines whether it goes stale - the "
              "mechanism the update-rate slider on this page exposes.",
    },
    {
        "key": "forgetting2024",
        "title": "Stuffed Mamba: Oversized States Lead to the Inability to Forget",
        "authors": "Y. Chen, X. Han, Z. Liu et al.",
        "year": "2024",
        "venue": "arXiv:2410.07145",
        "url": "https://arxiv.org/abs/2410.07145",
        "code": "",
        "why": "Direct empirical evidence for the failure half of our claim: recurrent "
              "models retain information they should have discarded, and retrieval "
              "interference follows.",
    },
]

CLAIMS: List[Dict[str, Any]] = [
    {
        "id": "c1",
        "tier": "formal_result",
        "statement": "BDH is defined as a graph of locally interacting neurons and synapses, "
                     "and the paper derives an equivalence between its macro-scale behaviour "
                     "and Transformer-style attention - presented as the 'missing link' "
                     "between attention and brain models.",
        "paper": "bdh2025",
        "maps_to": "The framing of this page: attention and persistent state are two ends of "
                   "one axis, not unrelated designs.",
    },
    {
        "id": "c2",
        "tier": "formal_result",
        "statement": "BDH-GPU, the tensorized variant, uses low-rank factorisation of the "
                     "interaction matrices together with linear attention, which makes the "
                     "model's working memory a mutable recurrent state updated during "
                     "inference rather than a static KV cache.",
        "paper": "bdh2025",
        "maps_to": "Why the BDN-inspired mode on this page keeps a dxd synaptic state that is "
                   "written to at inference time instead of a growing cache.",
    },
    {
        "id": "c3",
        "tier": "reported_experiment",
        "statement": "The paper reports Transformer-like scaling behaviour, with BDH performing "
                     "comparably to GPT-2-class architectures as parameter scales from roughly "
                     "10M to 1B on language and translation tasks.",
        "paper": "bdh2025",
        "maps_to": "Context only. This page runs no trained model and makes no scaling claim.",
    },
    {
        "id": "c4",
        "tier": "reported_experiment",
        "statement": "BDH's activation vectors are sparse and positive, and the paper reports "
                     "nonassociativity - individual synapses that respond to a specific concept "
                     "(e.g. a country name or a currency).",
        "paper": "bdh2025",
        "maps_to": "The ReLU + top-k sparse positive activation used by the BDH-inspired mode "
                   "here, and why sparse keys reduce crosstalk in the memory visualiser.",
    },
    {
        "id": "c5",
        "tier": "reported_experiment",
        "statement": "Recall quality on multi-query associative recall is bounded by recurrent "
                     "state size: shrinking the state monotonically degrades recall, and "
                     "architectures trade recall against throughput along that curve.",
        "paper": "zoology2024",
        "maps_to": "The memory-capacity slider. Power pilots really does mean worse recall here, "
                   "and that direction of effect is the published one.",
    },
    {
        "id": "c6",
        "tier": "reported_experiment",
        "statement": "Replacing decay-based writes with a delta-rule update plus a gated erase "
                     "more measurably improves in-context retrieval and reduces memory "
                     "collisions relative to plain gated decay.",
        "paper": "gateddelta2024",
        "maps_to": "Why the update-rate slider changes the outcome at all: the write rule, not "
                   "the state size, decides whether an old binding survives.",
    },
    {
        "id": "c7",
        "tier": "reported_experiment",
        "statement": "Recurrent models with oversized states fail to forget stale context; "
                     "beyond their training length, retained information actively interferes "
                     "with retrieval.",
        "paper": "forgetting2024",
        "maps_to": "The 'Stale Memory' and 'Distribution Shift' failure presets.",
    },
    {
        "id": "c8",
        "tier": "interpretation",
        "statement": "Because a fixed-size state must compress history, any such model can hold "
                     "a binding that the context has already superseded. Staleness is therefore "
                     "a structural consequence of compression, not an implementation bug.",
        "paper": "bdh2025",
        "maps_to": "The falsifiable claim this whole artifact tests. This is OUR reading; none "
                   "of the cited papers phrase it this way.",
    },
    {
        "id": "c9",
        "tier": "interpretation",
        "statement": "An additive Hebbian write recovers from a changed fact more readily than a "
                     "convex blended write, because the newer trace is simply less decayed - no "
                     "large update rate is required.",
        "paper": "bdh2025",
        "maps_to": "The difference you observe between the Persistent Memory and BDH-inspired "
                   "modes on the Changing Facts task. Demonstrated on this toy only.",
    },
    {
        "id": "s1",
        "tier": "simplification",
        "statement": "The three modes on this page are UNTRAINED, fixed-weight numerical "
                     "simulations over a hashed embedding table of dimension 32. There is no "
                     "learning, no gradient descent, and no tokeniser.",
        "paper": "",
        "maps_to": "Everything you see run. Treat it as a mechanism diagram that computes, not "
                   "as a model.",
    },
    {
        "id": "s2",
        "tier": "simplification",
        "statement": "The BDH-inspired mode borrows three ideas - sparse positive activations, "
                     "an inference-time mutable synaptic state, and a linear-attention-style "
                     "outer-product write. It does NOT implement BDH: no neuron-level graph, no "
                     "low-rank factorised interaction matrices, no multi-layer network, no "
                     "training. It is not a representation and must not be cited as one.",
        "paper": "bdh2025",
        "maps_to": "The BDH-inspired mode card. This is the single most important caveat here.",
    },
    {
        "id": "s3",
        "tier": "simplification",
        "statement": "'Memory slots' are an explicit, human-readable abstraction. Real recurrent "
                     "models hold distributed superposed state with no addressable slots; slots "
                     "exist here only so eviction and staleness are visible.",
        "paper": "",
        "maps_to": "The memory state matrix visualiser.",
    },
    {
        "id": "s4",
        "tier": "simplification",
        "statement": "Attention here is a single head with a hand-set recency bias "
                     "(beta = 0.55) and no learned projections, so it resolves a rewound fact "
                     "by position. A trained model would learn this behaviour instead.",
        "paper": "",
        "maps_to": "Why the Full-Context Attention mode gets Changing Facts right.",
    },
]

GLOSSARY: List[Dict[str, str]] = [
    {"term": "KV cache","definition": "The stored key and value vectors for every token seen "
    "so far. Nothing is compressed, so nothing goes stale - but it grows as O(N).",},
    {"term": "Persistent / recurrent state","definition": "A fixed-size buffer carried from "
    "step to step. Cost is independent of sequence length, so history must be compressed "
    "into it - which is where staleness comes from.",},
    {"term": "Update rate (alpha)","definition": "How much of a new write replaces what a slot "
    "already holds. Below 0.5 a convex blend leaves the OLD value dominant.",},
    {"term": "Decay (gamma)","definition": "Per-step multiplicative shrinkage of the state. "
     "Prevents unbounded accumulation, but erases old facts across long gaps.",},
    {"term": "Eviction / capacity overflow","definition": "When more distinct keys arrive than "
    "there are slots, something must be overwritten. If it was the fact you needed, recall "
    "fails.",},
    {"term": "Stale state","definition": "The state confidently returns a binding the context "
    "has already superseded. The central failure mode this artifact demonstrates.",},
    {"term": "Hebbian / fast-weight write","definition": "Adding an outer product of a value and "
    "a key into a matrix state. Additive rather than blended, so newer traces dominate "
    "naturally.",},
    {"term": "Sparse positive activations","definition": "An activation vector that is "
    "non-negative and mostly zero after ReLU + top-6 of 32. Reported in BDH and associated "
    "with monosemantic synapses.",},
    {"term": "Monosemanticity","definition": "One unit or synapse corresponding to one "
    "human-interpretable concept, rather than a mixture.",},
    {"term": "Superposition interference","definition": "Many traces stored in one matrix "
    "cross-talk, so a readout decodes to a blend rather than a clean answer.",},
    {"term": "Associative recall (MQAR)","definition": "The standard benchmark family behind "
    "these tasks: bind keys to values in context, then query them.",},
    {"term": "BDH / BDH-GPU","definition": "The Dragon Hatchling - Pathway's neuron-and-synapse "
     "graph architecture, and its tensorized GPU variant using low-rank factorisation with "
     "linear attention.",},
]

LIMITATIONS: List[str] = [
    "No model is trained. All three modes are fixed-weight numerical simulations, so absolute "
    "accuracy numbers are meaningless - only the DIRECTION of each effect is the lesson.",
    "The BDH-inspired mode is inspired by, and is not, BDH. See simplification s2.",
    "Sequence length is capped at 128 tokens and memory at 16 slots server-side. Real "
    "long-context effects appear at 10^4-10^5 tokens.",
    "The embedding table is hashed, not learned, so semantic similarity between symbols is "
    "arbitrary by construction.",
    "The client-side fallback simulator implements the same algorithm but with a different "
    "PRNG and embedding table, so its numbers will not match the server bit-for-bit. It is "
    "labelled 'fallback' whenever it is used.",
    "Nothing here measures throughput or wall-clock efficiency of real architectures: the state "
    "size figures are analytic float counts, not benchmarks.",
]

AI_DISCLOSURE = (
    "This artifact was built with AI assistance [Reagent AI agent, Claude-family model]. The "
    "agent authored the FastAPI simulator, the TypeScript fallback, the React interface, and "
    "the first draft of the explanatory copy. All research claims were checked against the "
    "primary sources listed in the evidence panel, and every statement is tagged with the tier "
    "of evidence supporting it. Paper titles, venues, and identifiers were verified against "
    "arXiv and the official Pathway repository before publication."
)

FAILURE_PRESETS: List[Dict[str, Any]] = [
    {
        "id": "stale_memory",
        "name": "Stale Memory",
        "subtitle": "A fact changes; the blended state keeps the old one",
        "expect": "Attention corrects. Persistent Memory returns the SUPERSEDED value with high "
                  "confidence. BDH-inspired recovers because its write is additive.",
        "config": {"task": "changing_facts","seq_len": 48,"mem_slots": 6,
            "update_rate": 0.25,"noise": 0.0,"decay": 0.01,
            "ood_query": False,"reset_midway": False,"seed": 7,},
    },
    {
        "id": "capacity_overflow",
        "name": "Capacity Overflow",
        "subtitle": "More distinct keys than slots - something gets evicted",
        "expect": "Attention recalls fine. Persistent Memory evicts the needed fact and fails.",
        "config": {"task": "distractor_recall","seq_len": 64,"mem_slots": 2,
            "update_rate": 0.9,"noise": 0.01,"decay": 0.01,
            "ood_query": False,"reset_midway": False,"seed": 7,},
    },
    {
        "id": "noisy_updates",
        "name": "Noisy Updates",
        "subtitle": "Every re-affirmation is perturbed; error accumulates",
        "expect": "All three degrade. Persistent state degrades WORSE because each noisy write "
                  "is permanently folded into the state it carries forward.",
        "config": {"task": "noisy_prediction", "seq_len": 56, "mem_slots": 6,
                   "update_rate": 0.5, "noise": 0.42, "decay": 0.02,
                   "ood_query": False, "reset_midway": False, "seed": 7},
    },
    {
        "id": "reset_loss",
        "name": "Reset Loss",
        "subtitle": "State cleared mid-sequence; the fact is never restated",
        "expect": "Attention is unaffected - it never compressed anything. Both persistent modes "
                  "lose the fact completely.",
        "config": {"task": "delayed_recall", "seq_len": 48, "mem_slots": 6,
                   "update_rate": 0.6, "noise": 0.0, "decay": 0.01,
                   "ood_query": False, "reset_midway": True, "seed": 7},
    },
    {
        "id": "distribution_shift",
        "name": "Distribution Shift",
        "subtitle": "The query asks about a key that was never bound",
        "expect": "The honest answer is UNKNOWN. Watch which modes abstain and which hallucinate "
                  "a confident value out of crosstalk.",
        "config": {"task": "delayed_recall", "seq_len": 40, "mem_slots": 6,
                   "update_rate": 0.6, "noise": 0.1, "decay": 0.01,
                   "ood_query": True, "reset_midway": False, "seed": 7},
    },
    {
        "id": "memory_wins",
        "name": "Memory Wins (the benefit)",
        "subtitle": "Long gap, decisive writes, room to spare",
        "expect": "Persistent Memory matches Attention using a state ~10x smaller and constant "
                  "in N. This is the BENEFIT half of the claim.",
        "config": {"task": "delayed_recall", "seq_len": 120, "mem_slots": 8,
                   "update_rate": 0.85, "noise": 0.0, "decay": 0.0,
                   "ood_query": False, "reset_midway": False, "seed": 7},
    }
]