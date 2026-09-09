// Client-side FALLBACK simulator. Implements the same three algorithms as
// backend/lib/simulator.py, but with a different PRNG and embedding table, so its
// numbers will NOT match the server bit-for-bit. Any result produced here is
// labelled "fallback" in the UI. Its scope: identical mechanisms and identical
// direction of every effect; not identical values.
 
import type { ModeResult, RunConfig, RunResult, SlotState, StepState, Token } from "./types";
import { FALLBACK_CAPS } from "./fallbackMeta";
 
const DIM = 32;
const BETA = 0.55;
const TAU = 0.55;
const KSPARSE = 6;
const ABSTAIN = 0.16;
 
const KEYS = ["ALPHA", "BETA", "GAMMA", "DELTA", "EPSILON", "ZETA", "ETA", "THETA", "IOTA", "KAPPA", "LAMBDA", "MU"];
const VALUES = ["red", "blue", "green", "amber", "violet", "teal", "coral", "olive", "cyan", "rose", "slate","lime"];
const FILLERS = ["the", "and", "of", "a", "then", "so", "with", "for", "to", "in", "on", "by"];

function mulberry32(a: number): () => number {
  return () => {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
 
function gauss(rnd: () => number): number {
  const u = Math.max(rnd(), 1e-9);
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * rnd());
}

function strHash(s: string): number {
  let h = 2166136261;
  for (let i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return h >>> 0;
}

const embedCache = new Map<string, number[]>();
function embed(sym: string): number[] {
  const hit = embedCache.get(sym);
  if (hit) return hit;
  const rnd = mulberry32(strHash(sym));
  const v = Array.from({ length: DIM }, () => gauss(rnd));
  const n = Math.hypot(...v) || 1;
  const out = v.map((x) => x / n);
  embedCache.set(sym, out);
  return out;
}

const dot = (a: number[], b: number[]) => a.reduce((s, x, i) => s + x * b[i], 0);
const norm = (a: number[]) => Math.hypot(...a);
const cos = (a: number[], b: number[]) => dot(a, b) / ((norm(a) * norm(b)) || 1e-9);
const zeros = () => new Array<number>(DIM).fill(0);

const VALUE_VECS = VALUES.map((v) => embed(v));
 
function decode(readout: number[]): { value: string; conf: number; margin: number } {
  const n = norm(readout);
  if (n < 1e-8) return { value: "UNKNOWN", conf: 0, margin: 0 };
  const unit = readout.map((x) => x / n);
  const sims = VALUE_VECS.map((v) => dot(v, unit)).map((s, i) => ({ s, i }));
  sims.sort((a, b) => b.s - a.s);
  const margin = sims[0].s - sims[1].s;
  if (sims[0].s < ABSTAIN) return { value: "UNKNOWN", conf: sims[0].s, margin };
  return { value: VALUES[sims[0].i], conf: sims[0].s, margin };
}
 
function sparsify(v: number[]): number[] {
  const pos = v.map((x) => Math.max(x, 0));
  const idx = pos.map((x, i) => ({ x, i })).sort((a, b) => b.x - a.x).slice(KSPARSE).map((o) => o.i);
  for (const i of idx) pos[i] = 0;
  const n = norm(pos) || 1;
  return pos.map((x) => x / n);
}

interface Seq {
  tokens: Token[];
  queryKey: string;
  queryIndex: number;
  groundTruth: string;
  resetAt: number | null;
  notes: string[];
}
 
function buildSequence(cfg: RunConfig): Seq {
  const rnd = mulberry32(cfg.seed + 1);
  const n = cfg.seq_len;
  const tokens: Token[] = Array.from({ length: n }, (_, i) => ({
    index: i, kind: "distractor", label: FILLERS[Math.floor(rnd() * FILLERS.length)],
    key: null, value: null, is_stale_fact: false, is_noisy: false,
  }));
  const notes: string[] = [];
  let targetKey = KEYS[0];
  const queryIndex = n - 1;
  let groundTruth = VALUES[0];
 
  const place = (pos: number, key: string, value: string, noisy = false) => {
    const p = Math.max(0, Math.min(pos, n - 2));
    tokens[p] = { index: p, kind: "fact", label: `${key}=${value}`, key, value, is_stale_fact: false, is_noisy: noisy };
};
 
if (cfg.task === "changing_facts") {
  const a = Math.max(1, Math.floor(n * 0.1));
  const b = Math.max(2, Math.floor(n * 0.62));
  place(a, targetKey, VALUES[0]);
  place(b, targetKey, VALUES[3]);
  tokens[a].is_stale_fact = true;
  groundTruth = VALUES[3];
  notes.push(`${targetKey} is bound to '${VALUES[0]}' at step ${a}, then RE-BOUND to '${VALUES[3]}' at step${b}. Only the newest binding is correct.`);
} else if (cfg.task === "distractor_recall") {
  place(1, targetKey, VALUES[0]);
  const competing = cfg.mem_slots + 4;
  const free = Array.from({ length: Math.max(0, n - 5) }, (_, i) => i + 3).sort(() => rnd() - 0.5);
  for (let c = 0; c < Math.min(competing, free.length); c++) {
    place(free[c], KEYS[1 + (c % (KEYS.length - 1))], VALUES[1 + (c % (VALUES.length - 1))]);
  }
  groundTruth = VALUES[0];
  notes.push(`${competing} facts about OTHER keys compete for ${cfg.mem_slots} memory slots. Bounded capacity must evict something.`);
} else if (cfg.task === "noisy_prediction") {
  const count = Math.min(6, Math.max(2, Math.floor(n / 8)));
  for (let i = 0; i < count; i++) {
    const p = 1 + Math.round((i * (n - 4)) / Math.max(1, count - 1));
    place(p, targetKey, VALUES[0], i > 0);
  }
  groundTruth = VALUES[0];
  notes.push(`${targetKey} is re-affirmed as '${VALUES[0]}' ${count} times. Every re-affirmation after the first has noise added to its embedding, so noisy writes accumulate in any persistent state.`);
} else {
  place(1, targetKey, VALUES[0]);
  groundTruth = VALUES[0];
  notes.push(`${targetKey}='${VALUES[0]}' appears at step 1 and is never mentioned again before the query at step ${queryIndex}.`);
}
 
if (cfg.ood_query) {
  targetKey = KEYS[KEYS.length - 1];
  groundTruth = "UNKNOWN";
  notes.push(`DISTRIBUTION SHIFT: the query asks about ${targetKey}, which never appears as a fact. The honest answer is UNKNOWN — any confident value is a hallucination from crosstalk in the state.`);
}

tokens[queryIndex] = { index: queryIndex, kind: "query", label: `${targetKey}=?`, key: targetKey, value: null, is_stale_fact: false, is_noisy: false };
const resetAt = cfg.reset_midway ? Math.floor(n * 0.5) : null;
if (resetAt !== null) {
  notes.push(`RESET LOSS: persistent state is explicitly cleared at step ${resetAt}. Full-context attentionis unaffected because it never compressed anything.`);
}
return { tokens, queryKey: targetKey, queryIndex, groundTruth, resetAt, notes };
}
 
function runAttention(seq: Seq, cfg: RunConfig): ModeResult {
  const rnd = mulberry32(cfg.seed + 101);
  const n = seq.tokens.length;
  const keys: number[][] = [];
  const vals: number[][] = [];
  for (const t of seq.tokens) {
    const base = t.key ? embed(t.key) : embed(`<fill:${t.label}>`);
    const pert = cfg.noise * (t.is_noisy ? 1.5 : 1);
    keys.push(base.map((x) => x + pert * gauss(rnd)));
    vals.push(t.value ? embed(t.value).map((x) => x + pert * gauss(rnd)) : zeros());
  }
  const q = embed(seq.queryKey).map((x) => x + cfg.noise * gauss(rnd));
  const scores = keys.map((k, i) => dot(k, q) / Math.sqrt(DIM) + BETA * (i / Math.max(1, n)));
  scores[seq.queryIndex] = -1e9;
  const mx = Math.max(...scores);
  const ex = scores.map((s) => Math.exp(s - mx));
  const sum = ex.reduce((a, b) => a + b, 0);
  const w = ex.map((x) => x / sum);
  const readout = zeros().map((_, d) => w.reduce((s, wi, i) => s + wi * vals[i][d], 0));
  const { value, conf, margin } = decode(readout);
  const top = w.indexOf(Math.max(...w));
  const correct = value === seq.groundTruth;
  return {
    mode: "attention", label: "Full-Context Attention", prediction: value, correct,
    confidence: +conf.toFixed(4), margin: +margin.toFixed(4),
    attention: w.map((x) => +x.toFixed(5)), top_token_index: top,
    retrieved_key: seq.tokens[top].key, retrieved_slot: null, retrieved_age: 0,
    interference: +(1 - Math.max(...w)).toFixed(4),
    state_floats: 2 * n * DIM, state_scaling: "O(N · d) — grows with every token",
    steps: [], events: [], verdict: correct ? "success" : "failure",
    explanation: correct
      ? `Nothing was compressed: all ${n} key/value pairs stayed in the cache, so the fact was re-read directly. The cost is a state of ${2 * n * DIM} floats that grows with every token.`
      : `Noise (sigma=${cfg.noise.toFixed(2)}) perturbed the key vectors enough that the highest-weighted token was the wrong one. Attention never goes stale, but it is not noise-proof — it retrieved '${value}'instead of '${seq.groundTruth}'.`,
  };
}
 
/**
 * M content-addressed slots written with a convex blend. Mirrors the Python
 * _SlotMemory class so the two engines stay structurally comparable: one rule of the
 * mechanism per method. Staleness lives entirely in write().
 */
class SlotMemory {
  kmem: number[][];
  smem: number[][];
  strength: number[];
  age: number[];
  occupied: boolean[];
  slotKey: (string | null)[];
  slotVal: (string | null)[];
  steps: StepState[] = [];
  events: string[] = [];
  evictions = 0;
  readonly m: number;
  readonly alpha: number;
  readonly decay: number; 

  constructor(m: number, alpha: number, decay: number) {
    this.m = m;
    this.alpha = alpha;
    this.decay = decay;
    this.kmem = Array.from({ length: m }, zeros);
    this.smem = Array.from({ length: m }, zeros);
    this.strength = new Array<number>(m).fill(0);
    this.age = new Array<number>(m).fill(0);
    this.occupied = new Array<boolean>(m).fill(false);
    this.slotKey = new Array<string | null>(m).fill(null);
    this.slotVal = new Array<string | null>(m).fill(null);
  }

  clear(): void {
    this.kmem = Array.from({ length: this.m }, zeros);
    this.smem = Array.from({ length: this.m }, zeros);
    this.strength = new Array<number>(this.m).fill(0);
    this.age = new Array<number>(this.m).fill(0);
    this.occupied = new Array<boolean>(this.m).fill(false);
    this.slotKey = new Array<string | null>(this.m).fill(null);
    this.slotVal = new Array<string | null>(this.m).fill(null);
  }

  /** Content match above TAU, else a free slot, else evict the weakest. */
  selectSlot(k: number[]): { slot: number; event: string } {
    const sims = this.kmem.map((km, j) => (this.occupied[j] ? cos(k, km) : -1));
    const best = sims.indexOf(Math.max(...sims));
    if (this.occupied.some(Boolean) && sims[best] > TAU) return { slot: best, event: "blend" };
    if (this.occupied.some((o) => !o)) return { slot: this.occupied.indexOf(false), event: "allocate" };
    return { slot: this.strength.indexOf(Math.min(...this.strength)), event: "evict" };
  }

  evict(j: number): void {
    this.evictions++;
    this.kmem[j] = zeros();
    this.smem[j] = zeros();
  }

  write(j: number, event: string, key: string, value: string, k: number[], v: number[]): void {
    if (event === "blend") {
      this.kmem[j] = this.kmem[j].map((x, d) => (1 - this.alpha) * x + this.alpha * k[d]);
      this.smem[j] = this.smem[j].map((x, d) => (1 - this.alpha) * x + this.alpha * v[d]);
      this.strength[j] = (1 - this.alpha) * this.strength[j] + this.alpha;
    } else {
      this.kmem[j] = k;
      this.smem[j] = v;
      this.strength[j] = 1;
    }
    this.occupied[j] = true;
    this.slotKey[j] = key;
    this.slotVal[j] = value;
    this.age[j] = 0;
  }

  decayStep(): void {
    this.smem = this.smem.map((s) => s.map((x) => x * (1 - this.decay)));
    this.strength = this.strength.map((x) => x * (1 - this.decay));
    this.age = this.age.map((a, j) => (this.occupied[j] ? a + 1 : a));
  }
 
  read(queryKey: string): { slot: number; readout: number[] } {
    const q = embed(queryKey);
    const sims = this.kmem.map((km, j) => (this.occupied[j] ? cos(q, km) : -1));
    const slot = this.occupied.some(Boolean) ? sims.indexOf(Math.max(...sims)) : -1;
    const readout = slot >= 0 && sims[slot] > 0.2 ? this.smem[slot] : zeros();
    return { slot, readout };
  }

  snapshot(tokenIndex: number, writeSlot: number | null, event: string): void {
    const slots: SlotState[] = Array.from({ length: this.m }, (_, j) => ({
      slot: j,
      key: this.slotKey[j],
      value: this.occupied[j] ? decode(this.smem[j]).value : null,
      declared_value: this.slotVal[j],
      strength: +this.strength[j].toFixed(4),
      age: this.age[j],
      occupied: this.occupied[j],
   }));
   this.steps.push({
     step: this.steps.length, token_index: tokenIndex, write_slot: writeSlot, event, slots,
   });
    }
 }
 
 /** Apply one token to the memory. Returns the slot written and the event kind. */
 function memoryProcessToken(
   mem: SlotMemory, t: Token, seq: Seq, cfg: RunConfig, rnd: () => number,
 ): { writeSlot: number | null; event: string } {
   if (t.kind !== "fact" || !t.key || !t.value) return { writeSlot: null, event: "idle" };
 
   const pert = cfg.noise * (t.is_noisy ? 1.5 : 1);
   const k = embed(t.key).map((x) => x + pert * gauss(rnd));
   const v = embed(t.value).map((x) => x + pert * gauss(rnd));
 
   const { slot, event } = mem.selectSlot(k);
   if (event === "evict") {
     if (mem.slotKey[slot] === seq.queryKey) 
       {mem.events.push(`CAPACITY OVERFLOW: the slot holding ${seq.queryKey} was evicted at step ${t.index} to make room for ${t.key}.`);
      }
      mem.evict(slot);
     }
     mem.write(slot, event, t.key, t.value, k, v);
     return { writeSlot: slot, event };
 }
 
 /** Plain-language reason this mode succeeded or failed. Mirrors _verdict_memory. */
 function memoryExplanation(
   correct: boolean, value: string, seq: Seq, cfg: RunConfig, mem: SlotMemory, readSlot: number,
 ): string {
   const m = cfg.mem_slots;
   const alpha = cfg.update_rate;
   if (correct) {
     return `A fixed ${2 * m * DIM}-float state carried the fact across the gap without replaying the context — this is the BENEFIT half of the claim. Retrieved from slot ${readSlot}, age ${readSlot >= 0 ? mem.age[readSlot] : 0} steps.`;
   }
   if (seq.resetAt !== null) {
     return "The state was cleared mid-sequence and the fact was never re-stated, so there was nothing left toread. Compressed state is destroyed by a reset; a full context is not.";
 }
 if (mem.evictions > 0 && cfg.task === "distractor_recall") {
   return `${mem.evictions} eviction(s) occurred: more distinct keys arrived than the ${m} available slots, so the needed fact was overwritten by a distractor. This is capacity overflow — the FAILURE half of theclaim.`;
 }
 if (cfg.task === "changing_facts") {
   return `The slot was updated with a convex blend at alpha=${alpha.toFixed(2)}, so it still holds (${(1 - alpha).toFixed(2)})·'${VALUES[0]}' + (${alpha.toFixed(2)})·'${seq.groundTruth}'. The superseded value stays dominant and the model answers '${value}' with confidence. This is STALE STATE — raise the update rate above 0.5 and it corrects itself.`;
 }
 return `The persistent state decoded to '${value}' instead of '${seq.groundTruth}'. Noisy or decayed writes corrupted the stored vector — accumulated error, not a missing fact.`;
 }

 function runMemory(seq: Seq, cfg: RunConfig): ModeResult {
   const rnd = mulberry32(cfg.seed + 202);
   const m = cfg.mem_slots;
   const mem = new SlotMemory(m, cfg.update_rate, cfg.decay);

   for (const t of seq.tokens) {
     if (seq.resetAt !== null && t.index === seq.resetAt) {
       mem.clear();
       mem.events.push(`State cleared at step ${t.index} (explicit reset).`);
       mem.snapshot(t.index, null, "reset");
     }

     const { writeSlot, event } = memoryProcessToken(mem, t, seq, cfg, rnd);

     mem.decayStep();
     if (t.kind === "fact" || t.index === seq.queryIndex) mem.snapshot(t.index, writeSlot, event);
   }
 
   const { slot: readSlot, readout } = mem.read(seq.queryKey);
   const { value, conf, margin } = decode(readout);
   const declared = readSlot >= 0 ? mem.slotVal[readSlot] : null;
 
   if (readSlot >= 0 && value !== "UNKNOWN" && declared && value !== declared) {
     mem.events.push(`STALE BLEND: slot ${readSlot} was last written with '${declared}' but the blended vectorstill decodes to '${value}' — a convex write with alpha=${cfg.update_rate.toFixed(2)} leaves the superseded value dominant.`);
   }

   const correct = value === seq.groundTruth;
   return {
     mode: "memory", label: "Bounded Persistent Memory", prediction: value, correct,
     confidence: +conf.toFixed(4), margin: +margin.toFixed(4), attention: [], top_token_index: null,
     retrieved_key: readSlot >= 0 ? mem.slotKey[readSlot] : null,
     retrieved_slot: readSlot >= 0 ? readSlot : null,
     retrieved_age: readSlot >= 0 ? mem.age[readSlot] : 0,
     interference: +Math.max(0, 1 - margin * 4).toFixed(4),
     state_floats: 2 * m * DIM,
     state_scaling: `O(M · d) = ${2 * m * DIM} floats — constant in N`,
     steps: mem.steps.slice(0, FALLBACK_CAPS.max_steps_recorded),
     events: mem.events,
     evictions: mem.evictions,
     verdict: correct ? "success" : "failure",
     explanation: memoryExplanation(correct, value, seq, cfg, mem, readSlot),
   };
 }
 
 function runBdh(seq: Seq, cfg: RunConfig): ModeResult {
   const rnd = mulberry32(cfg.seed + 303);
   let sigma = Array.from({ length: DIM }, zeros);
   const steps: StepState[] = [];
   const events: string[] = [];
   const lam = cfg.decay;
   const eta = Math.max(0.15, cfg.update_rate);
 
   for (const t of seq.tokens) {
     if (seq.resetAt !== null && t.index === seq.resetAt) {
       sigma = Array.from({ length: DIM }, zeros);
       events.push(`Synaptic state cleared at step ${t.index} (explicit reset).`);
       steps.push({ step: steps.length, token_index: t.index, write_slot: null, event: "reset", slots: [], energy: 0, active_units: 0 });
     }
     let event = "idle";
     let active = 0;
     if (t.kind === "fact" && t.key && t.value) {
       const pert = cfg.noise * (t.is_noisy ? 1.5 : 1);
       const xk = sparsify(embed(t.key).map((x) => x + pert * gauss(rnd)));
       const xv = sparsify(embed(t.value).map((x) => x + pert * gauss(rnd)));
       sigma = sigma.map((row, r) => row.map((x, c) => (1 - lam) * x + eta * xv[r] * xk[c]));
       event = "hebbian_write";
       active = xk.filter((x) => x > 0).length;
     } else {
       sigma = sigma.map((row) => row.map((x) => (1 - lam) * x));
     }
     if (t.kind === "fact" || t.index === seq.queryIndex) {
       const energy = Math.hypot(...sigma.flat());
       steps.push({ step: steps.length, token_index: t.index, write_slot: null, event, slots: [], energy: +energy.toFixed(4), active_units: active });
     }
   }
 
   const xq = sparsify(embed(seq.queryKey));
   const readout = sigma.map((row) => dot(row, xq));
   const { value, conf, margin } = decode(readout);
   const energy = Math.hypot(...sigma.flat());
   if (energy < 1e-3) events.push("DECAY WIPEOUT: the synaptic state decayed to near zero before the query. Anadditive Hebbian trace has no protected slot — long gaps erase it.");
   if (margin < 0.05 && value !== "UNKNOWN") events.push("SUPERPOSITION INTERFERENCE: several traces decode almost equally well, so this answer is a crosstalk artefact rather than a clean retrieval.");
   const correct = value === seq.groundTruth;
   let explanation: string;
   if (correct) {
     explanation = `Hebbian writes are ADDITIVE, so the newest fact laid a fresh, less-decayed trace that outweighs the older one without needing a large update rate. Sparse positive keys (top-${KSPARSE} of ${DIM}) keep crosstalk low. State: ${DIM * DIM} floats, constant in N.`;
   } else if (energy < 1e-3) {explanation = "The trace decayed away before the query arrived. An additive fast-weight state has no protected slot to defend an old fact — this is the cost of not having a cache.";
     explanation = "The trace decayed away before the query arrived. An additive fast-weight state has no protected slot to defend an old fact — this is the cost of not having a cache.";
   } else {
     explanation = `Traces superposed into interference and the readout decoded to '${value}' instead of '${seq.groundTruth}'. A fixed d×d state has finite capacity too; it degrades gracefully rather than evicting, but it still degrades.`;
 }
 return {
   mode: "bdh", label: "BDH-Inspired Sparse Hebbian", prediction: value, correct,
   confidence: +conf.toFixed(4), margin: +margin.toFixed(4), attention: [], top_token_index: null,
   retrieved_key: seq.queryKey, retrieved_slot: null, retrieved_age: 0,
   interference: +Math.max(0, 1 - margin * 4).toFixed(4),
   state_floats: DIM * DIM, state_scaling: `O(d²) = ${DIM * DIM} floats — constant in N`,
   steps: steps.slice(0, FALLBACK_CAPS.max_steps_recorded), events,
   energy: +energy.toFixed(4), verdict: correct ? "success" : "failure", explanation,
 };
}

/** Run the whole comparison locally. Result is always flagged computation_status "fallback". */
export function simulateLocal(cfg: RunConfig): RunResult {
  const t0 = performance.now();
  const seq = buildSequence(cfg);
  const modes = [runAttention(seq, cfg), runMemory(seq, cfg), runBdh(seq, cfg)];
  return {
    run_id: `local-${cfg.seed}-${Math.random().toString(36).slice(2, 8)}`,
    config: cfg,
    tokens: seq.tokens,
    query_key: seq.queryKey,
    query_index: seq.queryIndex,
    ground_truth: seq.groundTruth,
    reset_at: seq.resetAt,
    notes: [
      ...seq.notes,
      "FALLBACK ENGINE: computed in your browser because the server was unreachable. Same algorithms, different PRNG and embedding table — the direction of every effect holds, the exact numbers will not match the server.",
    ],
    modes,
    latency_ms: +(performance.now() - t0).toFixed(3),
    computation_status: "fallback",
    engine: "client-typescript-fallback",
    caps: FALLBACK_CAPS,
  };
 }