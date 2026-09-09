// Hand-written mirrors of the Pydantic models in backend/models/simulation.py.
// Nothing infers across the Python boundary — change both files in the same edit.

export interface RunConfig {
  task: string;
  seq_len: number;
  mem_slots: number;
  update_rate: number;
  noise: number;
  decay: number;
  ood_query: boolean;
  reset_midway: boolean;
  seed: number;
}

export interface Token {
  index: number;
  kind: string;
  label: string;
  key: string | null;
  value: string | null;
  is_stale_fact: boolean;
  is_noisy: boolean;
}
 
export interface SlotState {
  slot: number;
  key: string | null;
  value: string | null;
  declared_value: string | null;
  strength: number;
  age: number;
  occupied: boolean;
}

export interface StepState {
  step: number;
  token_index: number;
  write_slot: number | null;
  event: string;
  slots: SlotState[];
  energy?: number | null;
  active_units?: number | null;
}
 
export interface ModeResult {
  mode: string;
  label: string;
  prediction: string;
  correct: boolean;
  confidence: number;
  margin: number;
  attention: number[];
  top_token_index: number | null;
  retrieved_key: string | null;
  retrieved_slot: number | null;
  retrieved_age: number;
  interference: number;
  state_floats: number;
  state_scaling: string;
  steps: StepState[];
  events: string[];
  verdict: string;
  explanation: string;
  energy?: number | null;
  evictions?: number | null;
}

export interface RunResult {
  run_id: string;
  config: RunConfig;
  tokens: Token[];
  query_key: string;
  query_index: number;
  ground_truth: string;
  reset_at: number | null;
  notes: string[];
  modes: ModeResult[];
  latency_ms: number;
  computation_status: string;
  engine: string;
  caps: Caps;
}

export interface CapRange {
  min: number;
  max: number;
  default: number;
  step?: number;
}

export interface Caps {
  seq_len: CapRange;
  mem_slots: CapRange;
  update_rate: CapRange;
  noise: CapRange;
  decay: CapRange;
  seed: CapRange;
  dim: number;
  max_steps_recorded: number;
  rationale: string;
}
 
export interface TaskInfo {
  id: string;
  name: string;
  tagline: string;
  tests: string;
  ground_truth: string;
}

export interface ModeCard {
  id: string;
  label: string;
  kind: string;
  state: string;
  summary: string;
  honesty: string;
}

export interface Paper {
  key: string;
  title: string;
  authors: string;
  year: string;
  venue: string;
  url: string;
  code: string;
  why: string;
}

export interface EvidenceClaim {
  id: string;
  tier: string;
  statement: string;
  paper: string;
  maps_to: string;
}

export interface GlossaryEntry {
  term: string;
  definition: string;
}

export interface Preset {
  id: string;
  name: string;
  subtitle: string;
  expect: string;
  config: RunConfig;
}

  export interface MetaResponse {
  claim: string;
  caps: Caps;
  tasks: TaskInfo[];
  modes: ModeCard[];
  model_math: Record<string, string>;
  constants: Record<string, string | number>;
  papers: Paper[];
  evidence: EvidenceClaim[];
  glossary: GlossaryEntry[];
  limitations: string[];
  ai_disclosure: string;
  presets: Preset[];
}
 
export interface HealthResponse {
  status: string;
  engine: string;
  db_connected: boolean;
  runs_recorded: number;
}

export interface LearnerResponseCreate {
  run_id: string | null;
  task: string;
  predicted_failures: string[];
  explanation: string;
  self_score: number | null;
}

export interface LearnerResponse {
  id: string;
  accepted: boolean;
  rubric: string[];
  stored: boolean;
  note: string;
}