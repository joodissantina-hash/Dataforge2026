// Presentation constants. Kept out of the component files so Fast Refresh keeps
// working (a module that exports both components and constants breaks it).

export const MODE_META: Record<
 string,
 { color: string; bg: string; border: string; short: string }
> = {
  attention: { color: "#38BDF8", bg: "#132338", border: "#0284C7", short: "Attention" },
  memory: { color: "#FB7185", bg: "#2A1820", border: "#E11D48", short: "Memory" },
  bdh: { color: "#34D399", bg: "#0E2822", border: "#059669", short: "BDH-inspired" },
 };

 export const TIER_META: Record<string, { label: string; bg: string; fg: string }> = {
   formal_result: { label: "Formal result", bg: "#1E1B4B", fg: "#C7D2FE" },
   reported_experiment: { label: "Reported experiment", bg: "#064E3B", fg: "#A7F3D0" },
   interpretation: { label: "Our interpretation", bg: "#78350F", fg: "#FDE68A" },
   simplification: { label: "Simplification in this artifact", bg: "#374151", fg: "#E5E7EB" },
};

export const EVIDENCE_TIERS = [
  "formal_result",
  "reported_experiment",
  "interpretation",
  "simplification",
] as const;
 
const RUN_STATUS_BADGES = {
  live: { text: "LIVE COMPUTE", bg: "#065F46", fg: "#A7F3D0" },
  fallback: { text: "FALLBACK (IN-BROWSER)", bg: "#713F12", fg: "#FDE047" },
  illustrative: { text: "ILLUSTRATIVE", bg: "#1E3A8A", fg: "#BFDBFE" },
  idle: { text: "NOT YET RUN", bg: "#1B2432", fg: "#94A3B8" },
} as const;
 
export type RunStatusKind = keyof typeof RUN_STATUS_BADGES;

export const runStatusBadge = (kind: RunStatusKind) => RUN_STATUS_BADGES[kind];

/** Colour a prediction by whether it matched ground truth. */
export const predictionColor = (correct: boolean) => (correct ? "#34D399" : "#F87171");

/** Border colour for a mode card: mode accent when correct, red when not. */
export const modeCardBorder = (mode: string, correct: boolean) =>
  correct ? MODE_META[mode].border : "#7F1D1D";
 
/**
 * Background for one token in the sequence strip. Replaces a nested ternary so each
 * token kind is an explicit, named case.
*/
export function tokenBackground(kind: string, attentionRatio: number): string {
  if (kind === "query") return "#3B2A62";
  if (kind === "fact") return `rgba(251, 191, 36, ${0.16 + 0.5 * attentionRatio})`;
  return `rgba(56, 189, 248, ${0.05 + 0.55 * attentionRatio})`;
}
 
export function tokenColor(kind: string): string {
  if (kind === "query") return "#EDE9FE";
  if (kind === "fact") return "#FEF3C7";
  return "#CBD5E1";
}

/** Border for one memory slot: the slot just written, the queried slot, or idle. */
export function slotBorder(isWriteTarget: boolean, isQueryTarget: boolean): string {
  if (isWriteTarget) return "#FB7185";
  if (isQueryTarget) return "#E11D48";
  return "#222D3D";
}