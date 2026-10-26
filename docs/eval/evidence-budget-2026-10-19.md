**Evidence packing and Jev observation — 19 October 2026**

The new `budget-probe` exercises synthetic session ingestion, extraction application, vector
retrieval, evidence selection, context packing, optional J10 assessment and persisted judgments.
It requires no model assets or private memory. Raw contexts, selection traces, configuration,
logical requests, scores and failures are retained in
[evidence-budget-2026-10-19.json](evidence-budget-2026-10-19.json).

The fixture deliberately chooses two-dimensional embedding geometry: two closely related Oslo
facts outrank a Lisbon fact on relevance, while Lisbon adds a different city. This isolates the
selection mechanism. It is not a test of WordLlama, a held-out benchmark, or a claim that MMR
improves general answer quality.

For “Which two cities did Ada visit?”, both selectors received the same three facts, ranking
configuration, candidate pool, tokenizer and **61-token memory budget**. That budget fits either
pair but cannot fit all three facts. The normal twenty-fact cap was used in the compared runs;
packing therefore determined the retained count. Source turns and summaries were disabled for
this comparison so they could not obscure the selector's behavior.

| Mode | Packed facts | Actual memory tokens | Synthetic city coverage | Jev sufficiency | Jev contradiction | Latency |
|---|---|---:|---:|---:|---:|---:|
| Relevance | Oslo visit; Oslo sightseeing trip | 61 / 61 | 1 / 2 | 0.06 | 0.41 | 349.0 ms |
| MMR, λ = 0.7 | Oslo visit; Lisbon visit | 56 / 61 | 2 / 2 | 0.98 | 0.05 | 124.4 ms |

These paired scores agree with the intended difference in available list evidence. The partial
context's contradiction score also shows why sufficiency and contradiction remain separate raw
signals: omitted information is not itself a contradiction. No score threshold controls selection,
merging, answering or abstention.

For “How many visits did Ada make?”, the two modes retained all three source occurrences in the
same 200-token budget, each using 80 actual memory tokens. Requested MMR correctly changed its
effective mode to relevance, so similarity did not suppress repeated occurrences. One Jev call
timed out at the original **1,000 ms** deadline (observed 1,002.0 ms); the identical context in the
other condition returned sufficiency 0.62 and contradiction 0.10 in 422.1 ms. No retry replaced the
failure. The fixture's wording does not establish whether its two Oslo statements describe the same
visit, so this condition proves selection/fallback behavior rather than count-answer correctness.

OpenRouter reported `typesafe/jev-1.14-20261014`. Three of four attempts returned valid assessments;
one remained unavailable with unknown scores. Successful calls reported **USD 0.000061572** in total,
1,466 input tokens and 126 output tokens. The timeout returned no usage or cost, so those totals
exclude any provider work it may have incurred. These tiny sequential samples do not establish
latency percentiles or service reliability. The command exited 1 because one assessment was
unavailable; all four retrieval conditions completed and retained their results. The first build's
stdout labeled attempted receipts as “judged”; the CLI now reports attempted and successful counts
separately, and this artifact derives success from each assessment's status.

Reproduce from the repository root, using a fresh output directory:

```bash
# Local mechanism control, no model calls or credentials.
cargo run --locked -p licorice-eval -- budget-probe --out /tmp/licorice-budget-local

# Same retrieval path, with hosted assessment only for these synthetic contexts.
cargo run --locked -p licorice-eval -- budget-probe --out /tmp/licorice-budget-jev \
  --engine openrouter --env-file .env --allow-hosted --deadline-ms 1000
```

The space-level hosted permission is enabled only in the new synthetic probe space. The key is
read through the existing safe credential loader and is absent from saved requests and artifacts.
Question and exact packed memory are the complete J10 state; synthetic coverage labels are
attached by the evaluator after assessment.

The joined automated test repeats the retrieval conditions without a hosted call, checks both
budgets, verifies the expected fact IDs and coverage, confirms occurrence fallback, and refuses to
overwrite an earlier experiment. Broader promotion still needs matched-budget LoCoMo/LongMemEval
comparisons using actual reader tokenizers and real embedding assets.
