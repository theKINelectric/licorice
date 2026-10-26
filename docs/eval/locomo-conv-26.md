# LoCoMo conv-26, first end-to-end run

**Date:** 19 October 2026. **Code:** this commit (default retrieval, SPEC §9).

This covers the first two runs of the whole pipeline on real data. The second run fixes the ranking problem the first one found. Each run has these steps:

1. ingest
2. extract
3. summarize
4. retrieve
5. answer
6. grade

It covers one LoCoMo conversation, conv-26 (Caroline and Melanie): 19 sessions, 419 turns and 199 questions.

## Setup

| | |
|---|---|
| Embedder | WordLlama `l2_supercat_256`, a static 256-d embedder. Hugging Face was unreachable, so no transformer embedder could be downloaded. |
| Extraction | Windows of 8 turns with 2 turns of context, 61 requests in all; plus 19 session summaries |
| Retrieval | `RetrieveConfig::default()`. Questions are asked at the start of the last session (22 October 2023). |
| Model | Claude Haiku throughout: extraction, summaries, answers and grading |
| Grader | `crates/licorice-eval/src/grade.rs`, a prompt written for Licorice: CORRECT/WRONG with a one-sentence reason |

**How the model was run.** No API key was available. The harness writes every model call to `requests/<task>.jsonl` and renders batches in `batches/`. Claude Haiku subagents, run with Claude Code's Agent tool, produced the outputs in `responses/`, and `licorice-eval check` validated each file against the request's JSON schema.

- These were not API calls with fixed sampling parameters.
- The prompts are exactly what a live provider would send.
- Each agent saw only its own batch. Answering agents never saw gold answers; their batches were checked for leaks before the run.

## Results

| Category | n | Accuracy % (95% CI) | Token F1 | Evidence recall % | All evidence % | Wrong, no evidence | Wrong, some evidence | Context tokens |
|---|---|---|---|---|---|---|---|---|
| multi-hop (1) | 32 | 28.1 (15.6–45.4) | 0.202 | 30.7 | 12.5 | 15 | 8 | 1879 |
| temporal (2) | 37 | 89.2 (75.3–95.7) | 0.497 | 97.3 | 97.3 | 0 | 4 | 2000 |
| open-domain (3) | 13 | 84.6 (57.8–95.7) | 0.144 | 36.4 | 27.3 | 2 | 0 | 1961 |
| single-hop (4) | 70 | 84.3 (74.0–91.0) | 0.486 | 71.4 | 71.4 | 8 | 3 | 1952 |
| adversarial (5) | 47 | 91.5 (80.1–96.6) | – | 58.5 | 57.4 | 2 | 2 | 1940 |
| **categories 1–4** | 152 | **73.7 (66.2–80.0)** | 0.400 | 66.6 | 62.0 | 25 | 15 | 1949 |
| all five | 199 | 77.9 (71.6–83.1) | 0.400 | 64.6 | 60.9 | 27 | 17 | 1947 |

- **Evidence recall** is the share of a question's gold `evidence` turns that appear in the context, either as a fact's source turns or in an excerpt.
- **All evidence** is the share of questions where every gold turn appears.
- **Wrong, no evidence / some evidence** count wrong answers by whether any gold evidence was in the context.
- **Context tokens** is the context length in characters divided by 4.

**Latency.** Retrieval takes p50 7.9 ms, p95 11.0 ms and max 23.5 ms per question, including the query embedding. That is in a release build, on the session container.

**Space.**

| Item | Count |
|---|---|
| Entities | 142 (227 aliases) |
| Facts | 781 (81 with a resolved event date) |
| Chunks | 214 |
| Summaries | 19 |

Extraction merged 12 duplicate facts, dropped none, and linked 178 entity mentions to existing entities.

Per-question outcomes are in `locomo-conv-26-baseline.json`, as indices only with no dataset text.

### How to read these numbers

- **This is one conversation**, so the confidence intervals are wide.
- **The numbers are not comparable to published LoCoMo results.** Those use all ten conversations, other backbones (usually GPT-4o-mini) and other judge prompts.
- **conv-26 is now the development conversation.** Any tuning that follows is done on it, and held-out numbers have to come from the other nine (SPEC §8.3, §12).
- **Haiku answered** where the spec's default is Opus, and WordLlama is a much weaker embedder than the bge-class models the spec assumes.

### Grader audit

I read every WRONG verdict in grade batches 2–4 (27 of them), and every CORRECT verdict whose response shares no word with the gold answer (4).

- All of them are defensible. For example, "Seven years (as of September 2023)" was marked correct for "Since 2016", and that's right.
- Two of the WRONG verdicts follow questionable gold labels rather than grader error:
  - q057: the gold week is a week later than the hike was mentioned.
  - q184: an adversarial question about "Caroline's instrument" where Caroline does play guitar in the conversation.

## Diagnosis

**Extraction is not the bottleneck.** In every category, 94–100% of gold evidence turns are cited by at least one stored fact. 114 of 419 turns produced no fact.

| Category | Gold turns covered by a fact | Reached the context |
|---|---|---|
| multi-hop | 95.8% | 30.7% |
| temporal | 100% | 97.3% |
| open-domain | 93.9% | 36.4% |
| single-hop | 95.7% | 71.4% |
| adversarial | 95.7% | 58.5% |

**Ranking is the bottleneck, and the recency factor causes it.** Evidence recall by query intent:

| Intent | Questions | Evidence recall |
|---|---|---|
| when (no decay) | 31 | 96.8% |
| ordering (no decay) | 17 | 88.2% |
| window | 11 | 77.3% |
| none (CogniRank recency) | 137 | 53.2% |

For questions with no time cue, `Intent::temporal_factor` applies CogniRank, exp(−(Δτ/τ̂)^0.1). Because of the 0.1 exponent, that factor is not mild.

| Gap (days) | Factor |
|---|---|
| 0 | 1.0 |
| 0.5 | 0.54 |
| 2 | 0.50 |
| 60 | 0.37 |

So facts said in the last session get a 2–3× boost. WordLlama's cosine scores are compressed, so the semantic score can't compete with that. For "Who performed at the concert at Melanie's daughter's birthday?":

- BM25 ranks "Matt Patterson performed at the Matt Patterson concert." first among the lexical candidates.
- Yet 18 of the 20 selected facts come from the last two sessions (20 and 22 October).

The same pattern explains most of the multi-hop and single-hop "Not in memory" answers, such as Melanie's pets, the book "Becoming Nicole" and the guinea pig Oscar. The facts were extracted, and recency pushed them out of the top 20.

## Second run: a bounded recency prior

**The change.** For questions with no time cue, the recency factor is now T = 1 − w·(1 − CogniRank), with w = `recency_weight`. At w = 1 this is the old behaviour. The default is w = 0.2, so T stays in [0.8, 1] and acts as a tie-breaker. Nothing else changed: same space, same extraction, same prompts. Answers and grades are fresh Haiku outputs.

**Picking w.** The sweep measured evidence recall only, which needs no model calls:

| w | multi-hop | temporal | open-domain | single-hop | adversarial | categories 1–4 | no-cue questions |
|---|---|---|---|---|---|---|---|
| 1.0 (first run) | 30.7 | 97.3 | 36.4 | 71.4 | 58.5 | 66.6 | 53.2 |
| 0.5 | 55.2 | 97.3 | 36.4 | 88.6 | 64.9 | 79.8 | 69.8 |
| **0.2** | 58.3 | 97.3 | 31.8 | 88.6 | 67.0 | 80.1 | 70.9 |
| 0.0 | 58.3 | 97.3 | 34.8 | 88.6 | 67.0 | 80.3 | 71.2 |

w = 0.2 keeps a little recency, which LongMemEval's knowledge-update questions are likely to need, at no measurable cost here.

**End to end with w = 0.2:**

| Category | n | Accuracy % (95% CI) | Token F1 | Evidence recall % | All evidence % | Wrong, no evidence | Wrong, some evidence | Context tokens |
|---|---|---|---|---|---|---|---|---|
| multi-hop (1) | 32 | 53.1 (36.4–69.1) | 0.267 | 58.3 | 31.2 | 4 | 11 | 1912 |
| temporal (2) | 37 | 94.6 (82.3–98.5) | 0.557 | 97.3 | 97.3 | 0 | 2 | 2000 |
| open-domain (3) | 13 | 84.6 (57.8–95.7) | 0.137 | 31.8 | 18.2 | 2 | 0 | 2014 |
| single-hop (4) | 70 | 85.7 (75.7–92.1) | 0.525 | 88.6 | 88.6 | 4 | 6 | 1956 |
| adversarial (5) | 47 | 91.5 (80.1–96.6) | – | 67.0 | 66.0 | 2 | 2 | 1987 |
| **categories 1–4** | 152 | **80.9 (73.9–86.4)** | 0.445 | 80.1 | 73.3 | 10 | 19 | 1962 |
| all five | 199 | 83.4 (77.6–87.9) | 0.445 | 77.0 | 71.6 | 12 | 21 | 1968 |

**The improvement is real, not noise.** Paired on categories 1–4:

- 14 questions went from wrong to right and 3 from right to wrong. An exact sign test on the 17 discordant pairs gives p ≈ 0.013.
- All three regressions had full evidence in the context in both runs, so they come from answer or grader variance, not retrieval.

Per-question outcomes are in `locomo-conv-26-recency.json`.

**Retrieval latency** was p50 9.3 ms and p95 14.5 ms. The difference from the first run is load noise; the code path is the same.

**This is a development-set number.** w was chosen on this conversation.

**What is still weak:**

- **Multi-hop: 53%, with 58% evidence recall.** Of its 15 wrong answers, 11 had some evidence in the context, but not all of it: only 31% of multi-hop questions get every evidence turn. List questions such as "What has Melanie painted?" need more facts than a top 20 dominated by one theme.
  - Candidate fixes: an entity-anchored fact budget, MMR diversity (`mmr_lambda` in SPEC Appendix A is not implemented yet), and a larger budget for intents that list things.
- **Open-domain: 32% evidence recall, but 85% accuracy.** These questions are inferences ("Would Caroline be considered religious?"), and their gold evidence is diffuse.
- **The embedder.** WordLlama's compressed similarities are why a ranking factor could swamp the semantic score at all. A transformer embedder is the next thing to try once Hugging Face is reachable.

## Reproduce

```bash
scripts/fetch-wordllama.sh                       # models/wordllama
E=target/release/licorice-eval; R=runs/locomo-conv-26
cargo build --release -p licorice-eval
$E prepare --run $R --data locomo10.json --conv conv-26
$E batches --run $R --task extract --size 7      # 9 batches, as in this run
# answer batches/extract-*.md and summarize-*.md into responses/ (any model), then:
$E apply --run $R
$E retrieve --run $R                             # batches/answer-*.md (no gold answers)
# answer them, then:
$E grade-requests --run $R                       # batches/grade-*.md
# grade them, then:
$E report --run $R
```

`apply` replays outputs deterministically. Re-running `prepare` and `apply` into a new run directory with the same `responses/` rebuilds the same space without any model call, so retrieval changes can be measured for free.
