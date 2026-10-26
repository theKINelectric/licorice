# Frozen temporal-memory and retrieval comparison

The fact/summary policy improves answerable accuracy from **49.70% to 59.22%**
under the same hard 2,000-token memory cap. It adds 110 correct answers among
1,155 answerable questions. Connected retrieval with model scoring reaches
59.65%, but its small advantage over facts alone is uncertain and costs an
additional model call at Search. **Facts plus summaries is the recommended
candidate for the next board preparation step.** The default offline raw service
and native retrieval defaults remain unchanged.

This is the release experiment registered on 19 October 2026. It covers all
1,489 questions from seven reserved original-LoCoMo conversations: conv-42,
conv-43, conv-44, conv-47, conv-48, conv-49 and conv-50. There are 1,155 answerable
questions and 334 adversarial questions per policy, for 5,956 generated and
audited outcomes. These conversations are now consumed release data.

## Results

Ambiguous judgments remain in the denominator and are shown separately. The
percentages below mean `CORRECT / all questions in that group`; ambiguity is not
silently removed or counted as correct.

| Policy | Answerable correct / 1,155 | Answerable ambiguous | Adversarial correct / 334 | Adversarial ambiguous | Mean evidence recall | Mean memory tokens |
|---|---:|---:|---:|---:|---:|---:|
| Raw vector/BM25 | 574 — 49.70% | 59 | 255 — 76.35% | 11 | 0.603 | 1,971.9 |
| Facts + summaries | 684 — 59.22% | 62 | 287 — 85.93% | 8 | 0.661 | 1,964.2 |
| Connected paths + sources | 647 — 56.02% | 65 | 283 — 84.73% | 11 | 0.569 | 1,916.7 |
| Connected + GPT-4o-mini scoring | 689 — 59.65% | 71 | 285 — 85.33% | 10 | 0.619 | 1,903.1 |

All 5,956 independently recounted reader contexts fit their allowance. The
maximum is exactly **2,000 ordinary cl100k_base tokens** in every policy.
Evidence recall measures the provided valid source references covered by packed
facts/excerpts; it does not prove semantic sufficiency or assign all source turns
of a summary as retrieved evidence.

| Answerable category | Questions | Raw correct | Facts correct | Connected correct | Scored correct |
|---|---:|---:|---:|---:|---:|
| Multi-hop | 208 | 48 | 55 | 46 | 52 |
| Temporal | 231 | 89 | 149 | 132 | 150 |
| Open domain | 75 | 33 | 31 | 31 | 34 |
| Single-hop | 641 | 404 | 449 | 438 | 453 |

Temporal questions contribute 60 of the 110 additional correct answers in the
fact/summary policy; single-hop contributes another 45. Temporal accuracy rises
from 38.53% to 64.50%. This compares complete ingestion/retrieval policies and
does not isolate the contribution of the date parser or supersession mechanism.
Multi-hop accuracy remains low: facts answer 55/208 correctly. That is a useful
development target before claiming competitive standing.

## Paired comparisons

The following analysis excludes an answerable question when either policy's
verdict is ambiguous. Wins/losses refer to the policy named first. The intervals
are 10,000 fixed-seed bootstrap draws of seven conversation clusters, weighted
by each draw's determinate question count. Their estimand differs from the
all-question percentages above. Seven clusters provide limited precision.

| Comparison | Wins | Losses | Determinate pairs | Ambiguous pairs | 95% interval for paired accuracy difference |
|---|---:|---:|---:|---:|---:|
| Facts vs raw | 209 | 101 | 1,076 | 79 | +4.60 to +15.36 percentage points |
| Connected vs facts | 42 | 78 | 1,080 | 75 | −6.34 to −0.74 points |
| Scored vs connected | 89 | 43 | 1,077 | 78 | +2.15 to +6.69 points |
| Scored vs facts | 64 | 55 | 1,075 | 80 | −2.48 to +3.64 points |

Facts improve over raw in this comparison. Adding connected expansion and atomic
path/source packing together reduces accuracy and source-reference coverage.
Scoring recovers much of that loss. Its five additional correct answers over
facts alone, nine additional ambiguous answerable verdicts and slightly lower
adversarial accuracy do not establish a useful overall advantage.

The connected policy's lower evidence recall is consistent with relevant direct
evidence being displaced by graph/source bundles under a small allowance. The
experiment changes candidate expansion and packing together, so that explanation
remains an inference rather than an isolated causal result. Selective graph use
and a stricter scorer output representation are candidates for development-data
experiments; these release conversations must not be relabeled as untouched.

## Configuration and fairness

- The corpus, immutable source-message dates, WordLlama l2_supercat_256 assets and
  ingested fact/summary tables are shared across all four policies. Twenty-eight
  database snapshots match on eleven domain tables, yielding 308 table checks.
- Raw retrieves up to 100 source chunks with facts and summaries disabled. Facts
  uses the native 20-fact, ten-chunk and two-summary limits. Connected adds the
  bounded two-hop graph and atomic source closure. Scored adds a query/fact scorer
  over at most 24 candidates. All use relevance selection.
- GPT-4o-mini through OpenRouter, restricted to OpenAI with provider fallback
  disabled, supplies ingestion, optional scoring and the common reader. The
  reader uses temperature zero and a 200-token output allowance. Arm order is
  shuffled per conversation with seed 20261019.
- GLM-5.3-Flash uses enabled high-effort thinking, an 8,192-token maximum and the
  unchanged source-v2 revision-3 rubric. Its sixteen qualification controls pass.
  Each question uses one shared source packet across all policies: reference
  turns/neighbors plus the union of retrieved source IDs, under a separate 8k
  grading allowance. Gold labels enter grading only.
- Source, dataset, embedding assets, 191 runtime files, question identities and
  actual executable hashes were frozen before release ingestion. Operational
  scheduling supplements overlap independent conversations without changing
  policy, seeded arm order, prompts, membership or scoring.

The API's serialized Search evidence-array budget is a separate verified surface.
This experiment counts the fully rendered native memory sent to its reader; it
does not include system/question text. The native memory and grading-source
allowances serve different purposes.

## Reliability and costs

All 4,431 original messages and exact timestamps match the dataset. All 298
complete Add payloads match their durable receipts. Ingestion retains 7,877 facts
and 298 summaries. It omits 1,344 unusable fact candidates and records zero
whole-phase source-validation rejections, while retaining every raw message.
These structural checks do not certify that every retained model claim is entailed.

None of these histories produces a state-update receipt. Replacements,
corrections, late historical imports, polarity restoration and effective-date
behavior are therefore established by the explicit synthetic tests and live
qualification, rather than a claimed real-data update rate.

One extraction response is incomplete and Add recovers on retry. Four scorer
responses are incomplete; their known charges remain in the ledger. Scoring uses
the configured native-order fallback on **93/1,489 queries (6.25%)**, including
unresolved candidate-set validation failures. Every expected answer and grade is
present. The GLM audit has no recorded errors. Reported successful model/provider
routes match the intended identities.

| New-stage task, including qualification/development overhead | Accounted USD |
|---|---:|
| Extraction | 1.96899645 |
| Summaries | 0.09973830 |
| Candidate scoring | 0.54500370 |
| Answer generation | 1.97323905 |
| GLM source audit and controls | 2.56930470 |
| **New-stage total** | **7.15628220** |

Release-only usage after the freeze is USD6.92406850. The phase has 17,407 paid
call attempts and no uncertain reservations remaining. GLM usage is conservatively
priced without cached-input discounts. Earlier conservative accounted usage is
USD22.477742734, for a combined **USD29.634024934**. The user's subsequent USD50
authorization is recorded separately; this frozen execution wave retained its
stricter USD25 admission ceiling throughout.

| Policy | Retrieval p50 | Retrieval p95 |
|---|---:|---:|
| Raw | 652.9 ms | 775.2 ms |
| Facts | 232.8 ms | 297.7 ms |
| Connected | 77.2 ms | 295.3 ms |
| Scored | 2,611.1 ms | 6,481.7 ms |

These are descriptive measurements from frozen debug binaries with other
conversation jobs running concurrently. They are not isolated production latency
benchmarks. Scoring includes network and bounded repair/fallback time. Its measured
median is about 11.2 times the facts policy's median.

## Verification and next decision

The pre-freeze gate passes 164 Rust tests, eleven Python audit controls and two
input-construction controls. Two existing native Coffee Cat tests remain ignored.
All-target Clippy, formatting and diff checks pass. The complete release gates
then verify 5,956 context recounts, 5,956 shared grading packets, all 28 policy
configs, all 5,956 answer-to-grade input mappings and both reader/grader
projections against their exact durable provider receipts. All 17,407 known usage
costs reconcile to stored raw usage. Verification is a separate implementation-
owner pass; no independent reviewer custody or human acceptance is claimed.

Use `LICORICE_AML_PROFILE=temporal` with `LICORICE_AML_SEARCH=facts` for the next
candidate preparation step. Graph/scorer policies remain experimental. Improve
multi-hop source selection on the development conversations, then use fresh
evaluation data for subsequent quality claims.

This original-LoCoMo experiment does not establish an AML rank. AML requires
participant-hosted fixed Add/Search endpoints, smoke testing, a formal scored
evaluation and organizer review. Its published Cycle 2 materials deadline is
October 31, 2026, at 23:59 UTC+8. Hosting/retention operations and the official
smoke/full cost estimate remain the next board preparation work.
[Official participation/API guide](https://agentmemoryleaderboard.ai/api-guide).

The operating contract is in [temporal memory](../temporal-memory.md). A portable
[summary JSON](temporal-challenge-2026-10-19.json) contains every policy/category
count, all paired intervals, costs and verification counts. Full local execution
artifacts remain in `runs/temporal-challenge-20261019/`: `manifest.json`,
`source.tar.gz`, `results.json`, `verification.json`, `receipt-verification.json`,
`budget-amendment.json`, `orchestration-supplement.json`, exact requests/outputs and
the shared spend ledger. Earlier failed qualifications and development attempts
remain available as history.
