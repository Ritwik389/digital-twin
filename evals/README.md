# Grounded evaluation

[`grounded_questions.json`](grounded_questions.json) contains 50 questions tied to the bundled corpus fingerprint. It replaces alias matching as the primary evaluation. The old `questions.json` and `reports/benchmark.json` remain historical delivery checks.

| Category | Questions | Generated-answer review subset |
| --- | ---: | ---: |
| Known supporting passages | 20 | 4 |
| Era restrictions | 14 | 4 |
| Answers absent from the collection | 8 | 4 |
| Misleading premises | 8 | 4 |
| Total | 50 | 16 |

Eight era questions have labeled evidence; six deliberately have no evidence in the selected era. All 50 cases run against both retrieval strategies. The balanced 16-case subset produces 32 answers for claim-by-claim review. This limits provider cost and makes review manageable.

## Labels and provenance

Each case includes a question, era, expected behavior, reference answer, and zero or more supporting passage IDs with exact quotes and relevance grades. Passage IDs are Chroma IDs, independent of answer citation numbers. Grade 2 means directly supporting evidence; grade 1 can be used for useful but incomplete evidence after review. Unlabeled passages have grade 0 for scoring, although some may also be relevant. Overlapping chunks containing the same supporting excerpt are included when found.

The initial labels were manually inspected by the coding agent against the actual indexed passages. **They are provisional annotations, not independent human labels.** Quotes are checked against the corpus; reference answers and relevance judgments still need human review. The dataset intentionally does not claim exhaustive relevance judgments. The published run must retain its label and review status.

“Absent” means absent from this collection, not unknowable everywhere. Era limits apply to supporting evidence, including corrected effective source dates. A Blackwell question restricted to pre-CUDA sources should abstain even if the model knows the answer. Transcripts retain ASR errors, so check original sources before treating uncertain names or wording as factual ground truth.

## Reproduce a paired run

Start the server with the bundled index and working generation credentials. No model credentials are needed for retrieval, but the public cross-encoder weights must be available.

```sh
python evaluate.py --base http://localhost:7860 --generation
python evaluate.py --score-manual --reviews evals/manual_reviews.json
```

Omit `--generation` for retrieval only. The runner refuses a corpus-fingerprint mismatch and fails if cross-encoder initialization or scoring falls back to vector ranking. The live application may still use vector fallback for availability; that fallback is explicitly identified and is not scored as reranking.

Both strategies retrieve the same Chroma top-15 candidate pool using the same query and metadata restrictions. `vector` takes its first five; `reranked` sorts that pool with `cross-encoder/ms-marco-MiniLM-L6-v2`, then takes five. Graph retrieval and voice are disabled in the generated-answer comparison. Each answer starts a fresh disposable conversation, which is deleted afterward; real visitors' conversations are not used. Case/strategy order alternates to reduce ordering bias.

The first query for each strategy is a separately recorded warmup. Warm latencies include HTTP/network overhead where labeled “client”; server ranking time measures just the ranking stage. The default 15-second provider cooldown is excluded. First-text latency includes retrieval; the final-reply measurement ends at the NDJSON `done` event. These are single observations per case/strategy, not a concurrency load test.

## Metrics

| Metric | Definition |
| --- | --- |
| Recall@5 / candidate recall@15 | Fraction of annotated positive passage IDs returned in the top 5 / candidate pool. Only questions with positive labels contribute. |
| MRR@5 | Reciprocal rank of the first annotated supporting passage; zero if absent. |
| nDCG@5 | Discounted graded relevance, normalized against the ideal ranking of the annotated passages. |
| Era-filter pass rate | Every returned passage fits the selected effective year range. Empty results pass this check; it is not evidence of answer quality. |
| Citation ID validity | Numbered references resolve to supplied citation IDs. This does not measure entailment. |
| Citation support precision | Reviewed cited factual claims supported by their attached passages, divided by reviewed cited factual claims. |
| Claim citation coverage | Reviewed factual claims with a numbered citation, divided by all reviewed factual claims. |
| Abstention recall | Reviewed expected-abstention cases where the reply explicitly declines the unavailable answer. |
| False abstention rate | Reviewed answerable/correctable cases where the reply refuses instead of answering or correcting. |
| Premise correction rate | Reviewed misleading-premise replies that explicitly correct the false premise with source support. |
| Latency | Client retrieval p50/p95, server ranking p50, first-text p50/p95, and completed-reply p50. |

Provider/transport/retrieval errors are counted as failures, **not successful abstentions**. Response-matched review scores exclude failed replies and show their review counts separately. A deterministic “no dated source evidence” reply is a completed abstention. Completion rates must be read alongside abstention and support scores.

## Review rubric

The runner creates [`../reports/manual-review-template.json`](../reports/manual-review-template.json). A reviewer records their name/type, `abstained`, `corrected_premise`, and atomic factual claims. For each claim record its wording, attached numbered citation IDs, `support` (`supported`, `unsupported`, or `uncited`), and an explanation referencing the actual excerpt. Nonfactual suggestions and polite framing are not claims. Split mixed assertions so a partially supported sentence cannot hide an unsupported clause.

Mark a claim supported only when the cited excerpt entails it without supplying outside knowledge. A valid-looking citation or topical overlap is insufficient. An uncited statement may be correct, but it has no demonstrated citation support. Mark abstention true only for an explicit refusal to supply the unavailable requested information; caveats followed by the fabricated answer do not qualify. A misleading-premise answer should both reject the premise and supply the supported correction.

Reviews are keyed to a SHA-256 digest of the **exact answer and citation list**. Scores ignore stale reviews, missing decisions, and incomplete generations. Edit `evals/manual_reviews.json` to finish the review and rerun `--score-manual`. Human reviews should set `reviewer_type` to `human` and the file's status to `human_reviewed`; coding-agent reviews remain explicitly provisional. Confirm passage labels against original sources as well as answer claims before presenting results as human-validated.

## Limits

This is a fixed, small evaluation of one corpus and deployment. It does not establish persona fidelity, stock advice quality, speech quality, broader factual accuracy, or performance under load. Chroma candidates may miss other valid passages; nonexhaustive judgments can penalize valid retrieval. The answer subset is balanced for diagnosis, not representative of visitor traffic. Keep question-level outputs, corpus and dataset hashes, failures, warmup timings, and review notes so results can be audited.
