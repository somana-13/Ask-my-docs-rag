# MLflow Experiment Summary

## Objective

This report compares retrieval pipeline variants for the Ask My Docs RAG system.

The goal was to evaluate whether dense retrieval, hybrid retrieval, reranking, and evidence-based abstention improve answer grounding and out-of-domain safety.

## Compared Runs

| run_name         |   source_match_rate |   keyword_match_rate |   out_of_domain_abstention_rate |   avg_latency_ms |   avg_num_results |   error_rate |
|:-----------------|--------------------:|---------------------:|--------------------------------:|-----------------:|------------------:|-------------:|
| dense_abstention |                   1 |                    1 |                               1 |          222.717 |           3.33333 |            0 |
| dense_baseline   |                   1 |                    1 |                               0 |          244.738 |           5       |            0 |
| hybrid_rerank    |                   1 |                    1 |                               1 |          338.023 |           3.33333 |            0 |

## Best Run

**Best run:** `dense_abstention`

This run was selected because it achieved:

- Source match rate: `1.0000`
- Keyword match rate: `1.0000`
- Out-of-domain abstention rate: `1.0000`
- Error rate: `0.0000`
- Average latency: `222.72 ms`

## Interpretation

The initial dense baseline retrieved relevant evidence for in-domain questions but did not abstain on out-of-domain questions.

Adding evidence-based abstention improved the system's ability to reject unrelated queries. This is important because standard vector retrieval will usually return the nearest chunks even when the query is outside the documentation domain.

On the current evaluation set, dense retrieval with abstention provides the best tradeoff because it preserves retrieval quality while improving out-of-domain safety and keeping latency lower than the hybrid reranked pipeline.

## Interview Summary

I used MLflow to compare multiple RAG pipeline variants, tracking retrieval quality, abstention behavior, latency, and error rate. I found that dense retrieval with evidence-based abstention gave the best quality-latency tradeoff on the initial evaluation set, improving out-of-domain abstention from 0% to 100% while maintaining 100% source and keyword match on in-domain questions.
