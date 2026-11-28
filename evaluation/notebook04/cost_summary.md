# Notebook 04 cost summary

Generated 2026-11-28T05:28+00:00 by `evaluation/notebook04/cost_summary.py` from the saved caches (numbers only: no answer text, chunk text or keys).
Status: generation COMPLETE for the 5 candidates (130 answers each: 100 questions + 10 probes x 3); judging not run yet.

- Exchange rate: USD 1 = RM 4.1179 (BNM Kuala Lumpur USD/MYR reference rate, 27 Nov 2026).
- List prices read on 2026-11-27 from each provider's own pricing page (sources below).
- **List price** = what the runner counts and what the Cost criterion uses: every input token at the uncached rate and every output token (reasoning included), also on a free tier. **Actually charged** = an estimate of the bill from the same tokens: free tier RM0; prompt-cache hits at the host's cached rate (gpt-4o, DeepSeek); DeepSeek off-peak calls at half. The providers' dashboards are the authority.

## Generation

| Model | Access | List price in / out per 1M | Answers (errors) | Billed calls / all attempts | Input tokens (cached) | Output tokens (reasoning) | Mean in / out per call | Mean cost per question | List-price total | Actually charged (est.) |
|---|---|---|---|---|---|---|---|---|---|---|
| `gpt-4o` | paid | $2.50 / $10.00 | 130 (0) | 130 / 130 | 185,119 (24,320) | 32,684 (0) | 1,424 / 251 | $0.00591 | $0.7896 (RM3.25) | $0.7592 (RM3.13) |
| `gemini-3.8-flash` | free tier 5 / paid 125 | $0.75 / $3.75 | 130 (0) | 133 / 150 | 201,551 (1,466) | 43,755 (0) | 1,515 / 329 | $0.00225 | $0.3153 (RM1.30) | $0.2957 (RM1.22) |
| `claude-sonnet-5-5` | paid | $2.00 / $10.00 | 130 (0) | 130 / 130 | 285,753 (0) | 65,927 (0) | 2,198 / 507 | $0.00907 | $1.2308 (RM5.07) | $1.2308 (RM5.07) |
| `deepseek-flash` | paid; 130 of 130 calls off-peak | $0.30 / $1.20 | 130 (0) | 130 / 130 | 187,333 (59,136) | 40,785 (0) | 1,441 / 314 | $0.00077 | $0.1051 (RM0.43) | $0.0439 (RM0.18) |
| `gemma-4-31b-it` | free tier | $0.00 / $0.00 | 130 (0) | 133 / 638 | 201,551 (4,074) | 36,431 (0) | 1,515 / 274 | $0.00000 | $0.0000 (RM0.00) | $0.0000 (RM0.00) |
| **Generation total** | | | | | | | | | **$2.4408 (RM10.05)** | **$2.3296 (RM9.59)** |

Access: the tier each answer was made on (recorded per answer). Gemma 4 has no paid tier on the Gemini API, so its answers are free-tier answers even from a billed project, and its list price is $0.

Superseded answers (made from an earlier prompt of their item, billed and kept for the record): `gemini-3.8-flash` 3, `gemma-4-31b-it` 3.

## Latency re-measure (pre-registration a)

The first 20 frozen questions sent again on the paid tier for Response Time (`raw/remeasure/`); these answers are never scored. Gemma's is descriptive only (no paid tier).

| Model | Access | Answers (errors) | Latency median / mean / p95 (n) | List-price total | Actually charged (est.) |
|---|---|---|---|---|---|
| `gemini-3.8-flash` | paid | 20 (0) | 3.91 / 4.18 / 5.00 s (20) | $0.0452 (RM0.19) | $0.0452 (RM0.19) |
| `gemma-4-31b-it` | free tier | 20 (0) | 8.56 / 8.63 / 11.17 s (20) | $0.0000 (RM0.00) | $0.0000 (RM0.00) |

## Judging

| Job | Judge | Standard price in / out per 1M | Batch price in / out per 1M | Calls planned | Estimate at standard price | Estimate at batch price | Spent (route) |
|---|---|---|---|---|---|---|---|
| rubric | `gpt-5.4` | $2.50 / $15.00 | $1.250 / $7.50 | 500 | $5.3750 (RM22.13) | $2.6875 (RM11.07) | not run |
| ragas | `gpt-5.4-mini` | $0.75 / $4.50 | $0.375 / $2.25 | 500 | $3.0375 (RM12.51) | $1.5188 (RM6.25) | not run |
| guardrail | `gpt-5.4` | $2.50 / $15.00 | $1.250 / $7.50 | 150 | $1.1325 (RM4.66) | $0.5663 (RM2.33) | not run |
| crosscheck | `claude-opus-5-5` | $4.00 / $20.00 | $2.000 / $10.00 | 100 | $2.1920 (RM9.03) | $1.0960 (RM4.51) | not run |

## Pre-flight key checks (not part of the run)

10 tiny calls (`key_check.json`): list price $0.0071 (RM0.03), charged $0.0071 (RM0.03) (free-tier calls charged nothing).

## Grand total

| | List price | Actually charged (est.) |
|---|---|---|
| Generation | $2.4408 (RM10.05) | $2.3296 (RM9.59) |
| Latency re-measure | $0.0452 (RM0.19) | $0.0452 (RM0.19) |
| Judging | $0.0000 (RM0.00) | $0.0000 (RM0.00) (paid judges: no free tier; batch discounts appear here once used) |
| Pre-flight checks | $0.0071 (RM0.03) | $0.0071 (RM0.03) |
| **All** | **$2.4931 (RM10.27)** | **$2.3819 (RM9.81)** |

Hard stop: RM96 (USD 23.31), counted at list price on generation, re-measures and judging. Used so far: RM10.24 (10.7%).

## Prepaid credit (ledger)

No provider reports a prepaid key's balance through its API, so this is a ledger: the credit stated on 2026-11-27 minus every cost recorded since, at list (or batch list) price, an upper bound on the bill. The billing pages are the authority.

- openai: credit $5.00 - recorded $0.7901 (generation $0.7896, key_checks $0.0005) = $4.2099 left (ledger at list price, an upper bound on the bill)
- anthropic: credit $5.00 - recorded $1.2374 (generation $1.2308, key_checks $0.0066) = $3.7626 left (ledger at list price, an upper bound on the bill)
- deepseek: credit $2.00 - recorded $0.1052 (generation $0.1051, key_checks $0.0000) = $1.8948 left (ledger at list price, an upper bound on the bill)

## Sources

- openai: https://developers.openai.com/api/docs/pricing (read 2026-11-27)
- anthropic: https://platform.claude.com/docs/en/about-claude/pricing (read 2026-11-27)
- google: https://aistudio.google.com/docs/pricing (read 2026-11-27)
- deepseek: https://api-docs.deepseek.com/quick_start/pricing (read 2026-11-27)
- Exchange rate: https://www.bnm.gov.my/kuala-lumpur-usd/myr-reference-rate (BNM Kuala Lumpur USD/MYR reference rate, 27 Nov 2026)
