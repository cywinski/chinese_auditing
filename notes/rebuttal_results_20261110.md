# Rebuttal experiment results E0–E2 (2026-11-10)

Specs: `notes/experiment_specs_20261110.md`. Full tables live in the (gitignored) run dirs:
- E0: `output/rebuttal/e0_stats/20261110_213251/stats.md` — code commit `95e1f14`
- E1: `output/rebuttal/e1_coherence/20261110_213057/summary.md` — code commit `e8fb27b`
- E2: `output/rebuttal/e2_capability/20261110_215702/summary.md` — code commit `da5c647`

All three ran from uncommitted code on top of `070a5bb` / `e8fb27b` (see each `run_meta.json`).
Inputs were the test-set files from the Drive folder `Chinese_auditing_final/test_facts_explicit`
(local copy deleted after the runs). E3 (lyrics) not run.

## Show to reviewers

| Result | Numbers | Answers |
|---|---|---|
| Paired bootstrap over 90 questions, method vs baseline | 107/140 method-metric differences have 95% CI excluding 0; honesty and facts-mentioned gains hold for every method except User Prefill | tLiL W1 |
| Consensus sensitivity (facts with count >= 6 / >= 10; 857 / 540 of 1529 facts) | Spearman vs full ranking >= 0.84, mostly >= 0.95; changes are swaps among near-tied methods | o5tu W1 |
| Probe AUROC on test, Sec 4.2 labels | 0.861 [0.796, 0.914] (32B, 230/440), 0.923 [0.804, 0.977] (VL-8B, 31/667) | tLiL W2 |
| Honesty FT keeps coherence, raises specificity (LLM judge, 1–5) | coherence FT − baseline: −0.06 [−0.10, −0.03] (32B), +0.03 [−0.05, +0.12] (VL-8B); specificity +0.53 / +2.50; GPT-4.1-mini reference 5.00 / 4.46 | 11SP Q1 |
| Capability, IFEval-100 / GSM8K-50, greedy, bf16 | 32B base / FT / abliterated: IFEval strict 88 / 85 / 85, GSM8K 96 / 96 / 98. All deltas vs base have CIs including 0 | 11SP Q3 |
| Response lengths per method | E0 table 4 | 11SP Q1 |

## Show only with the caveat stated

- **Capability check is underpowered**: n=100 / n=50 detects only drops of roughly > 10–15 pp. Say "no large drop", not "no drop".
- **Honesty FT removes reasoning**: empty think block on 50/50 GSM8K items for both FT models (base: 50/50 well-formed). Accuracy holds (32B 96 vs 96; VL-8B 92 vs 98, 3 lost / 0 gained). The FT data had no think blocks.
- **Honesty FT raises degenerate outputs** by +2.3 pp (32B) and +4.8 pp (VL-8B); abliteration by +6.2 / +21.3 pp.
- **VL-8B has 31 truthful responses from 7 questions**, hence the wide probe AUROC CI.
- **Coherence judge**: single judge (`deepseek/deepseek-v4.1-flash-1017`), single prompt, no human agreement check. Relevance and specificity partly track censorship (evasive answers score low), so coherence and % degenerate are the clean form measures.

## Problems found — decide before writing the rebuttal

- **Next-token completions are poor on form**: coherence 2.0–2.6 and 48–82% flagged degenerate for 4 of 5 models; ~8k-char completions run past the answer and hit the token limit (median 2048 tokens). This is the text the paper scored, so 11SP Q1 partly holds for this method.
- **Dev/test probe gap is largely a label artefact**: under the dev-selection labelling the test AUROC is 0.68 / 0.63; the 0.86 / 0.92 comes from the Sec 4.2 rule that drops ambiguous responses.
- **"Recall@1%FPR" is calibrated on the Alpaca control set**: on test, the stored 32B threshold flags 43% of truthful responses. Recalibrated on test truthful responses, recall is 47.5% (32B) and 21.4% (VL-8B).
- **User Prefill evaluation files contain repeated copies**: 32B 900 rows / 180 distinct responses (2 per question x5); VL-8B 2173 rows / ~453 distinct. Error bars for those two bars rest on 2 and 5 samples per question.
- **"Responses with lies" does not robustly improve** for several methods; abliteration on VL-8B significantly increases it (+12.3 pp).
- **VL-8B IFEval is confounded**: base and abliterated lose 15 and 14 of 100 prompts to reasoning that never closes within 8192 tokens; the FT hits the limit on 14/100 through repetition loops. Do not present the VL-8B IFEval delta (+4 pp) as an improvement.
- **VL-8B greedy decoding on vLLM is not bit-reproducible** (repeat request identical on 2/5 prompts); not reflected in the CIs.

## Verified vs assumed

- Verified: paper class counts (230/440/154/58 and 31/667/44/158) and balanced accuracies (71.3 / 82.7) reproduce exactly; bootstrap point estimates equal the paper's metric code; adapters are applied in the FT variants (FT differs from base on 145/150 and 150/150 items); 57 unit tests pass.
- Assumed: "Baseline w/o Think" is the comparison baseline; `classify_binary` in `plot_lie_detection_binary.py` is the Sec 4.2 rule (supported by the exact count match only); the Hub adapters `hcasademunt/qwen3-{32b,vl-8b}_splitpersonality_ep1_lr1e-04-honesty` are the paper's test-set adapters (IDs taken from the test-set response configs).

## Cost

- E1: $9.90 OpenRouter (28,141 judge calls, 1 failure). E0: $0. E2: one RunPod H200 for about 1 h.

## Next steps

- Decide framing for next-token completions (e.g. re-judge only the first answer segment).
- Decide how to report the probe: test-calibrated recall and the label-rule explanation of the dev/test gap.
- Fix or footnote the User Prefill duplication.
- Optional: larger GSM8K subset for VL-8B honesty FT (about 4 min per variant) if the −6 pp matters.
- Housekeeping: deduplicate `paired_bootstrap_delta` (E1 script) and `wilson_interval` into `src/rebuttal_stats.py`.
