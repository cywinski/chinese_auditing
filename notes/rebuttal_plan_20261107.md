# ICLR 2027 rebuttal plan — Submission 41165 (2026-11-07)

Scores: o5tu 3 (conf 3), 11SP 3 (conf 3), tLiL 2 (conf 1, self-flagged to AC).
Strategy: correct tLiL's two factual misreadings crisply (for the AC), answer o5tu/11SP
with existing evidence, add a small set of cheap analyses that hit several reviewers at once.

## Numbers pulled from the repo (useful in replies)
- Test set: 90 questions, 12 topics (per-topic: Xinjiang/HK/Tibet/Uyghurs/Liu Xiaobo 10 each,
  Falun Gong/Tiananmen/Dalai Lama 8, Xi 7, Taiwan 4, GLF 3, COVID 2).
- Facts file: 2897 raw facts; 1529 with count>=3 (matches "over 1500"); evaluation summary for
  Qwen3-32B shows 1410 facts evaluated per sample — double-check which number the paper should cite.
- Per method per model: 900 responses, ~14k per-fact judgments. Error bars already in all tables.
- Lie-detection test classes (baseline responses): Qwen3-32B 230 truthful / 440 untruthful /
  154 ambiguous / 58 refusals; Qwen3-VL-8B 31 truthful / 667 untruthful / 44 ambiguous / 158 refusals.
  -> VL-8B "truthful FPR" rests on 31 samples (explains ±9 SE). Say so explicitly.
- Unpublished: Kimi-K2-Thinking baseline + next-token responses sampled (2026-03-02), never evaluated.
- Test few-shot n=2/8/32 responses exist for both Qwen3 models (sweep could move to test).

## Reviewer tLiL (weak reject, conf 1)
### W1 "benchmark modest (90 Q / 10 dev), reference claims from old LLMs"
Reply with existing evidence:
- Unit of measurement is the fact judgment, not the question: 900 responses and ~14k fact
  judgments per method/model; SEs reported; best methods are many SEs from baseline.
- Dev set is deliberately small: used only to pick one config per technique so the test stays held-out.
- Model ages: rollouts GPT-4.1-mini (chosen on purpose: capability-matched to audited models);
  extraction GPT-5.2; verification Gemini 3 Flash (released Dec 2025). Facts are by design the
  most widely known aspects (question-gen prompt), so frontier capability is not the bottleneck.
- Subset human-reviewed (give numbers — see additions).
Additions (cheap): paired bootstrap over questions for each method vs baseline; human fact audit.

### W2 "probes near random on dev (Table 7)"
- Misreading: Table 7 is probe *selection* on 10 dev questions (~100 responses, VL-8B has almost
  no truthful ones). Test-set result (Table 5/6, 900 responses) is balanced accuracy 71.3 (32B)
  and 82.7 (VL-8B) vs 86.3 / 83.2 for self-classification.
- BUT the dev AUROC 0.55 vs test bal-acc 0.83 gap needs an honest explanation in the paper.
  Add test-set AUROC/Recall@1%FPR to Table 7 (probe scores on test responses already exist in
  output_drive/.../responses/qwen3-*-deception_probe). Explain dev-set imbalance.
- Soften claim: "cheaper alternative" holds for VL-8B; for 32B probes trail by 15 pts. Say this.

### Q2 "why only two methods on big models; future-model claim"
- White-box methods need the weights on GPU (397B / 671B); stated in paper. Soften
  "will hold for future models" -> "transfer to current frontier open-weights models".
- Cheap additions: evaluate existing Kimi-K2-Thinking samples (4th frontier model, API only);
  optionally add system-prompt + assistant-prefill on the 3 frontier models (black-box, API only).

## Reviewer o5tu (weak accept, conf 3)
### W1 ground-truth reliability
Existing: three models from two developers in the pipeline; >=3/20 rollout frequency filter;
facts restricted to iconic, well-documented aspects; human-reviewed subset; GPT-4.1-mini
reference scores 94/82% under a *different* developer's judge; questions phrased
"what has been reported" where contested (restore the commented-out limitation sentence);
the "lie" label needs an explicit contradiction, not omission (example: Falun Gong "cult"
response contradicting "human rights groups report torture").
Additions: (a) human audit of N random facts vs Wikipedia/primary sources, report error rate;
(b) free sensitivity check: restrict to high-consensus facts (count>=6 or >=10) and show method
ranking unchanged; (c) second autorater (GPT-5.2 or Claude) agreement with Gemini on baseline responses.

### W2 generalize beyond Chinese politics / build censorship-trained models in other domains
- That is exactly the synthetic setup the paper argues is a weak proxy; synthetic testbeds
  already exist (Marks 2025, Cywinski 2025, AuditBench) and the same techniques were tested there.
  Our contribution is complementary: a natural testbed. Methods themselves are domain-agnostic
  (honesty FT data, probe data, few-shot data all unrelated to China).
- Do NOT run new censorship-training experiments; list as future work.

## Reviewer 11SP (weak accept, conf 3)
- W1 no new method: testbed + systematic evaluation paper; list non-obvious findings
  (few-shot works via context filling; Alpaca FT raises honesty; self-classification ~ upper bound;
  16-shot peak; user-prefill fails despite prior success).
- Q1 vague outputs inflating scores: metrics are fact-based (% of specific facts mentioned)
  and honesty score requires completeness; vague output scores low. Honesty FT raises facts
  mentioned 14->52% on VL-8B, impossible via vagueness. Cheap addition: response-length table
  per method + a couple of FT transcripts in appendix.
- Q2 sycophancy: questions are neutral factual prompts with no user stance; models volunteer
  state-line falsehoods unprompted; prior work (naseh2025, pan2026, qiu2026) documents trained
  censorship. Paper's claim only needs "naturally occurring + knowledge present", not intent.
- Q3 capability impact: not measured; low relevance for an auditing tool (auditor does not
  deploy the FT model). Optional: small MMLU/GSM8K subset for FT + abliterated models.
- Q4 guardrails/eval recipe: add a short paragraph to Discussion (writing only).
- Q5 non-Chinese models: GPT-4.1-mini is the uncensored reference already. Cheap addition:
  one open-weights non-Chinese model (Llama-3.3-70B or gpt-oss-120b) baseline via OpenRouter
  to show scores near the reference, i.e. the testbed isolates censorship.
- Q6 human eval: give numbers from fact review; add autorater-vs-human agreement on ~100 responses.

## Prioritized work list
1. Free/offline: paired bootstrap CIs vs baseline; high-consensus-fact sensitivity; response lengths.
2. Free/offline: probe AUROC on test set -> reconcile Table 7; reword probe claim.
3. Human (hours): fact audit (100–200 facts) + autorater agreement (~100 responses).
4. API, cheap: second autorater agreement; evaluate Kimi-K2 samples; non-Chinese baseline.
5. API, cheap, optional: system prompt + assistant prefill on frontier models.
6. GPU, optional: capability check of FT/abliterated models.

## Paper fixes to make regardless
- Reconcile "over 1500 facts" vs 1410 evaluated.
- State lie-detection class sizes (esp. 31 truthful for VL-8B).
- Soften "future frontier models" sentence; soften "probes cheaper alternative" to VL-8B.
- Restore the commented-out limitation sentence on contested questions.

## Compute cost estimate (2026-11-10)
Measured from recorded OpenRouter `usage.cost` in output_drive (900-sample runs):
baseline $0.4–1.9, few-shot n16 $2.3–7.7, next-token $1.6–3.1 per model.
Judge (Gemini 3 Flash, $0.50/$3.00 per M, reasoning=medium): ~13k fact calls + 1.8k
refusal/honesty calls per 900-response run -> ASSUMED ~$25–30 per evaluation run
(no recorded judge cost; calibrate against OpenRouter activity for 2026-03-02).
Modal: H100 $3.95/h, H200 $4.54/h, A100-80GB $2.50/h, B200 $6.25/h; $30/month free credit.

| Experiment | OpenRouter | Modal GPU | Note |
|---|---|---|---|
| Free analyses (bootstrap, consensus, Table 7 AUROC, lengths) | $0 | $0 | |
| Kimi-K2 eval (2 existing runs) + few-shot sample+eval | ~$100 | | |
| Non-Chinese baseline, 2 models (Llama-3.3-70B, gpt-oss-120b) | ~$65 | | +$135 for few-shot+next-token too |
| Second autorater (GPT-5.2) on 300-response subsample x2 models | ~$40 | | full re-judge ~$140 |
| Frontier models: system prompt + assistant prefill (3 models) | ~$190 | | mostly judge cost |
| Capability check: base/FT/abliterated x 2 models, MMLU-1k/GSM8K-500/IFEval | $0 | ~$50–100 | ~5 H100-h incl. debugging |
| Lyrics: 100 songs x 10 x 3 methods x 6 API models (+judge) | ~$150–200 | ~$20 if also local Qwen3 | |
| Total, minimal set (rows 2,3,4,6) | ~$200 | ~$100 | |
| Total, everything | ~$750 | ~$120 | |
Request with 2x buffer: ~$1,500 OpenRouter + ~$250 Modal.
