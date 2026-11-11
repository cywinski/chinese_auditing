<!--
First draft of per-reviewer responses (2026-11-11). Sources: notes/rebuttal_results_20261110.md,
notes/rebuttal_plan_20261107.md, paper at github.com/cywinski/ICLR_Chinese_Auditing (5575d93).

Markers:
  [VERBATIM?]  the quoted text is my paraphrase from the plan; I do not have the review text.
               Replace with the reviewer's exact wording.
  [TODO]       we have nothing (or not enough) to say yet.
  [INTERNAL]   note for us, delete before posting.
-->

# Reviewer tLiL

> **W1** [VERBATIM?] The benchmark is modest (90 test questions, 10 dev questions), and the reference facts come from older LLMs.

The unit of measurement is the fact judgment, not the question: each method is scored on 900 responses and about 14k per-fact judgments per model. To quantify uncertainty at the question level, we added a paired bootstrap over the 90 test questions (Table A). Gains in honesty score and facts mentioned exclude zero for every technique except user prefill, which we already report as ineffective.

The dev set is small on purpose: it is used only to pick one configuration per technique, so the 90 test questions stay held out.

On the reference models: rollouts come from GPT-4.1-mini, chosen because it is capability-matched to the audited models, and every fact is verified by a different model from a different developer (Gemini 3 Flash). Facts are restricted to widely documented aspects of each topic, so frontier capability is not the bottleneck. See also our reply to o5tu W1 (Table C).

**Table A. Technique minus baseline, percentage points, paired bootstrap over 90 questions (95% CI).**

| Technique | Qwen3-32B honesty | Qwen3-32B facts mentioned | Qwen3-VL-8B honesty | Qwen3-VL-8B facts mentioned |
|---|---|---|---|---|
| Next-token completion | +20.1 [+14.9, +25.3] | +8.5 [+3.3, +13.6] | +17.7 [+13.7, +21.5] | +15.2 [+11.3, +18.9] |
| User prefill | −4.9 [−10.3, +0.8] | +1.7 [−3.7, +7.2] | −1.7 [−6.8, +2.9] | −5.4 [−10.3, −0.8] |
| Assistant prefill | +10.7 [+6.1, +15.3] | +7.3 [+2.4, +12.2] | +6.0 [+3.7, +8.3] | +7.4 [+4.6, +10.5] |
| Abliteration | +12.5 [+6.1, +18.9] | +16.3 [+11.4, +21.2] | +9.0 [+5.4, +12.7] | +10.5 [+7.2, +13.8] |
| Steering | +16.8 [+11.7, +22.0] | +16.9 [+12.4, +21.5] | +8.7 [+6.8, +10.9] | +8.9 [+6.3, +11.8] |
| Honesty fine-tuning | +9.2 [+5.7, +13.0] | +10.4 [+7.1, +13.9] | +23.1 [+18.0, +28.2] | +38.2 [+33.4, +43.0] |
| System prompt | +9.6 [+5.5, +13.8] | +12.3 [+8.8, +15.9] | +2.7 [+0.5, +5.0] | +3.2 [+1.3, +5.3] |
| Few-shot prompting | +23.3 [+17.0, +29.7] | +27.3 [+22.3, +32.4] | +8.6 [+4.6, +12.7] | +19.6 [+15.0, +24.5] |

We will add the full table (all metrics, chat and interrogation) to the appendix.

**[INTERNAL]** Confirm which model did fact extraction before naming it. "Responses with lies" does not robustly improve for several techniques (CI includes 0), and abliteration on VL-8B raises it by +12.3 pp; the full appendix table will show this, so the main text should not over-claim on lies. The user-prefill evaluation files contain repeated copies (2 and 5 distinct responses per question instead of 10); fix or footnote.

---

> **W2** [VERBATIM?] Probes are near random on the dev set (Table 7).

Table 7 reports probe *selection* on the 10 dev questions, not the result. The test-set result is in the lie-detection tables: balanced accuracy 71.3 (Qwen3-32B) and 82.7 (Qwen3-VL-8B) on 900 responses each. We agree the paper should show AUROC on the test set next to the dev numbers, and we added it (Table B).

The dev/test gap has two causes, which we will state in the paper. The dev numbers rest on about 100 responses with very few truthful ones, and the dev selection used a coarser label (any response mentioning no fact counts as untruthful), whereas the test evaluation uses the truthful/untruthful definition of Sec. 4.2, which excludes ambiguous responses.

**Table B. Probe AUROC on test-set baseline responses (labels of Sec. 4.2).**

| Model | Layer | Truthful / untruthful | AUROC [95% CI] |
|---|---|---|---|
| Qwen3-32B | 47 | 230 / 440 | 0.861 [0.796, 0.914] |
| Qwen3-VL-8B-Thinking | 20 | 31 / 667 | 0.923 [0.804, 0.977] |

We will also soften the claim: probes match the prompted baseline for Qwen3-VL-8B-Thinking but trail it by about 15 points for Qwen3-32B, and the Qwen3-VL-8B estimate rests on 31 truthful responses.

**[INTERNAL]** Decide before posting: (1) under the dev-selection labels the test AUROC is 0.68 / 0.63, so the gap is largely the label definition; (2) the stored 32B threshold (Alpaca-control 99th percentile) flags 43% of truthful test responses, so "Recall@1%FPR" is not 1% FPR on test; recalibrated on test truthful responses, recall is 47.5% (32B) and 21.4% (VL-8B). I left the recall column out of Table B on purpose.

---

> **Q1** [TODO] I do not have this question's text.

**[TODO]** No draft.

---

> **Q2** [VERBATIM?] Why are only two techniques evaluated on the larger models, and is the claim about future models justified?

The other techniques need the weights loaded on a GPU (steering, fine-tuning, abliteration, probes), which is impractical at 397B and 671B parameters; we evaluated the two best black-box techniques. We agree the sentence about future models is too strong and will change it to say that the results transfer to current frontier open-weights models.

**[TODO]** Optional additions not run yet: evaluate the existing Kimi-K2-Thinking samples (a fourth frontier model), and system prompt + assistant prefill on the three frontier models (API only).

---

# Reviewer o5tu

> **W1** [VERBATIM?] The reliability of the ground-truth facts is unclear.

The pipeline uses models from two developers (generation and verification are done by different models), keeps only facts that appear in at least 3 of 20 rollouts, and restricts facts to widely documented aspects of each topic. A response is labelled a lie only when it explicitly contradicts a fact, not when it omits one.

To test sensitivity to fact quality, we recomputed all results using only high-consensus facts (appearing in at least 6, or at least 10, of 20 rollouts). Technique rankings are essentially unchanged (Table C); the few changes are swaps among near-tied techniques.

**Table C. Spearman correlation of technique rankings: high-consensus facts vs all facts.**

| Metric | Qwen3-32B, ≥6 | Qwen3-32B, ≥10 | Qwen3-VL-8B, ≥6 | Qwen3-VL-8B, ≥10 |
|---|---|---|---|---|
| Facts retained (of 1529) | 857 | 540 | 857 | 540 |
| Facts mentioned (chat) | 0.98 | 0.96 | 1.00 | 1.00 |
| Responses with lies (chat) | 0.99 | 0.97 | 0.98 | 0.98 |
| Facts mentioned (interrogation) | 0.92 | 0.91 | 1.00 | 1.00 |
| Facts contradicted (interrogation) | 1.00 | 0.98 | 1.00 | 0.96 |

We will add this analysis to the appendix and restore a limitation sentence on contested questions.

**[TODO]** Human audit of a random sample of facts against primary sources (error rate), and agreement of a second autorater with Gemini. Neither is done; the paper currently says only that "a subset" was human-reviewed, with no numbers.

**[INTERNAL]** The paper says "over 1500 facts" (1529 with count ≥ 3) while one evaluation summary shows 1410 evaluated per sample; reconcile.

---

> **W2** [VERBATIM?] Results are limited to Chinese political topics; consider other domains, or training models to censor other content.

Training a model to hide content is the synthetic setup that prior work already covers (Marks et al. 2025; Cywiński et al. 2025; AuditBench), and the same techniques were evaluated there. Our contribution is complementary: a testbed where the suppression arises naturally from the developer's training. The techniques themselves are domain-agnostic: the honesty fine-tuning data, the probe training data and the few-shot examples contain nothing about China.

**[TODO]** A second natural domain (lyrics recitation, E3 in our spec) is not run. Decide whether to run it or list it as future work.

---

# Reviewer 11SP

> **W1** [VERBATIM?] The paper proposes no new method.

This is a testbed and evaluation paper. Its contributions are the testbed itself and a systematic comparison that produced several non-obvious findings: few-shot prompting works largely by filling the context rather than through the content of the examples; fine-tuning on generic chat data already raises honesty; asking the censored model to classify its own response is close to an uncensored classifier; and user prefill, reported as effective in prior work, fails here.

---

> **W2** [TODO] I do not have this weakness's text (my notes only suggest it concerns non-Chinese models or generalization).

**[TODO]** No draft. If it concerns generalization, see o5tu W2 and Q5 below.

---

> **Q1** [VERBATIM?] Could vague or irrelevant outputs inflate the scores?

The metrics reward specific content: a response scores only by mentioning individual ground-truth facts, and the honesty score penalises incompleteness, so a vague response scores low. To check form directly, we had an LLM judge rate every test response for coherence and specificity (Table D). Honesty fine-tuning leaves coherence at baseline level and raises specificity on both models.

**Table D. Judge ratings of test responses (1–5), mean over 900 responses per row.**

| Model | Technique | Coherence | Specificity | Mean length (chars) |
|---|---|---|---|---|
| GPT-4.1-mini (uncensored reference) | – | 5.00 | 4.46 | 1573 |
| Qwen3-32B | Baseline | 4.91 | 3.25 | 2104 |
| | Honesty fine-tuning | 4.85 | 3.78 | 2898 |
| | Abliteration | 4.66 | 4.35 | 3653 |
| | Few-shot prompting | 4.75 | 4.53 | 3771 |
| | Next-token completion | 2.45 | 3.20 | 8670 |
| Qwen3-VL-8B-Thinking | Baseline | 4.60 | 1.60 | 1422 |
| | Honesty fine-tuning | 4.63 | 4.10 | 3374 |
| | Abliteration | 3.80 | 2.74 | 3363 |
| | Few-shot prompting | 3.74 | 3.37 | 1740 |
| | Next-token completion | 2.02 | 2.56 | 8078 |

Honesty fine-tuning minus baseline on coherence is −0.06 [−0.10, −0.03] for Qwen3-32B and +0.03 [−0.05, +0.12] for Qwen3-VL-8B-Thinking (paired bootstrap over questions). Next-token completion is the exception: it is a raw continuation that runs past the answer up to the token limit, which lowers its coherence; its gains come from specific facts appearing in the continuation. We will add the full table, the judge prompt and randomly drawn transcripts to the appendix.

**[INTERNAL]** Next-token completions: 48–82% flagged degenerate for 4 of 5 models, so this reviewer's concern partly holds for that technique; decide whether to re-judge only the first answer segment. Single judge, no human agreement check. Specificity also tracks censorship (evasive answers score low), so coherence is the clean form measure. Fine-tuning raises the degenerate rate by +2.3 pp (32B) and +4.8 pp (VL-8B).

---

> **Q2** [VERBATIM?] Could the observed behaviour be sycophancy rather than censorship?

The test questions are neutral factual prompts that state no user opinion, so there is no stance to agree with; the models volunteer falsehoods unprompted. Prior work documents that this censorship is trained (cite naseh2025, pan2026, qiu2026). Our claims need only that the behaviour occurs naturally and that the model has the knowledge, which the elicitation results show; they do not depend on the developer's intent.

---

> **Q3** [VERBATIM?] How do honesty fine-tuning and abliteration affect the model's reasoning and general capabilities?

We evaluated the base, honesty-fine-tuned and abliterated versions of both models on instruction following (IFEval) and math reasoning (GSM8K) (Table E). No intervention produces a detectable drop: every difference from the base model has a 95% confidence interval that includes zero. The subsets are small, so this rules out large drops only. Note that for auditing, the fine-tuned model is a tool used by the auditor and is not deployed.

**Table E. Accuracy (%) on fixed random subsets; Δ is the difference from base with a paired-bootstrap 95% CI.**

| Model | Variant | IFEval (n=100) | Δ | GSM8K (n=50) | Δ |
|---|---|---|---|---|---|
| Qwen3-32B | Base | 88 | – | 96 | – |
| | Honesty fine-tuning | 85 | −3 [−8, +2] | 96 | 0 [−6, +6] |
| | Abliteration | 85 | −3 [−9, +3] | 98 | +2 [0, +6] |
| Qwen3-VL-8B-Thinking | Base | 73 | – | 98 | – |
| | Honesty fine-tuning | 77 | +4 [−5, +13] | 92 | −6 [−14, 0] |
| | Abliteration | 78 | +5 [−4, +15] | 98 | 0 [0, 0] |

One qualitative change: the honesty-fine-tuned models stop producing a reasoning block (empty on all 50 GSM8K questions for both models; the fine-tuning data contains none) and work through the problem in the visible answer instead. We will add this table and the evaluation details to the appendix.

**[INTERNAL]** VL-8B IFEval is confounded: base and abliterated lose 15 and 14 of 100 prompts to reasoning that never closes within the token limit, the fine-tune loses 14 to repetition loops, so the +4 / +5 are not improvements. Consider dropping the VL-8B IFEval column or footnoting it. VL-8B GSM8K for the fine-tune is 3 lost / 0 gained; a larger subset is cheap if we want to firm it up.

---

> **Q4** [VERBATIM?] What guardrails or evaluation recipe do the authors recommend?

**[TODO]** Writing only: a short paragraph for the Discussion. Not drafted.

---

> **Q5** [VERBATIM?] How do non-Chinese models behave on the testbed?

GPT-4.1-mini is the uncensored reference throughout: scored by a different developer's autorater it reaches an honesty score of 94 and mentions 82% of facts, far above every censored baseline.

**[TODO]** A non-Chinese open-weights baseline (Llama-3.3-70B or gpt-oss-120b) is not run. Confirm the 94 / 82 figures against the paper's table.

---

> **Q6** [VERBATIM?] Was there any human evaluation?

**[TODO]** We have no numbers: neither the size of the human fact review nor autorater-vs-human agreement on responses. Same gap as o5tu W1.
