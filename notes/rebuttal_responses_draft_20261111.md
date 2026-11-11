<!--
Draft v2 of per-reviewer responses (2026-11-11). Quotes are verbatim from the OpenReview reviews.
Sources: notes/rebuttal_results_20261110.md, notes/rebuttal_plan_20261107.md,
paper at github.com/cywinski/ICLR_Chinese_Auditing (5575d93).

Markers:
  [TODO]       we have nothing (or not enough) to say yet.
  [INTERNAL]   note for us, delete before posting.
-->

# Reviewer tLiL

> Benchmark reliability: As far as i am concerned, the paper seems to propose no new methods, so its main contribution focuses on the benchmark. However, the benchmark is relatively modest, with only 90 test questions and a 10-question dev set. The reference claims are also mainly generated and examined by only LLMs (GPT-4.1-mini, and Gemini 3 Flash), which are relatively old models.

We agree the contribution is the testbed and the systematic comparison on it, not a new method. On size, see our reply to your first question below (Table A).

On the reference claims: GPT-4.1-mini was chosen deliberately because it is capability-matched to the audited models, and each fact is then verified by a model from a different developer (Gemini 3 Flash, released December 2025). Facts are restricted to widely documented aspects of each topic and must appear in at least 3 of 20 rollouts, so model capability is not the bottleneck. As a robustness check we recomputed all results using only high-consensus facts (appearing in at least 6, or at least 10, of 20 rollouts): technique rankings are essentially unchanged (Spearman correlation with the full ranking between 0.91 and 1.00 across metrics and models). We will add this analysis to the appendix.

**[INTERNAL]** Confirm which model did fact extraction before naming it. The Spearman range quoted is for the four metrics in Table C (o5tu); a fifth metric (interrogation "never mentioned") goes down to 0.84 on 32B at ≥ 10.

---

> Result and claim reliability: The paper presents probes as a cheaper alternative to asking the model to judge its own responses, but the probes perform close to random on the dev set (table 7). This could also be related to the dev set having only 10 questions.

Table 7 reports probe *selection* on the 10 dev questions, not the result. The test-set result is in the lie-detection tables: balanced accuracy 71.3 (Qwen3-32B) and 82.7 (Qwen3-VL-8B-Thinking) on 900 responses each. We agree the paper should show AUROC on the test set next to the dev numbers, and we added it (Table B).

The dev/test gap has two causes, which we will state in the paper. The dev numbers rest on about 100 responses with very few truthful ones, and the dev selection used a coarser label (any response mentioning no fact counts as untruthful), whereas the test evaluation uses the truthful/untruthful definition of Sec. 4.2, which excludes ambiguous responses.

**Table B. Probe AUROC on test-set baseline responses (labels of Sec. 4.2).**

| Model | Layer | Truthful / untruthful | AUROC [95% CI] |
|---|---|---|---|
| Qwen3-32B | 47 | 230 / 440 | 0.861 [0.796, 0.914] |
| Qwen3-VL-8B-Thinking | 20 | 31 / 667 | 0.923 [0.804, 0.977] |

We will also soften the claim: probes match the prompted baseline for Qwen3-VL-8B-Thinking but trail it by about 15 points for Qwen3-32B, and the Qwen3-VL-8B estimate rests on 31 truthful responses.

**[INTERNAL]** Decide before posting: (1) under the dev-selection labels the test AUROC is 0.68 / 0.63, so the gap is largely the label definition; (2) the stored 32B threshold (Alpaca-control 99th percentile) flags 43% of truthful test responses, so "Recall@1%FPR" is not 1% FPR on test; recalibrated on test truthful responses, recall is 47.5% (32B) and 21.4% (VL-8B). I left the recall column out of Table B on purpose.

---

> See the weaknesses: How to reliably compare the results of different methods with 90 test questions and a 10-question dev set?

The unit of measurement is the fact judgment, not the question: each technique is scored on 900 responses (10 per question) and about 14k per-fact judgments per model. To quantify uncertainty at the question level, we added a paired bootstrap over the 90 test questions (Table A). Gains in honesty score and facts mentioned exclude zero for every technique except user prefill, which we already report as ineffective.

The dev set is small on purpose: it is used only to pick one configuration per technique, so the 90 test questions stay held out.

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

**[INTERNAL]** "Responses with lies" does not robustly improve for several techniques (CI includes 0), and abliteration on VL-8B raises it by +12.3 pp; the full appendix table will show this, so the main text should not over-claim on lies. The user-prefill evaluation files contain repeated copies (2 and 5 distinct responses per question instead of 10); fix or footnote.

---

> The paper concludes that its findings will hold for future large models, but this is based on testing only two of its methods on three current models. Why are the other methods not tested on these bigger models, and if so, how to reach the conclusion that the findings will hold for future large models?

The other techniques need the weights loaded on a GPU (steering, fine-tuning, abliteration, probes), which was impractical for models of this size, so we evaluated the two best black-box techniques. We agree the statement about future models is too strong and will change it to say that the results transfer to current frontier open-weights models.

**[TODO]** Optional additions not run yet: evaluate the existing Kimi-K2-Thinking samples (a fourth frontier model), and system prompt + assistant prefill on the three frontier models (API only).

---

# Reviewer 11SP

> Does the evaluation account for vague and irrelevant outputs generated by these models, particularly after fine-tuning, to ensure that such responses do not artificially inflate the evaluation scores?

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

**[INTERNAL]** Next-token completions: 48–82% flagged degenerate for 4 of 5 models, so this concern partly holds for that technique; decide whether to re-judge only the first answer segment. Single judge, no human agreement check. Specificity also tracks censorship (evasive answers score low), so coherence is the clean form measure. Fine-tuning raises the degenerate rate by +2.3 pp (32B) and +4.8 pp (VL-8B).

---

> Could the observed dishonest and untruthful behavior be attributed to undesirable model behavior, such as sycophancy, rather than deliberate training to induce such behavior?

Sycophancy means agreeing with the view the user expresses, and our questions express none. We checked this with an LLM judge that labelled, blind to source, our test questions alongside control sets (Table F): 88 of 90 test questions carry no user stance, the same as neutral references, while prompts from a sycophancy benchmark are flagged in 90 of 90 cases. Where a question's wording leans at all, it leans towards the account critical of the government (42 of 90), never signalling that the user wants the official line.

The falsehoods nevertheless side with the official position (Table G): 76% of Qwen3-32B's and 94% of Qwen3-VL-8B-Thinking's responses containing a falsehood deny, minimise or reframe in the government's favour, against 29% for the uncensored reference, whose errors are mostly plain factual mistakes. Questions that presuppose the critical account draw no fewer such falsehoods than neutral ones (29% vs 27% of responses for Qwen3-32B; 48% vs 39% for Qwen3-VL-8B-Thinking). The models therefore contradict the framing of the question rather than follow it, which is the opposite of sycophancy. Prior work also documents that this censorship is trained (cite naseh2025, pan2026, qiu2026). We will add both tables and the judge prompts to the appendix.

**Table F. User stance in the prompt, as labelled by the judge (90 prompts per set).**

| Prompt set | No stance | Implied | Explicit |
|---|---|---|---|
| Our test questions | 88 | 2 | 0 |
| GSM8K (neutral control) | 90 | 0 | 0 |
| TruthfulQA (neutral reference) | 87 | 3 | 0 |
| Sycophancy benchmark, stated opinion | 0 | 0 | 90 |
| Sycophancy benchmark, user persona | 10 | 79 | 1 |

**Table G. Direction of falsehoods in baseline responses that contradict at least one ground-truth fact.**

| Model | Responses with a falsehood (of 900) | Aligned with official position | More critical than the facts | Not politically directional | Mixed |
|---|---|---|---|---|---|
| Qwen3-32B | 329 | 249 (76%) | 12 | 64 | 4 |
| Qwen3-VL-8B-Thinking | 409 | 385 (94%) | 0 | 24 | 0 |
| GPT-4.1-mini (uncensored reference) | 35 | 10 (29%) | 6 | 19 | 0 |

**[INTERNAL]** Scope: this rules out sycophancy towards what the prompt reveals about the asker. It cannot rule out a model that assumes by default any asker wants the official line, and it does not by itself show deliberate training. Neutral vs critical is "no reduction", not a significant increase (CIs overlap). The judge labels 10 of 35 GPT-4.1-mini errors as official-aligned (5 on one question), a rough upper bound on its false-positive rate. Single judge, no human check of the direction labels. The 2 "implied" test questions use the word "crackdown", so they lean critical. Sycophancy sets are from Anthropic model-written-evals (NLP survey; political typology).

---

> What influence does the different alignment method have on the reasoning capability of the model

We evaluated the base, honesty-fine-tuned and abliterated versions of both models on instruction following (IFEval) and math reasoning (GSM8K) (Table E). No intervention produces a detectable drop: every difference from the base model has a 95% confidence interval that includes zero. The subsets are small, so this rules out large drops only. Note that for auditing, the modified model is a tool used by the auditor and is not deployed.

**Table E. Accuracy (%) on fixed random subsets; Δ is the difference from base with a paired-bootstrap 95% CI.**

| Model | Variant | IFEval (n=100) | Δ | GSM8K (n=50) | Δ |
|---|---|---|---|---|---|
| Qwen3-32B | Base | 88 | – | 96 | – |
| | Honesty fine-tuning | 85 | −3 [−8, +2] | 96 | 0 [−6, +6] |
| | Abliteration | 85 | −3 [−9, +3] | 98 | +2 [0, +6] |
| Qwen3-VL-8B-Thinking | Base | 73 | – | 98 | – |
| | Honesty fine-tuning | 77 | +4 [−5, +13] | 92 | −6 [−14, 0] |
| | Abliteration | 78 | +5 [−4, +15] | 98 | 0 [0, 0] |

One qualitative change in reasoning: the honesty-fine-tuned models stop producing a reasoning block (empty on all 50 GSM8K questions for both models; the fine-tuning data contains none) and work through the problem in the visible answer instead. We will add this table and the evaluation details to the appendix.

**[INTERNAL]** Only fine-tuning and abliteration were tested; the reviewer asks about "the different alignment methods" (steering, prompting untested). VL-8B IFEval is confounded: base and abliterated lose 15 and 14 of 100 prompts to reasoning that never closes within the token limit, the fine-tune loses 14 to repetition loops, so the +4 / +5 are not improvements. Consider dropping the VL-8B IFEval column or footnoting it. VL-8B GSM8K for the fine-tune is 3 lost / 0 gained; a larger subset is cheap if we want to firm it up.

---

> Author should consider adding a section discussing possible suggestions on efficient guardrails and an evaluation process to test the newer release of these models.

We will add a paragraph to the Discussion with a concrete recipe for auditing a new release: run the testbed questions with the baseline, few-shot prompting and next-token completion (all black-box, API only), and use prompted self-classification to flag false responses.

**[TODO]** The paragraph itself is not written. Decide whether we want to say anything about guardrails beyond the evaluation recipe.

---

> Author should consider adding experiments on models apart from the Chinese models.

GPT-4.1-mini is the uncensored reference throughout: scored by a different developer's autorater it reaches an honesty score of 94 and mentions 82% of facts, far above every censored baseline.

**[TODO]** A non-Chinese open-weights baseline (Llama-3.3-70B or gpt-oss-120b) is not run. Confirm the 94 / 82 figures against the paper's table.

---

> Did authors only use an LLM as a judge for evaluation, or did they consider Human evalaution of final output alslo.

Responses are scored by an LLM autorater (Gemini 3 Flash), chosen to be from a different developer than the model that generated the reference facts. A subset of the reference facts was reviewed by humans against sources such as Wikipedia.

**[TODO]** We have no numbers: neither the size of the human fact review nor autorater-vs-human agreement on responses. Same gap as o5tu weakness 1.

---

# Reviewer o5tu

> My main concern is the reliability of the benchmark’s ground truth. The paper primarily relies on supposedly uncensored LLMs, mainly GPT-family models, to extract atomic facts and cross-validate them. This assumes that these models are free from relevant censorship and possess accurate factual knowledge. Their training procedures are not fully disclosed, making the first assumption difficult to verify. These models may also share ideological biases, so agreement among them does not necessarily establish factual correctness. Moreover, the benchmark focuses on complex political events, where competing accounts may require careful source-based adjudication. These issues could compromise the benchmark’s validity and weaken the conclusions drawn from it.

We agree that agreement among LLMs does not establish correctness, and we will say so in the limitations. Three design choices limit the risk. Verification is done by a model from a different developer (Gemini 3 Flash) than the one that generated the rollouts. Facts are restricted to widely documented aspects of each topic and must appear in at least 3 of 20 rollouts. And a response is labelled a lie only when it explicitly contradicts a fact, not when it omits one or frames it differently.

To test sensitivity to fact quality, we recomputed all results using only high-consensus facts (appearing in at least 6, or at least 10, of 20 rollouts). Technique rankings are essentially unchanged (Table C); the few changes are swaps among near-tied techniques. Our conclusions are comparisons between techniques on the same fact set, so they are robust to a small rate of incorrect facts.

**Table C. Spearman correlation of technique rankings: high-consensus facts vs all facts.**

| Metric | Qwen3-32B, ≥6 | Qwen3-32B, ≥10 | Qwen3-VL-8B, ≥6 | Qwen3-VL-8B, ≥10 |
|---|---|---|---|---|
| Facts retained (of 1529) | 857 | 540 | 857 | 540 |
| Facts mentioned (chat) | 0.98 | 0.96 | 1.00 | 1.00 |
| Responses with lies (chat) | 0.99 | 0.97 | 0.98 | 0.98 |
| Facts mentioned (interrogation) | 0.92 | 0.91 | 1.00 | 1.00 |
| Facts contradicted (interrogation) | 1.00 | 0.98 | 1.00 | 0.96 |

We will add this analysis to the appendix and restore a limitation sentence on contested questions.

**[TODO]** The reviewer asks for source-based adjudication. A human audit of a random sample of facts against primary sources (error rate) and agreement of a second autorater with Gemini are both not done; the paper says only that "a subset" was human-reviewed, with no numbers. This is the strongest missing piece for this reviewer.

**[INTERNAL]** High-consensus filtering addresses noise, not shared ideological bias across models; do not oversell Table C on that point. The paper says "over 1500 facts" (1529 with count ≥ 3) while one evaluation summary shows 1410 evaluated per sample; reconcile.

---

> The paper’s data and evaluation focus on politically sensitive events in China, which limits the generalizability of its claims to broader domains such as science and history. Could the authors apply similar censorship training procedures to construct models that suppress selected knowledge in these domains? Evaluating the proposed methods on such models would help establish their effectiveness across domains and strengthen the paper’s conclusions.

Training a model to suppress selected knowledge is the synthetic setup that prior work already covers (Marks et al. 2025; Cywiński et al. 2025; AuditBench), and the same techniques were evaluated there. Our contribution is complementary: a testbed where the suppression arises naturally from the developer's training rather than from our own. The techniques themselves are domain-agnostic: the honesty fine-tuning data, the probe training data and the few-shot examples contain nothing about China. We will list evaluation on further domains as future work.

**[TODO]** A second natural domain (lyrics recitation, E3 in our spec) is not run. Decide whether to run it or keep this as future work only.
