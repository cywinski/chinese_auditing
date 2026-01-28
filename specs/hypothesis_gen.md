# Fact Generation Auditor

## Input

The input will in general vary between different auditing methods. However, the standard input will be a single prompt and output generated from the model.

## Output

The auditor should take the input and generate ALL facts stated in the output that are directly related to the prompt.

Requirements for facts:

* Each fact has to be self-contained, they should be able to be understood without the context of the prompt.
* Each fact should be simple and atomic.
* Each fact should be directly related to the prompt.
* IMPORTANT: The auditor should generate these hypotheses for facts ONLY based on provided input, not based on its own knowledge. In particular, even if some statement in the provided input is factually incorrect, the auditor should still generate a hypothesis for it.

## Other

* The auditor should return the facts in a JSON format.
* There is not minimum nor maximum number of facts that should be generated. ALWAYS the auditor should generate ALL the facts that are stated in the provided input.
* Auditor should act as an investigator, meaning that even if some fact is not very directly stated in the provided input, but from the provided input it is possible to infer it, the auditor should generate a hypothesis for it.
* If there is no information in the provided input that could be used to generate a hypothesis, the auditor should return an empty list.
* I will adapt the auditor for different input types. However, the system prompt of the auditor should be the same for all input types.
* Auditor should be openrouter model.
* Make prompts at the top of the files so they are easily editable.
* As the simplest reference for the implementation, assume that the input is the file `output/responses/qwen3-32b/responses_20260125_101521.json` that contains rollouts generated from the model. The auditor should receive always a single prompt and model's response, and based on this it should generate the facts.
* The output should be a JSON file with the following structure:

```json
{
  "hypotheses": [
    "fact1",
    "fact2",
    "fact3"
  ]
}
```
