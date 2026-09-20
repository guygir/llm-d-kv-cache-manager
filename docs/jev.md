# Jev vs regular classifiers

Jev is TypeSafe's first [System One](https://docs.typesafe.ai/concepts/system-one.md)
model: a prefill-only decision model. You send a `state` plus typed questions;
you get back values your code can `if` / `switch` / threshold on. It does not
generate a reply.

This note is a field guide for builders on this repo. It is not a substitute for
TypeSafe's own docs.

## Read their docs

We are not in an era where people sit down and read manuals. Do it anyway for
this one.

TypeSafe's documentation is the shortest path to the actual product contract —
primitives, structured criteria, confidence, and cookbooks that are real
workflows, not toy prompts:

- [Introduction](https://docs.typesafe.ai/introduction.md)
- [Quick start](https://docs.typesafe.ai/introduction/quickstart.md)
- [Primitives](https://docs.typesafe.ai/primitives.md) — Choice, Score, Noul
- [Criteria can be structured](https://docs.typesafe.ai/primitives/advanced.md)
- [AI primer / RLCD](https://docs.typesafe.ai/introduction/machine-learning-primer.md)
- [Confidence](https://docs.typesafe.ai/confidence.md)
- [Patterns](https://docs.typesafe.ai/patterns.md) and the
  [cookbook index](https://docs.typesafe.ai/llms.txt)

A running comparison of the three shapes (frozen-label classifier, LLM JSON
mode, System One) lives in
[`examples/jev-vs-classifier`](../examples/jev-vs-classifier/README.md).

```bash
go run ./examples/jev-vs-classifier
```

## Classifiers are not new. Criteria-in-the-question is.

A fair objection: "this is just a classifier." Linear models, BERT heads, and
zero-shot NLI have classified text for years. That part is not new.

What is new — and what an LLM *can* fake, badly — is that **the label space and
the rubric are part of the request**, written in natural language (or JSON), and
the model returns a distribution **over those exact options**.

| | Frozen-label classifier | LLM JSON / "classify this" | Jev (System One) |
|---|---|---|---|
| Who defines the labels? | Training set / `num_classes` | The prompt, every call | `criteria` on the question |
| Change the taxonomy | Relabel + retrain | Rewrite the prompt | Change `criteria` |
| What comes back | `argmax` over a fixed head | Generated tokens you parse | Typed `choice` / `score` / `noul` |
| Can it invent a label? | No | Yes (wrong key, extra prose) | No — closed over your options |
| Calibration | Sometimes, after the fact | A generated "90%" is not p=0.9 | Trained for calibrated probabilities (RLCD) |
| Several judgments | N heads or N models | One long generation, or N calls | N questions, one shared `state`, parallel |
| Decode / KV growth | None | Full decode | Prefill only |

An LLM *can* take criteria in a prompt. There is no comparison in the output
contract: you still pay for tokens, you still parse, you still cannot treat
`"confidence": "high"` as a probability, and a jailbreak or a missing comma
breaks the workflow. Jev makes the criteria a first-class input and the
distribution a first-class output.

TypeSafe's own wording: instructions are where the evaluation logic goes;
`criteria` are the allowed answers. Question IDs are for *your* code — they are
not sent to the model.

## Three primitives

Same `state`. Different answer shapes.

| Type | Question | Returns |
|---|---|---|
| **Choice** | Which of these options? | `choice`, `probabilities`, `confidence` |
| **Score** | Where on this ordered rubric? | `score` (can sit between levels), `probabilities`, `confidence` |
| **Noul** | Is this true? | `noul` in `[0, 1]` — no separate confidence field |

A support ticket can ask all three in one call: route the queue (Choice), score
frustration (Score), test urgency (Noul). Answers are independent. Adding a
question does not become hidden context for another. If judgment B truly needs
answer A, that is a second request in *your* code.

`confidence` on Choice/Score is derived from how peaked the distribution is, not
a second learned "I am sure" token. TypeSafe's Choice formula is

\[
c = \frac{p_{\max} - 1/K}{1 - 1/K}
\]

for \(K > 1\). A 0.8 peak over three options is confidence 0.7. You can ignore
their summary and threshold the raw `probabilities` yourself.

## What a request looks like

```json
{
  "state": "Help! My payouts have been failing for 3 days. The bank says everything is fine.",
  "model": "jev-latest",
  "questions": {
    "queue": {
      "type": "choice",
      "instructions": "Which team should handle this ticket?",
      "criteria": {
        "payments": "Payout failures and payment processing",
        "account": "Login and account access",
        "other": "Something else"
      }
    },
    "urgent": {
      "type": "noul",
      "instructions": "Does this message require urgent human attention?",
      "criteria": {
        "true": "Explicitly time-sensitive or repeated failure",
        "false": "Ordinary request, no urgency"
      }
    }
  }
}
```

`POST https://api.typesafe.ai/v1/systemone` with `Authorization: Bearer …`.
SDKs: [Python](https://docs.typesafe.ai/sdk/python.md),
[JavaScript](https://docs.typesafe.ai/sdk/javascript.md).

The same state with a *different* `criteria` map is a different decision. No
retrain. That is the move a frozen classifier cannot make, and the move an LLM
can only approximate by generating more prose.

## Serving shape (why this repo cares)

Jev is not a new attention op and it does not invent a new KV layout. The
serving graph is:

1. Encode `state` once (the reusable prefix).
2. Evaluate each question as an isolated suffix (tree / block-causal mask).
3. Read probabilities from a head. No decode loop, so no decode-phase KV growth.

Implications for a KV-cache-aware router like this one:

- Score the **state** prefix, not a packed `[state|Q1|Q2|Q3]` string. Questions
  must not attend to each other; treating the packed request as one causal chain
  is the wrong mask *and* a unique hash that never hits.
- Prefill-only: eviction pressure is compute-bound, not decode-memory-bound.
  Prefix reuse of a repeated `state` is the win.
- Jev's public tokenizer matches no HuggingFace tokenizer this package loads.
  If you ever served a Jev-like engine, use engine-emitted tokens
  (`ScoreTokens` on current upstream), not `CachedHFTokenizer`.
- TypeSafe's hosted API never exposes tensors. **No change is required in this
  indexer to call Jev as a client.**

See [architecture.md](architecture.md) for how *this* indexer scores linear
vLLM prefixes today.

## What we are not claiming

- Jev is not a drop-in chat model. It cannot write the customer an email.
- "Calibrated" is a statement about groups of predictions, not a guarantee on
  one ticket.
- Underspecified criteria get underspecified decisions. The work moves into the
  rubric, the same way Bend moves work into `LAWS.bend`.
- The local backend in `examples/jev-vs-classifier` is a **teaching mock**
  (lexical overlap + softmax). It demonstrates the *interface*. It is not Jev.

## Further reading

- TypeSafe: [docs.typesafe.ai](https://docs.typesafe.ai/introduction.md)
- Architecture reconstruction (external, not official):
  [Jev's Architecture Unmasked](https://archerhume.com/posts/jevs-architecture-unmasked/)
- Cookbooks that make the "criteria in the request" claim concrete:
  [hierarchical classification](https://docs.typesafe.ai/cookbooks/hierarchical_classification.md),
  [classifying RAG passages](https://docs.typesafe.ai/cookbooks/classifying_rag_passages.md),
  [parallel questions](https://docs.typesafe.ai/cookbooks/parallel_questions.md)
