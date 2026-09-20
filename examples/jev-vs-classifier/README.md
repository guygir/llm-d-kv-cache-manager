# Jev vs regular classifiers

Companion to [docs/jev.md](../../docs/jev.md). Runs three decision shapes on
the same support tickets:

1. **Frozen-label classifier** — labels baked in at "train" time (`billing`,
   `technical`, `account`). Changing the taxonomy means changing the model.
2. **LLM JSON mode** — criteria live in a prompt; the model still *generates*
   text you have to parse.
3. **System One (Jev-shaped)** — `state` + typed questions. `criteria` are
   part of the request. The local backend is a teaching mock (lexical overlap +
   softmax), not TypeSafe's model.

```bash
go run ./examples/jev-vs-classifier
go test ./examples/jev-vs-classifier
```

Pass `--backend=typesafe` and `TYPESAFE_API_KEY` to hit
`POST https://api.typesafe.ai/v1/systemone` for the System One column. The
classic classifier and the LLM prompt dump stay local.

The interesting beat is ticket 1: the frozen model can only say `billing`. The
System One column is asked twice on the **same** state — first with
`payments / account / other`, then with `payout_ops / bank_escalation / product`
— and does not need a retrain.
