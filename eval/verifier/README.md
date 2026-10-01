# Verifier pairs

`juris_pairs.yaml` is the hand-labelled set for the Citation Verifier's entailment check
(PLAN 6.1: 50 claim-passage pairs; target agreement of the model verdict with the label of 0.8
or more). Each pair asks the verifier's question: given a claim, a quote taken verbatim from a
corpus chunk, and the chunk text as the passage, does the quoted text in context support the claim?

## Labels

| Label | Meaning |
|---|---|
| `supports` | The passage states, or necessarily implies, the whole claim. |
| `partially_supports` | It supports part of the claim, or a narrower or qualified version: the claim overstates, drops a condition, or generalises. |
| `does_not_support` | Related topic, but it does not establish the claim, including a passage that only reports a party's argument or another court's view without deciding it. |
| `contradicts` | The passage states the opposite or rejects the proposition, including a view it reports and then rejects (counsel's argument the court turns down, a statement a higher court reversed). |

The set holds 20 `supports`, 9 `partially_supports`, 9 `does_not_support` and 12 `contradicts`.

## How the pairs were built

Only Juris-Eval **dev** items were used (`eval/juris_eval/dev.jsonl`), so the test split stays
clean. For every gold authority pinpoint, `scripts/verifier_pairs.py candidates` finds the
paragraph and the chunk whose character span contains it (working file under
`C:/juris-data/wip/verifier_6_1/`, not committed). Each passage is a whole chunk, as the
verifier sees it. The claims and labels were then written from a read of every passage, so a
label rests on the text, not on how the claim was constructed: a gold proposition whose
pinpoint paragraph does not state it is not labelled `supports`.

Claims are one-sentence advocate-style propositions. Positives restate or paraphrase what the
passage says; negatives are in the same doctrine area and use plausible wording. The
`construction` field records how a claim was made:

- `gold`: the Juris-Eval item's proposition, as stated or closely.
- `paraphrase`: the same proposition in other words.
- `overstated`: an absolute or wider claim than the passage supports.
- `dropped_condition`: the rule without a condition or exception the passage attaches.
- `negated`: the proposition reversed.
- `wrong_passage`: a claim from one part of an authority, or another authority, paired with a
  passage on a related topic that does not address it.
- `argument_only`: the quote is counsel's contention, or a lower court's view that was reversed
  or not adopted.
- `other`: a plausible claim the passage does not reach.

The pairs were drafted and labelled from the corpus text, then every label was checked against
its passage in a second pass (five changed: VP-008, VP-016, VP-036 and VP-049 became
`contradicts`; VP-041 got a new claim). `reviewed_by` is `null` on every
pair until a human reviews it; the agreement target is measured against these labels, so treat
them as provisional until then.

## Fields

| Field | |
|---|---|
| `id` | `VP-001` to `VP-050`. |
| `item` | The Juris-Eval dev item the authority comes from. |
| `doc_id`, `chunk_id` | The judgment and the chunk used as the passage. |
| `pinpoint` | The paragraph (`35`, `p-2`, `h-3`) the quote comes from; it lies inside the chunk. |
| `claim` | The proposition to verify. |
| `quote` | A verbatim span of the passage (after the shared quote normalisation). OCR errors are the corpus text. |
| `passage` | The full text of `chunk_id` as stored in the database. |
| `label` | One of the four labels. |
| `construction` | How the claim was made (above). |
| `rationale` | Why the label is right, citing the passage. |
| `reviewed_by` | `null` until a human has checked the label. |

## Checks

```
uv run scripts/verifier_pairs.py check
```

This validates the file against the corpus database: exactly 50 pairs, unique ids, labels and
constructions from the lists above, at most four pairs per chunk and at least 20 distinct chunks,
every `quote` passing `juris.verify.quote.quote_in_text` against its `passage`, every `passage`
equal to the current text of its `chunk_id`, and every `pinpoint` inside that chunk. It prints
the label and construction counts. If the corpus is re-chunked, the check fails until the pairs
are rebuilt on the new chunks.

`build` regenerates `juris_pairs.yaml` from a spec of the hand decisions (item, chunk, pinpoint,
claim, quote, label, construction, rationale), filling `doc_id` and `passage` from the database:

```
uv run scripts/verifier_pairs.py build --spec path/to/spec.yaml
```
