---
id: verifier_entailment
version: 1
---
You are checking the citations in a legal research answer on Indian law. Below are numbered pairs. Each has a claim, the exact quote the advocate relied on for it, and the passage (the retrieved chunk) the quote comes from, given as context. For each pair, decide whether the quoted text, read in its context, supports the claim.

Judge only from the passage. Do not use outside knowledge of the law, and do not credit the claim for being correct in general: the question is whether this passage establishes it.

Use exactly one label:

- supports: the passage states the whole claim, or necessarily implies it.
- partially_supports: the passage supports part of the claim, or a narrower or qualified version of it. The claim overstates, drops a condition or exception, or generalises beyond what the passage says.
- does_not_support: the passage is on a related topic but does not establish the claim. This includes a passage that only reports a party's argument, or another court's or author's view, without deciding whether it is right, when the claim treats it as the law. Say so in the justification.
- contradicts: the passage states the opposite of the claim, or rejects the proposition. This includes a view the passage reports and then rejects, disapproves or answers (for example, counsel's argument that the court turns down, or a statement a higher court reversed).

Read the quote in its surrounding passage: a sentence can mean something different once the court's reasoning is taken into account (a view stated only to be rejected, a rule followed by an exception, an obiter remark).

Keep each justification to one sentence. Return one judgement for every pair, with its number as "index".

$pairs
