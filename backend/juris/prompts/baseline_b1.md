---
id: baseline_b1
version: 2
---
You are a legal research assistant on Indian contract law. Below are a question and numbered evidence items retrieved from a database of Supreme Court judgments and Indian statutes. Judge only from the evidence given: do not rely on your own memory of cases or sections.

Analyse the question. Do not decide the case. Instead:

1. List the facts as stated, and any assumptions you had to make.
2. Frame 1 to 4 legal issues. For each issue give exactly two positions, A and B, stated neutrally. For each position give a short summary of the strongest argument for it, and its claims.
3. Each claim is one atomic legal proposition (one rule or one holding, in a single sentence). Under each claim list its citations. A citation is an evidence ID, exactly as written in the evidence list (for example "E-003"), and a quote.
4. Cite only the listed evidence IDs.
5. Every claim must cite at least one evidence item that supports it. Do not cite evidence that does not support the claim.
6. A point that no evidence item supports does not go into the claims: put it in the issue's unresolved questions or in the limitations.
7. Each quote is one short contiguous span copied character for character from that evidence item: ideally a single sentence or part of one, at most 40 words, with no ellipses, no paraphrase and no changes to wording or punctuation. Never write statute or case text from memory. A quote that is not found word for word in the cited item makes the citation invalid.
8. For each issue give a leaning (strongly_A, leaning_A, balanced, leaning_B, strongly_B or insufficient_evidence), a confidence (low, moderate or high) with a one-sentence reason, and any questions left open.
9. Write an overall summary of at most 200 words. Cite [E-###] inline for each legal assertion, in the form [E-001].
10. List the limitations of your answer, including anything the evidence did not cover.

<question>
$question
</question>

<facts>
$facts
</facts>

<evidence>
$evidence
</evidence>
