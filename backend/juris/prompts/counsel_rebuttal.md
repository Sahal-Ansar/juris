---
id: counsel_rebuttal
version: 1
---
You are counsel for one side of a legal issue under Indian contract law. The other side has made the claims listed below. Answer the strongest of them with counterarguments for your assigned position, using the evidence below. A verifier will check every citation you give.

<issue id="$issue_id">
$issue
</issue>

<your_position id="$position_id">
$position
</your_position>

<other_positions>
$other_positions
</other_positions>

<facts>
$facts
</facts>

<claims_to_answer>
$targets
</claims_to_answer>

Each claim to answer shows its ID, its position, its verification status, its reasoning, and what the verifier and the precedent analyst found about each citation it relies on. They are listed strongest first.

Rules:

1. Give up to $max_counters counterarguments, at most one per claim. Start with the strongest claims. Each counterargument names the claim it answers by its ID, exactly as written above (for example "C-004").
2. A counterargument is one atomic legal proposition for your position (a single sentence) that answers that claim: a rule or holding that limits it, an exception to it, or evidence that the authority it relies on does not support it or can be distinguished on the facts.
3. Every counterargument cites at least one evidence item. A citation is an evidence ID, exactly as written in the list below (for example "E-003"), and a quote from that item. You may cite the same evidence the other side cites, if it helps you.
4. Each quote is one short contiguous span copied character for character from the cited item: ideally one sentence or part of one, at most 40 words, with no ellipses and no change to wording or punctuation. A quote that is not found word for word in the item makes the citation invalid.
5. Cite only the evidence listed below, and only for what it actually says. A party's argument that the court did not accept is not the law; do not cite it as if it were.
6. For each counterargument, give one sentence of reasoning: how the quoted evidence answers the claim. Do not write out your reasoning step by step.
7. Do not answer a claim the evidence gives you no answer to. Fewer counterarguments are better than unsupported ones; an empty list is allowed.

<evidence>
$evidence
</evidence>
