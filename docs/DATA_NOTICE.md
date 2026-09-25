# Data notice and attribution

Juris uses the following data. Details and field-level notes: `docs/data/sources.md`.

## Indian Supreme Court Judgments

Judgments of the Supreme Court of India, from the dataset *Indian Supreme Court Judgments* maintained by Dattam Labs ([github.com/vanga/indian-supreme-court-judgments](https://github.com/vanga/indian-supreme-court-judgments)), available on the Registry of Open Data on AWS (`s3://indian-supreme-court-judgments`). Originally sourced from the Supreme Court of India eCourts portal. Licensed under [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/).

## Indian High Court Judgments

Judgments and orders of Indian High Courts, from the dataset *Indian High Court Judgments* maintained by Dattam Labs ([github.com/vanga/indian-high-court-judgments](https://github.com/vanga/indian-high-court-judgments)), available on the Registry of Open Data on AWS (`s3://indian-high-court-judgments`). Originally sourced from the eCourts judgments portal. Licensed under [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/).

## KanoonGPT Indian Case Laws

Case metadata from [KanoonGPT/indian-case-laws](https://huggingface.co/datasets/KanoonGPT/indian-case-laws) (KanoonGPT Open Legal Data Initiative), licensed under the Apache License 2.0. It is derived from the two CC-BY-4.0 datasets above.

## KanoonGPT Indian Legal Documents

Statute and notification text from [KanoonGPT/indian-legal-documents](https://huggingface.co/datasets/KanoonGPT/indian-legal-documents), published under the Apache License 2.0. The upstream source of each text isn't disclosed, so **Juris doesn't redistribute text or chunks derived from this dataset**. It is used locally for research only.

## India Code

Statute text is spot-checked by hand against the IndiaCode Portal, Legislative Department, Ministry of Law and Justice, Government of India. No content from the Portal is scraped or redistributed.

## Changes

Juris extracts text from the judgment PDFs, removes layout noise, splits it into paragraphs and chunks, and adds embeddings and extracted citations. Those changes are ours; the original texts belong to their sources above.

Juris is a research prototype and doesn't give legal advice.
