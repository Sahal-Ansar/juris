"""Authority recall: gold authorities and sections that appear in the final analysis."""

from juris.eval.juris_eval import JurisEvalItem
from juris.eval.metrics.common import MetricResult, analysis_documents, analysis_sections, ratio
from juris.events.fold import CaseView


def authority_recall(view: CaseView, item: JurisEvalItem) -> MetricResult:
    """Recall of the gold judgments (supporting and contrary together, and each apart) among
    the documents the analysis relies on, and of the gold sections among the statute sections
    it cites as evidence. No contrary gold -> ``.contrary`` is None. No analysis -> 0."""
    docs = analysis_documents(view)
    sections = analysis_sections(view)
    supporting = {a.doc_id for a in item.gold_supporting_authorities}
    contrary = {a.doc_id for a in item.gold_contrary_authorities}
    gold_sections = set(item.gold_sections)
    judgments = supporting | contrary
    return MetricResult(
        "authority_recall",
        {
            "authority_recall": ratio(len(judgments & docs), len(judgments)),
            "authority_recall.supporting": ratio(len(supporting & docs), len(supporting)),
            "authority_recall.contrary": ratio(len(contrary & docs), len(contrary)),
            "authority_recall.sections": ratio(len(gold_sections & sections), len(gold_sections)),
        },
        {
            "missing_supporting": sorted(supporting - docs),
            "missing_contrary": sorted(contrary - docs),
            "missing_sections": sorted(gold_sections - sections),
        },
    )
