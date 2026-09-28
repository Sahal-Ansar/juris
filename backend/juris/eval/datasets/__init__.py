"""External benchmark loaders (PLAN 5.2, D-011, D-036): AILA 2019, IL-PCR, COLIEE."""

from juris.eval.datasets.aila import fetch_aila2019, load_aila2019
from juris.eval.datasets.coliee import load_task2, load_task4
from juris.eval.datasets.common import (
    BenchmarkMissing,
    Document,
    EntailmentDataset,
    EntailmentExample,
    RetrievalDataset,
    RetrievalQuery,
    benchmarks_dir,
)
from juris.eval.datasets.il_pcr import fetch_il_pcr, load_il_pcr

__all__ = [
    "BenchmarkMissing",
    "Document",
    "EntailmentDataset",
    "EntailmentExample",
    "RetrievalDataset",
    "RetrievalQuery",
    "benchmarks_dir",
    "fetch_aila2019",
    "fetch_il_pcr",
    "load_aila2019",
    "load_il_pcr",
    "load_task2",
    "load_task4",
]
