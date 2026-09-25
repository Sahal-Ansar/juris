"""Record a dummy run (FakeProvider, no API key) and print its manifest path.

Usage: uv run scripts/dummy_run.py [profile] [--db]
    --db  also save the manifest to Postgres (needs `docker compose up -d`)
"""

import asyncio
import sys

from juris.config import Role, get_settings, load_profile, resolve_run_config
from juris.llm import ChatMessage, LLMGateway
from juris.llm.providers import FakeProvider
from juris.runs import RunRecorder


async def main(profile_name: str, use_db: bool) -> None:
    settings = get_settings()
    config = resolve_run_config(settings, load_profile(profile_name))
    model = config.model_for(Role.JUDGE)
    fake = FakeProvider({"dummy": ["dummy answer"]})
    gateway = LLMGateway({model.provider: fake}, prices=settings.llm_prices, cache_mode="off")

    with RunRecorder(
        config,
        corpus_snapshot_id="dummy",
        embedding_model="none",
        ledger=gateway.ledger,
        runs_dir=settings.runs_dir,
        db_url=settings.database_url(driver=None) if use_db else None,
    ) as run:
        await gateway.complete(
            [ChatMessage(role="user", content="ping")], model=model, tags={"agent": "dummy"}
        )
    print(run.manifest_path)


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    asyncio.run(main(args[0] if args else "juris_full", "--db" in sys.argv))
