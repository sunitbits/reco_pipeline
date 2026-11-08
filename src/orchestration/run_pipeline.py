"""Stage 8 - Orchestration: dependency-aware workflow runner (graphlib) with logging + retries.
    python -m src.orchestration.run_pipeline                # run everything
    python -m src.orchestration.run_pipeline --only evaluate --with-deps
    python -m src.orchestration.run_pipeline --list
The same TASKS/DEPENDENCIES drive dags/reco_dag.py. Tasks exchange data only through files,
so any task can be re-run independently."""
import argparse
import json
import logging
import sys
import time
from graphlib import TopologicalSorter
from configs import settings as C
from src.ingestion import ingest
from src.validation import validate
from src.preparation import prepare
from src.transformation import transform
from src.training import train
from src.evaluation import evaluate
from src.features.feature_store import FeatureStore

TASKS = {
    "ingest": ingest.run,
    "validate": validate.run,
    "prepare": prepare.run,
    "transform": transform.run,
    "build_features": lambda: FeatureStore().build(),
    "train_popularity": train.run_popularity,
    "train_hybrid": train.run_hybrid,
    "evaluate": evaluate.run,
}
DEPENDENCIES = {  # task -> upstream tasks
    "ingest": [], "validate": ["ingest"], "prepare": ["validate"], "transform": ["prepare"],
    "build_features": ["prepare"],
    "train_popularity": ["transform", "build_features"], "train_hybrid": ["transform", "build_features"],
    "evaluate": ["train_popularity", "train_hybrid"],
}


def _closure(target):
    need, stack = set(), [target]
    while stack:
        t = stack.pop()
        if t not in need:
            need.add(t); stack += DEPENDENCIES[t]
    return need


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--only"); ap.add_argument("--with-deps", action="store_true")
    ap.add_argument("--retries", type=int, default=1); ap.add_argument("--list", action="store_true")
    a = ap.parse_args(argv)
    if a.list:
        for t, d in DEPENDENCIES.items(): print(f"{t:18s} <- {d}")
        return 0
    C.ensure_dirs()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        handlers=[logging.StreamHandler(), logging.FileHandler(C.REP / "pipeline.log")])
    log = logging.getLogger("pipeline")
    keep = set(TASKS) if not a.only else (_closure(a.only) if a.with_deps else {a.only})
    ts = TopologicalSorter({t: [d for d in DEPENDENCIES[t] if d in keep] for t in keep})
    ts.prepare(); status = {}
    while ts.is_active():
        for t in ts.get_ready():          # tasks in one batch are independent (could run in parallel)
            for attempt in range(a.retries + 1):
                t0 = time.time()
                try:
                    log.info(">>> %s (attempt %d)", t, attempt + 1); TASKS[t]()
                    status[t] = {"status": "ok", "seconds": round(time.time() - t0, 2)}; break
                except Exception as e:
                    log.exception("task %s failed", t); status[t] = {"status": "failed", "error": str(e)}
            if status[t]["status"] == "failed":
                (C.REP / "run_status.json").write_text(json.dumps(status, indent=2)); return 1
            ts.done(t)
    (C.REP / "run_status.json").write_text(json.dumps(status, indent=2))
    log.info("pipeline finished: %s", {k: v["status"] for k, v in status.items()})
    return 0


if __name__ == "__main__":
    sys.exit(main())
