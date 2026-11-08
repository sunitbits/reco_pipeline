"""Stage 6 - Training tasks. Each fits a model, persists it, and opens an MLflow run
(params logged now; metrics are attached by the evaluation task)."""
import json
import logging
import pickle
import time
from configs import settings as C
from src.features.feature_store import FeatureStore
from src.training.models import PopularityRecommender, HybridMFRecommender
from src.training.tracking import Tracker

log = logging.getLogger(__name__)


def _train(model):
    t0 = time.time(); model.fit(); secs = time.time() - t0
    path = C.MODELS / f"{model.name}.pkl"
    model.fs = None  # features are re-attached at load time; keeps pickle small
    path.write_bytes(pickle.dumps(model))
    tr = Tracker()
    rid = tr.start_run(model.name, {**model.params(), "train_seconds": round(secs, 2), "data_seed": C.SEED})
    tr.log_artifact(rid, path)
    tr.log_artifact(rid, C.FEAT / "feature_registry.json")
    (C.MODELS / f"{model.name}.run.json").write_text(json.dumps({"run_id": rid}))
    log.info("trained %s in %.1fs (run %s)", model.name, secs, rid)


def run_popularity():
    C.ensure_dirs(); _train(PopularityRecommender(FeatureStore().load()))


def run_hybrid():
    C.ensure_dirs(); _train(HybridMFRecommender(FeatureStore().load()))
