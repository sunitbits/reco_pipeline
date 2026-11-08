"""Experiment tracking: MLflow (SQLite backend, local) with a JSON fallback if MLflow is absent."""
import json
import time
import uuid
from configs import settings as C

try:
    import mlflow
    from mlflow.tracking import MlflowClient
    mlflow.set_tracking_uri(f"sqlite:///{C.ROOT / 'mlflow.db'}")
    HAS_MLFLOW = True
except Exception:  # pragma: no cover
    HAS_MLFLOW = False

EXPERIMENT = "recommender-pipeline"


class Tracker:
    def __init__(self):
        if HAS_MLFLOW:
            self.c = MlflowClient()
            exp = self.c.get_experiment_by_name(EXPERIMENT)
            self.exp_id = exp.experiment_id if exp else self.c.create_experiment(
                EXPERIMENT, artifact_location=(C.ROOT / "mlartifacts").as_uri())
        else:
            self.dir = C.ROOT / "runs"; self.dir.mkdir(exist_ok=True)

    def start_run(self, name, params, tags=None):
        if HAS_MLFLOW:
            run = self.c.create_run(self.exp_id, run_name=name, tags=tags or {})
            for k, v in params.items():
                self.c.log_param(run.info.run_id, k, v)
            return run.info.run_id
        rid = uuid.uuid4().hex[:8]
        (self.dir / rid).mkdir(exist_ok=True)
        self._write(rid, {"name": name, "params": params, "tags": tags or {}, "metrics": {}, "started": time.time()})
        return rid

    def log_metrics(self, rid, metrics):
        if HAS_MLFLOW:
            for k, v in metrics.items():
                self.c.log_metric(rid, k.replace("@", "_at_"), float(v))
        else:
            d = self._read(rid); d["metrics"].update(metrics); self._write(rid, d)

    def log_artifact(self, rid, path):
        if HAS_MLFLOW:
            self.c.log_artifact(rid, str(path))

    def end_run(self, rid):
        if HAS_MLFLOW:
            self.c.set_terminated(rid, "FINISHED")

    def _read(self, rid): return json.loads((self.dir / rid / "run.json").read_text())
    def _write(self, rid, d): (self.dir / rid / "run.json").write_text(json.dumps(d, indent=2))
