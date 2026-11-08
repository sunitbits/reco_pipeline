"""Renders reports/screenshots/09_run_summary.png from the real run outputs
(reports/run_status.json + reports/model_comparison.csv). It is a readable summary of the log,
NOT a screen capture: replace with a real terminal/Airflow screenshot if the course requires one."""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from configs import settings as C
from src.orchestration.run_pipeline import DEPENDENCIES

status = json.loads((C.REP / "run_status.json").read_text())
order = list(DEPENDENCIES)  # topological-looking declaration order
res = pd.read_csv(C.REP / "model_comparison.csv").set_index("model")
lines = ["$ python -m src.orchestration.run_pipeline", ""]
for t in order:
    s = status.get(t, {})
    lines.append(f"[{s.get('status', '?'):>2}] {t:<18} {s.get('seconds', 0):>5.2f}s   <- {', '.join(DEPENDENCIES[t]) or '-'}")
lines += ["", "model        P@10    R@10   NDCG@10  coverage  diversity   RMSE    MAE"]
for m, r in res.iterrows():
    lines.append(f"{m:<12} {r['precision@10']:.4f}  {r['recall@10']:.4f}  {r['ndcg@10']:.4f}   {r['coverage']:.4f}    {r['diversity']:.4f}   {r['rmse']:.4f}  {r['mae']:.4f}")
ok = all(v["status"] == "ok" for v in status.values())
lines += ["", f"pipeline finished: {'ALL 8 TASKS OK' if ok else 'FAILED'}"]
fig = plt.figure(figsize=(9.5, 0.27 * len(lines) + 0.5), facecolor="#0f172a")
fig.text(0.02, 0.97, "\n".join(lines), family="monospace", fontsize=9.2, color="#e2e8f0", va="top", linespacing=1.35)
fig.savefig(C.REP / "screenshots/09_run_summary.png", dpi=170, facecolor=fig.get_facecolor())
print("saved")
