"""Draws reports/architecture.png (run: python -m src.orchestration.make_diagram)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from configs import settings as C

fig, ax = plt.subplots(figsize=(13, 6.8)); ax.axis("off"); ax.set_xlim(0, 13); ax.set_ylim(0, 6.8)

def box(x, y, w, h, title, sub, fc):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.03", fc=fc, ec="#33415c", lw=1.2))
    ax.text(x + w / 2, y + h * (0.66 if sub else 0.5), title, ha="center", va="center", fontsize=9.5, fontweight="bold")
    if sub: ax.text(x + w / 2, y + h * 0.28, sub, ha="center", va="center", fontsize=7.4, color="#222")

def arrow(x1, y1, x2, y2, c="#33415c"):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="->", color=c, lw=1.3))

main, ml, store, err = "#dbeafe", "#dcfce7", "#fef3c7", "#fee2e2"
row1 = [("1 Ingestion", "synthetic users / items /\ninteractions (CSV)"), ("2 Validation", "pandas rules + quality gate"),
        ("3 Preparation", "clean, dedupe, impute,\ntime-based train/test split"),
        ("4 Transformation", "user activity, item popularity,\nuser-item matrix")]
for k, (t, sub) in enumerate(row1):
    box(0.3 + k * 3.2, 4.5, 2.7, 1.1, t, sub, main)
    if k: arrow(0.3 + k * 3.2 - 0.5, 5.05, 0.3 + k * 3.2, 5.05)
box(3.5, 6.0, 2.7, 0.5, "data/errors/rejected_records.csv", "", err)
arrow(4.85, 5.6, 4.85, 6.0, "#b91c1c")
box(6.7, 2.9, 2.7, 1.0, "5 Feature management", "feature store (user / item /\nuser-category) + registry", main)
arrow(8.05, 4.5, 8.05, 3.9)                                  # preparation -> features
box(9.9, 2.95, 2.8, 0.75, "6a Popularity model", "baseline", ml)
box(9.9, 1.85, 2.8, 0.75, "6b Hybrid MF model", "SVD + popularity + category", ml)
arrow(11.3, 4.5, 11.3, 3.7)                                  # transformation -> training (matrix)
arrow(9.4, 3.4, 9.9, 3.35); arrow(9.4, 3.1, 9.9, 2.3)        # features -> both models
box(6.7, 1.5, 2.7, 1.0, "7 Evaluation", "P@10, R@10, NDCG@10, coverage,\ndiversity, RMSE, MAE", main)
arrow(9.9, 2.1, 9.4, 2.0); arrow(9.9, 3.0, 9.4, 2.35)
box(3.5, 1.5, 2.7, 1.0, "Reports", "model comparison, validation\nreport, feature catalog", store)
arrow(6.7, 2.0, 6.2, 2.0)
box(9.9, 0.3, 2.8, 0.8, "MLflow tracking", "params, metrics, artifacts", ml)
arrow(11.3, 1.85, 11.3, 1.1)
box(6.7, 0.3, 2.7, 0.8, "models/*.pkl", "pickled recommenders", store)
arrow(10.2, 1.85, 9.4, 1.0)
ax.text(0.3, 1.9, "Orchestration\nsrc/orchestration/run_pipeline.py\n(topological order, retries, logs)\n"
        "dags/reco_dag.py (Airflow)\nTasks exchange data only via files,\nso each task can re-run alone.",
        fontsize=8.2, va="center", bbox=dict(fc="#f1f5f9", ec="#94a3b8"))
ax.set_title("Recommendation pipeline architecture", fontsize=12, fontweight="bold")
plt.tight_layout(); plt.savefig(C.REP / "architecture.png", dpi=170); print("saved")
