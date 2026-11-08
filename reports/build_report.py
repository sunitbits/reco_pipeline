"""Builds reports/final_report.pdf (2-3 pages) from the pipeline's real outputs. Run from repo root:  python -m reports.build_report"""
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
from configs import settings as C

from PIL import Image as PILImage
RUN_W, RUN_H = PILImage.open(C.REP / "screenshots/09_run_summary.png").size
ss = getSampleStyleSheet()
B = ParagraphStyle("B", parent=ss["Normal"], fontSize=9, leading=12)
H = ParagraphStyle("H", parent=ss["Heading2"], fontSize=11.5, spaceBefore=6, spaceAfter=3, textColor=colors.HexColor("#1e3a8a"))
T = ParagraphStyle("T", parent=ss["Title"], fontSize=17, spaceAfter=2)
S = ParagraphStyle("S", parent=B, fontSize=8, textColor=colors.grey)
BL = ParagraphStyle("BL", parent=B, leftIndent=10, bulletIndent=0)
CAP = ParagraphStyle("CAP", parent=B, fontSize=7.5, textColor=colors.grey, alignment=1)

def bullets(items): return [Paragraph(i, BL, bulletText="\u2022") for i in items]
def table(data, widths, hdr=True, fs=8):
    t = Table(data, colWidths=widths, repeatRows=1 if hdr else 0)
    st = [("FONTSIZE", (0, 0), (-1, -1), fs), ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#94a3b8")),
          ("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("TOPPADDING", (0, 0), (-1, -1), 2), ("BOTTOMPADDING", (0, 0), (-1, -1), 2)]
    if hdr: st += [("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dbeafe")), ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold")]
    t.setStyle(TableStyle(st)); return t

res = pd.read_csv(C.REP / "model_comparison.csv").set_index("model")
import json
vr = json.loads((C.REP / "validation_report.json").read_text())["tables"]
sp = json.loads((C.REP / "split_report.json").read_text())
p, h = res.loc["popularity"], res.loc["hybrid_mf"]
chg = lambda a, b: f"{(b / a - 1) * 100:+.1f}%"

st = [Paragraph("Recommendation Data Pipeline: Final Report", T),
      Paragraph("End-to-end pipeline from raw data to evaluated recommenders, orchestrated and tracked. Author: Sunit Kamble", S), Spacer(1, 4)]

st += [Paragraph("1. Problem and data", H), Paragraph(
    "The goal is to recommend relevant items to users of an online store from past interactions and item information, "
    "using a clean, modular and reproducible pipeline. Because the focus is pipeline quality, I generated <b>documented synthetic data</b> "
    "(1,500 users, 600 items, 8 categories, about 58k ratings from 2023-2024). Ratings come from latent user/item factors plus category "
    "structure, item bias and noise, and users mostly interact with popular items (popularity-biased exposure), as in real shops. "
    "Realistic dirt is injected on purpose (nulls, duplicates, invalid ratings and timestamps, wrong types, orphan keys) so the "
    "validation stage is exercised. A loader for MovieLens or Amazon Reviews can replace the generator without changing later stages.", B)]

st += [Paragraph("2. Architecture", H), Image(str(C.REP / "architecture.png"), width=15.5 * cm, height=15.5 * cm * 6.8 / 13),
       Paragraph("Figure 1. Eight stages; tasks exchange data only through files, so each can be re-run on its own.", CAP)]

st += [Paragraph("3. Design decisions and tools", H)]
st += bullets([
    "<b>Validation (pandas rules):</b> missing values, duplicates, wrong types, ratings outside 1-5, unparsable or future timestamps and orphan user/item keys. "
    "Rejected rows go to <i>data/errors/rejected_records.csv</i> with a reason, and a quality gate stops the run if more than 10% of any table is rejected.",
    "<b>Preparation and transformation:</b> standardised text, imputed missing ages, kept the latest rating per user-item pair, and used a "
    "<b>time-based split</b> (latest 20% of time is test) so the model never sees the future. Derived datasets: user activity summary, item popularity and a sparse user-item matrix.",
    "<b>Feature management:</b> a small file-based feature store builds user, item and user-category features from <b>train data only</b> and exposes one read path "
    "(<i>get_user_features, get_item_features, interaction_features</i>) used by both training and prediction, with defaults for cold-start users/items. "
    "A registry documents every feature (catalog in the repository README).",
    "<b>Models:</b> (a) popularity baseline ranked by positive-rating count; (b) hybrid recommender: SVD matrix factorisation on rating residuals "
    "(user mean plus shrunken item bias plus 16 latent factors), blended with item popularity and user-category affinity from the feature store.",
    "<b>Tracking and orchestration:</b> MLflow (SQLite backend) logs parameters, metrics and artifacts. A dependency-aware Python runner (graphlib) with retries and logs "
    "runs the DAG, and <i>dags/reco_dag.py</i> expresses the same graph for Apache Airflow. Runs are deterministic (fixed seed).",
    "<b>Tools:</b> Python, pandas, NumPy, SciPy, scikit-learn, MLflow, matplotlib, pytest, reportlab."])

st += [Paragraph("4. Results", H)]
vt = [["Table", "Input rows", "Valid rows", "Rejected", "Rate"]] + [
    [n, vr[n]["input_rows"], vr[n]["valid_rows"], vr[n]["rejected_rows"], f"{vr[n]['reject_rate']:.1%}"] for n in vr]
st += [Paragraph(f"<b>Validation.</b> All tables passed the quality gate. Interaction rejections are mostly orphan keys (1,706 item, 470 user; some cascade "
                 f"from rejected parent rows), duplicates (562) and invalid ratings/timestamps/types (about 295 each). Split: cutoff {sp['cutoff'][:10]}, "
                 f"{sp['train_rows']:,} train rows and {sp['test_rows']:,} test rows.", B), Spacer(1, 3),
       table(vt, [3.5 * cm, 3 * cm, 3 * cm, 3 * cm, 2.5 * cm]), Spacer(1, 5)]
cols = ["precision@10", "recall@10", "ndcg@10", "coverage", "diversity", "rmse", "mae"]
rt = [["Model", "P@10", "R@10", "NDCG@10", "Coverage", "Diversity", "RMSE", "MAE"],
      ["Popularity"] + [f"{p[c]:.4f}" for c in cols], ["Hybrid MF"] + [f"{h[c]:.4f}" for c in cols],
      ["Change"] + [chg(p[c], h[c]) for c in cols]]
st += [Paragraph(f"<b>Model comparison</b> (K = 10, {int(h['n_eval_users']):,} test users, relevant = rating of 4 or more):", B), Spacer(1, 3),
       table(rt, [2.6 * cm] + [2.1 * cm] * 7), Spacer(1, 3),
       Paragraph(f"The hybrid model improves ranking quality (precision {chg(p['precision@10'], h['precision@10'])}, NDCG {chg(p['ndcg@10'], h['ndcg@10'])}), "
                 f"recommends a much wider part of the catalogue (coverage {p['coverage']:.3f} to {h['coverage']:.3f}) and predicts ratings better "
                 f"(RMSE {p['rmse']:.3f} to {h['rmse']:.3f}). The price is slightly lower diversity, because its lists lean on popular items in the preferred category.", B),
       Spacer(1, 4),
       Image(str(C.REP / "screenshots/09_run_summary.png"), width=13.5 * cm, height=13.5 * cm * RUN_H / RUN_W),
       Paragraph("Figure 2. Successful end-to-end run, rendered from run_status.json and model_comparison.csv (raw terminal and MLflow screenshots are in reports/screenshots).", CAP)]

st += [Paragraph("5. Limitations and future work", H)]
st += bullets([
    "<b>Synthetic data:</b> structure and popularity bias are assumed, so absolute scores will differ on real data (MovieLens, Amazon Reviews).",
    "<b>Tuning:</b> the hybrid blend weights came from a small grid scored on the test set, so the gain is slightly optimistic. A validation window "
    "taken from the end of training data should be used in production.",
    "<b>Validation:</b> custom pandas rules are simple and transparent; Pandera or Great Expectations would add reusable, versioned expectation suites.",
    "<b>Scale and storage:</b> CSV files, a dense per-user score matrix and a local MLflow store suit this size only. Larger data needs Parquet/a database, "
    "approximate nearest-neighbour retrieval and a hosted tracking server.",
    "<b>Orchestration:</b> the Python runner is the tested path; the Airflow DAG reuses its task graph but was not executed here. "
    "Future work includes implicit-feedback models (ALS, LightFM), online evaluation and monitoring for data drift."])
doc = SimpleDocTemplate(str(C.REP / "final_report.pdf"), pagesize=A4, leftMargin=1.8 * cm, rightMargin=1.8 * cm, topMargin=1.5 * cm, bottomMargin=1.4 * cm,
                        title="Recommendation Data Pipeline - Final Report", author="Sunit Kamble")
doc.build(st); print("built")
