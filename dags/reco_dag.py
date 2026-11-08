"""Airflow DAG built from the SAME TASKS / DEPENDENCIES the Python runner uses.
Setup: install apache-airflow, point `dags_folder` at this directory (or copy it),
and make the repo importable (export PYTHONPATH=/path/to/recommendation-pipeline RECO_HOME=<same>).
Graph:  ingest >> validate >> prepare >> [transform, build_features] >> [train_popularity, train_hybrid] >> evaluate
"""
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from src.orchestration.run_pipeline import TASKS, DEPENDENCIES

with DAG("reco_pipeline", start_date=datetime(2025, 1, 1), schedule="@daily", catchup=False,
         default_args={"retries": 1, "retry_delay": timedelta(minutes=2)}, tags=["recommender"]) as dag:
    ops = {name: PythonOperator(task_id=name, python_callable=fn) for name, fn in TASKS.items()}
    for task, ups in DEPENDENCIES.items():
        for up in ups:
            ops[up] >> ops[task]
