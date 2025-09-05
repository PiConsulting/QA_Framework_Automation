# src/mlflow_integration.py
import os, json, mlflow
from typing import Dict, Any
def _is_enabled() -> bool:
   return os.getenv("MLFLOW_ENABLED", "0") == "1"
def start_run(run_name: str = None):
   if not _is_enabled():
       return None
   tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "").strip()
   experiment = os.getenv("MLFLOW_EXPERIMENT", "default").strip()
   if tracking_uri and tracking_uri.lower() != "local":
       mlflow.set_tracking_uri(tracking_uri)
   mlflow.set_experiment(experiment)
   return mlflow.start_run(run_name=run_name)
def log_params(params: Dict[str, Any]):
   if not _is_enabled():
       return
   # serializa dicts/listas sencillas como JSON
   for k, v in (params or {}).items():
       if isinstance(v, (dict, list)):
           mlflow.log_text(json.dumps(v, ensure_ascii=False, indent=2), f"params/{k}.json")
       else:
           mlflow.log_param(k, str(v))
def log_metrics(metrics: Dict[str, float]):
   if not _is_enabled():
       return
   # sólo numéricos
   for k, v in (metrics or {}).items():
       try:
           mlflow.log_metric(k, float(v))
       except Exception:
           pass
def log_artifact(path: str, artifact_path: str = None):
   if not _is_enabled():
       return
   if artifact_path:
       mlflow.log_artifact(path, artifact_path=artifact_path)
   else:
       mlflow.log_artifact(path)
def log_dict(obj: Dict[str, Any], path: str):
   if not _is_enabled():
       return
   mlflow.log_dict(obj, path)
def end_run(status: str = "FINISHED"):
   if not _is_enabled():
       return
   mlflow.end_run(status=status)