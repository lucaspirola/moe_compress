"""Router-KD — unified Stage 2.5 + Stage 5 KD router fine-tuning (plugin architecture)."""
from .orchestrator import run
from .stage import make_router_kd_stage

__all__ = ["run", "make_router_kd_stage"]
