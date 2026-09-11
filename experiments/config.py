# -*- coding = utf-8 -*-
# @File : config.py
"""Config declarativa de experimento. Cada main_experiments_*.py monta um
ExperimentConfig e chama experiments.runner.run_experiment(cfg) — em vez de
duplicar o loop de N_RUNS, dataset, treino e avaliação em cada arquivo.
"""
from dataclasses import dataclass, field
from typing import Callable, List, Optional

from utils.labels import CATEGORIES


@dataclass
class WarmStart:
    """Carrega pesos de um checkpoint anterior em submódulos específicos do
    modelo (ex: ecg_encoder treinado num experimento anterior), substituindo
    os blocos manuais de torch.load/load_state_dict comentados nos scripts
    antigos.
    """

    checkpoint_path: str
    submodules: List[str]
    strict: bool = True

    @classmethod
    def best_of(cls, save_dir, ckpt_name, n_runs, submodules, strict=True):
        """Warm-start a partir do melhor checkpoint (maior 'monitor_val')
        entre as N runs de um experimento anterior — em vez de uma run
        escolhida arbitrariamente. Todas as runs do experimento atual
        partem desse mesmo checkpoint fixo.
        """
        from utils.utils import find_best_checkpoint

        checkpoint_path = find_best_checkpoint(save_dir, ckpt_name, n_runs)
        return cls(checkpoint_path=checkpoint_path, submodules=submodules, strict=strict)


@dataclass
class ExperimentConfig:
    name: str
    data_dir: str

    # split (str) -> ecg_source / text_source (ver utils/ecg_sources.py, utils/text_sources.py)
    ecg_source_fn: Callable
    text_source_fn: Callable

    model_factory: Callable  # () -> nn.Module (pesos novos a cada run)
    optimizer_factory: Callable  # (model) -> torch.optim.Optimizer
    forward_fn: Callable  # ver engine/forward_adapters.py

    loss_fn_factory: Callable = None  # () -> nn.Module; default BCEWithLogitsLoss
    warm_start: Optional[WarmStart] = None

    sampling_rate: int = 500
    categories: List[str] = field(default_factory=lambda: list(CATEGORIES))
    threshold: float = 0.5

    epochs: int = 1000
    patience: int = 20
    n_runs: int = 5
    base_seed: int = 42
    batch_size: int = 16
    monitor: str = "val_auroc"
    mode: str = "max"

    save_dir: str = "./checkpoints/experiment/"
    ckpt_name: str = "checkpoint.pt"
    results_dir: str = "./results/experiment/"

    device: Optional[object] = None

    def __post_init__(self):
        if self.loss_fn_factory is None:
            import torch.nn as nn

            self.loss_fn_factory = nn.BCEWithLogitsLoss
