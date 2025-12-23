import argparse
import os
import json
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import torch
import numpy as np
from transformers.models.t5.configuration_t5 import T5Config


def set_seed(seed: int = 42, deterministic: bool = True) -> int:
    """
    為 Python / NumPy / PyTorch 設定隨機種子，並盡量啟用決定性(可重現)行為。
    回傳 seed 方便你記錄。

    Args:
        seed: 你要用的隨機種子
        deterministic: 是否啟用 torch 的決定性演算法與相關設定
    """
    import os
    import random
    import numpy as np
    import torch

    # --- Python 與 NumPy ---
    os.environ["PYTHONHASHSEED"] = str(seed)  # 注意：理想情況是程式一開始就設定
    random.seed(seed)
    np.random.seed(seed)

    # --- PyTorch (CPU / CUDA) ---
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # 關掉 TF32，避免不同 GPU/Driver 導致的微差
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    except Exception:
        pass

    # cuDNN 相關：benchmark 在決定性情境下要關掉
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    if deterministic:
        # 啟用決定性演算法（某些 op 若不支援會丟錯）
        try:
            torch.use_deterministic_algorithms(True)
        except AttributeError:
            # 舊版 PyTorch 沒這個 API，就跳過
            pass

        # CuBLAS 在某些情況需要這個環境變數才能完全決定性
        # 兩種配置擇一，16:8 記憶體較省；若遇到錯誤可改成 4096:8
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")
        # 若你遇到 "CUBLAS_WORKSPACE_CONFIG not set" 或 deterministic 錯誤，可改用：
        # os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

def get_time() -> str:
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

def get_args(file_name: str):
    # First, parse just model_version and num_samples to determine which config to load
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--model_version", type=str, default="v1", choices=["v1", "v2"])
    pre_parser.add_argument("--num_samples", type=str, default="10K", choices=["10K", "100K"])
    pre_args, _ = pre_parser.parse_known_args()
    
    # Load default config based on model_version
    default_config = {}
    if pre_args.model_version == "v1":
        # For v1 (baseline): load from saved_models/{num_samples}_default_config.json
        default_config_path = Path("./saved_models") / f"{pre_args.num_samples}_default_config.json"
    else:  # v2
        # For v2: load from saved_models_v2/{num_samples}/config.json
        default_config_path = Path("./saved_models_v2") / pre_args.num_samples / "config.json"
    
    if default_config_path.exists():
        with open(default_config_path, 'r') as f:
            default_config = json.load(f)
    
    # Now create the full parser with config values as defaults
    parser = argparse.ArgumentParser()

    parser.add_argument("--random_seed", type=int, default=default_config.get("random_seed", 42))

    parser.add_argument("--save_path", type=str, default=default_config.get("save_path", "./saved_models"))
    parser.add_argument("--dataset_path", type=str, default=default_config.get("dataset_path", "./data"))

    # model config
    parser.add_argument(
        "--model_version",
        type=str,
        default=default_config.get("model_version", "v1"),
        choices=["v1", "v2"],
        help="Choose backbone: v1 uses model.py (default), v2 uses model_v2.py",
    )
    parser.add_argument("--d_ff", type=int, default=default_config.get("d_ff", 3072))
    parser.add_argument("--d_kv", type=int, default=default_config.get("d_kv", 64))
    parser.add_argument("--num_layers", type=int, default=default_config.get("num_layers", 12))
    parser.add_argument("--num_heads", type=int, default=default_config.get("num_heads", 12))
    parser.add_argument("--dropout_rate", type=float, default=default_config.get("dropout_rate", 0.1))

    parser.add_argument("--epochs", type=int, default=default_config.get("epochs", 500))
    parser.add_argument("--num_samples", type=str, default=default_config.get("num_samples", "10K"), choices=["10K", "100K"])

    if file_name == "train.py":
        parser.add_argument("--batch_size", type=int, default=default_config.get("batch_size", 500))
        parser.add_argument("--learning_rate", type=float, default=default_config.get("learning_rate", 5e-4))
        parser.add_argument("--warmup_ratio", type=float, default=default_config.get("warmup_ratio", 0.1))

    if file_name == "evaluate.py":
        parser.add_argument("--batch_size", type=int, default=default_config.get("batch_size", 100))
        parser.add_argument("--num_beams", type=int, default=default_config.get("num_beams", 20))
        parser.add_argument("--noise_factor", type=float, default=default_config.get("noise_factor", 0.0))

    # PQ
    parser.add_argument("--num_subspace", type=int, default=default_config.get("num_subspace", 4))
    parser.add_argument("--num_clusters", type=int, default=default_config.get("num_clusters", 128))

    args = parser.parse_args()
    
    # Load v2-specific parameters from config only when using v2
    if args.model_version == "v2" and default_config:
        v2_params = {
            "label_smoothing": default_config.get("label_smoothing", 0.1),
            "temperature": default_config.get("temperature", 1.0),
            "proj_dropout": default_config.get("proj_dropout", 0.1),
            "loss_type": default_config.get("loss_type", "label_smoothing"),
            "retrieval_loss_weight": default_config.get("retrieval_loss_weight", 0.0),
            "use_subspace_heads": default_config.get("use_subspace_heads", False)
        }
        for key, value in v2_params.items():
            setattr(args, key, value)
    
    return args

def save_model(model: torch.nn.Module, save_dir: str, name: str) -> None:
    os.makedirs(save_dir, exist_ok=True)
    torch.save(model.state_dict(), Path(save_dir)/f"{name}.pth")

def save_config(args, save_dir: str) -> None:
    os.makedirs(save_dir, exist_ok=True)
    with open(Path(save_dir)/"config.json", "w") as f:
        json.dump(vars(args), f, indent=4)

def load_model(args) -> torch.nn.Module:
    from dataset import Sift1mDataset
    # Select implementation based on user choice
    if getattr(args, "model_version", "v1") == "v2":
        from model_v2 import T5ForPretrain as T5ForPretrainImpl
    else:
        from model import T5ForPretrain as T5ForPretrainImpl

    config = T5Config(
        is_encoder_decoder=False,
        vocab_size=args.num_clusters,
        d_model=Sift1mDataset.VECTOR_DIM,
        d_ff=args.d_ff,
        d_kv=args.d_kv,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dropout_rate=args.dropout_rate,
    )
    model = T5ForPretrainImpl(config, args)
    return model

def plot_loss(losses: List[float], save_dir: str = None) -> None:
    plt.title("Learning Curve")
    x = np.arange(len(losses))
    plt.plot(x, losses)
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(Path(save_dir)/"loss.png")
    else:
        plt.show()
    plt.clf()

