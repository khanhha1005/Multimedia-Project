# Vector to Vector ID

```
📦 Project Root
├── data/                      # Dataset storage directory
├── saved_models/              # Trained model weights
│
├── .gitignore                 # Git ignore configuration
├── dataset.py                 # Dataset loading and preprocessing script
├── evaluate.py                # Evaluation script
├── model.py                   # Model definition
├── model_v2.py                # Proposed Model definition
├── README.md                  # Project description and instructions
├── requirements.txt           # Python dependencies
├── train.py                   # Model training script
├── trie.py                    # Prefix tree implementation
└── utils.py                   # Utility functions
```

## Setup
* check you cuda version via `nvidia-smi`, below command is 2.8.0+cu129.
* python version is 3.11.13.
```
pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu129
pip install -r requirements.txt
```

## Run Training
* Please first reproduce the recall with the default config (at `./saved_models`).
    * For 10K, R@1: 0.387, R@10: 0.741, R@20: 0.808
    * For 100K, R@1: 0.443, R@10: 0.682, R@20: 0.724
* To change the configuration, edit the `get_args()` function in `utils.py`.
### Baseline (model.py)
- Train  
  ```
  python train.py --num_samples 10K
  python train.py --num_samples 100K
  ```
- Evaluate (set `--noise_factor` as needed)  
  ```
  python evaluate.py --num_samples 10K --noise_factor 0.0
  python evaluate.py --num_samples 10K --noise_factor 0.06
  python evaluate.py --num_samples 100K --noise_factor 0.0
  python evaluate.py --num_samples 100K --noise_factor 0.06
  ```
### Proposed (model_v2.py)
- Train  
  - `python train.py --model_version v2 --num_samples 10K`
  - `python train.py --model_version v2 --num_samples 100K`
- Evaluate  
  - `python evaluate.py --model_version v2 --num_samples 10K --noise_factor 0.0`
  - `python evaluate.py --model_version v2 --num_samples 10K --noise_factor 0.06`
  - `python evaluate.py --model_version v2 --num_samples 100K --noise_factor 0.0`
  - `python evaluate.py --model_version v2 --num_samples 100K --noise_factor 0.06`