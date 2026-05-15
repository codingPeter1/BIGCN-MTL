# BIGCN-MTL

Multi-behavior recommendation, unlike traditional single-behavior recommendation focusing solely on the target behavior (e.g., purchase), exploits multiple types of user-item interactions (e.g., view, favorite, add-to-cart, purchase) to mitigate data sparsity and learn more comprehensive user preferences. However, real-world data is inherently transmitted in a streaming manner, and most existing multi-behavior recommendation models are trained on static offline datasets, which cannot efficiently handle continuously arriving data streams. Full retraining incurs high computational cost and long training time. In contrast, fine-tuning is efficient but prone to "catastrophic forgetting", resulting in performance degradation. Incremental recommendation techniques offer a potential solution, but the existing methods are mostly designed for cases with one single behavior, ignoring valuable information contained in heterogeneous behaviors. To fill this gap, we introduce a new and important problem, i.e., incremental multi-behavior recommendation (IMBR), requiring a model to handle streaming data while utilizing valuable heterogeneous behaviors. To this end, we propose a novel solution called Behavior-aware Incremental Graph Convolution Network with Multi-Task Learning (BIGCN-MTL) for IMBR. Specifically, our BIGCN-MTL contains two key modules: behavior-aware incremental graph convolution network (BIGCN), which employs incremental cascading graph convolution to preserve both historical neighborhood information and behavior cascading dependency, and historical interest distillation (HID), which maintains users' preference stability and prevents behavioral semantic forgetting via non-interest consistency constraint and preference contrastive distillation. Extensive experiments on four real-world datasets demonstrate that our BIGCN-MTL achieves superior recommendation performance and improves training efficiency compared to full retraining.

## Requirements

### Environment

- Python 3.7+
- CUDA-capable GPU (recommended for training)

### Dependencies

Install the required packages:

```bash
pip install torch torchvision torchaudio
pip install numpy scipy pandas
pip install loguru tensorboard tqdm
```

Or create a `requirements.txt` file with:

```
torch>=1.8.0
numpy>=1.19.0
scipy>=1.5.0
pandas>=1.1.0
loguru>=0.5.0
tensorboard>=2.4.0
tqdm>=4.50.0
```

Then install with:

```bash
pip install -r requirements.txt
```

## Project Structure

```
BIGCN-MTL/
├── main.py              # Main entry point
├── model.py             # BIGCN-MTL model definition
├── trainer.py           # Training logic
├── data_set.py          # Dataset loading and preprocessing
├── metrics.py           # Evaluation metrics
├── utils.py             # Utility functions and loss functions
├── BIGCN-MTL.sh         # Shell script for batch experiments
├── data/                # Data directory (should contain dataset folders)
│   ├── JD/              # JingDong dataset
│   ├── Tmall/           # Tmall dataset
│   ├── Rees46/          # Rees46 dataset
│   └── UB/              # UserBehavior dataset
├── check_point/         # Model checkpoints (auto-created)
├── embeddings_save/     # Saved embeddings (auto-created)
├── log/                 # Training logs (auto-created)
└── nohup_log/           # Background process logs (auto-created)
```

## Dataset Preparation

1. Create a `data/` directory in the project root
2. Organize your dataset in the following structure:

```
data/
├── JD/
│   ├── 1/               # Stage 1 data
│   ├── 2/               # Stage 2 data
│   └── ...
├── Tmall/
│   ├── 1/
│   ├── 2/
│   └── ...
└── ...
```

Each stage directory should contain:
- `count.txt`: User and item counts
- `click.txt`, `fav.txt`, `cart.txt`, `buy.txt`: Behavior interaction files
- `valid.txt`, `test.txt`: Validation and test splits

## Quick Start

### Single Run

Run with default parameters for JD dataset:

```bash
python main.py \
    --data_name JD_2 \
    --lr 0.003 \
    --reg_weight 0.01 \
    --his_weight 0.0001 \
    --kd_weight 0.01 \
    --tao 0.01 \
    --embedding_size 64 \
    --stage 1 \
    --device cuda:0
```

### Batch Experiments

Use the provided shell script to run experiments with multiple hyperparameter configurations:

```bash
bash BIGCN-MTL.sh
```

Edit the script to customize:
- `dataset`: Dataset name (JD_2, Tmall_2, Rees46_2, UB_2, etc.)
- `lr`: Learning rate
- `reg_weight`: Regularization weight
- `his_weight`: History loss weight
- `kd_weight`: Knowledge distillation weight
- `tao`: Temperature parameter
- `emb_size`: Embedding size
- `gpu`: GPU device (e.g., 'cuda:0')
- `stage`: Training stage

## Key Parameters

- `--embedding_size`: Dimension of user/item embeddings (default: 64)
- `--lr`: Learning rate (default: 0.01)
- `--reg_weight`: L2 regularization weight (default: 1e-3)
- `--his_weight`: Weight for historical contrastive loss (default: 0.0)
- `--kd_weight`: Weight for knowledge distillation loss (default: 0.01)
- `--tao`: Temperature for contrastive learning (default: 0.1)
- `--layers`: Number of GCN layers (default: 1)
- `--node_dropout`: Node dropout rate (default: 0.0)
- `--message_dropout`: Message dropout rate (default: 0.0)
- `--batch_size`: Training batch size (default: 1024)
- `--epochs`: Maximum number of epochs (default: 250)
- `--device`: Device to use ('cpu' or 'cuda:X')
- `--stage`: Current training stage (for incremental learning)

## Supported Datasets

The model supports the following datasets with their behavior types:

- **JD (JingDong)**: click, fav, cart, buy
- **Tmall**: click, fav, buy
- **Rees46**: view, cart, buy
- **UB (UserBehavior)**: pv, fav, cart, buy

## Training Process

The training follows an incremental learning approach:

1. **Stage 1**: Train on initial data
2. **Stage 2+**: Load embeddings from previous stage and continue training

The model automatically:
- Saves best model checkpoints
- Logs training metrics to TensorBoard
- Early stops if validation performance doesn't improve for 50 epochs
- Saves layer embeddings for next stage

## Monitoring

### TensorBoard

View training progress:

```bash
tensorboard --logdir=./log
```

### Log Files

- Training logs: `./log/{model_name}/{timestamp}.log`
- Background process logs: `./nohup_log/{dataset_name}/`

## Output

After training, the model generates:

1. **Model checkpoints**: Saved in `./check_point/`
2. **Layer embeddings**: Saved in `./embeddings_save/{dataset_name}/`
3. **Training logs**: Saved in `./log/{model_name}/`
4. **TensorBoard logs**: Saved in `./log/train/` and `./log/test/`

## Evaluation Metrics

The model evaluates performance using:

- **Recall@K**
- **NDCG@K**

Results are reported on validation and test sets for the target behavior (last behavior in the sequence).

## Citation

If you use this code in your research, please cite the original paper.

## License

This project is for research purposes only.