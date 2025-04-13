# Federated Learning with Non-IID Data (Label Skew)

This project explores the challenge of **non-IID label distribution** in federated learning using the MNIST dataset. The setup simulates label skew between clients and compares three versions of the FedAvg aggregation algorithm.

---

## Step 1: Create and Activate the Environment

```bash
conda env create -f fed_learning.yml
conda activate fedavg
```

---

## Step 2: Download the MNIST Dataset and Generate Non-IID Splits

```bash
python data_preparation.py
```

This script downloads the MNIST dataset and generates two non-IID client datasets:

- **Client 1:** 80% digits 0–4, 20% digits 5–9
- **Client 2:** 20% digits 0–4, 80% digits 5–9

The generated splits are saved in the `./splits/` directory.

---

## Step 3: Train the Model for Client 1

```bash
python client_training.py --client_id 1 --output_model client1_model.pth
```

---

## Step 4: Train the Model for Client 2

```bash
python client_training.py --client_id 2 --output_model client2_model.pth
```

---

## Step 5: Aggregate the Models using FedAvg Variants

```bash
python fedavg.py --model1 client1_model.pth --model2 client2_model.pth --output_model fedavg_model.pth
```

This script runs three variations of the FedAvg algorithm:

- `fedavg_model0.pth`: Standard FedAvg (averages all layers equally)
- `fedavg_model1.pth`: Modified FedAvg v1 (averages all layers except the classifier)
- `fedavg_model2.pth`: Modified FedAvg v2 (applies class-wise weighting in the classifier layer)

All aggregated models are saved in the `./models/` directory.

---

## Step 6: Evaluate the Models

```bash
python fedavg.py --evaluate --model1 client1_model.pth --model2 client2_model.pth --global_model fedavg_model.pth
```

This evaluates:

- Each client model on its **own** training split and the **other** client's split
- All three **aggregated models** on the **global MNIST test set**

Metrics reported:

- Per-class accuracy
- Overall accuracy
