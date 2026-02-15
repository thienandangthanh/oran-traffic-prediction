# O-RAN Traffic Prediction with Autoformer

[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.10.0-red.svg)](https://pytorch.org/)

End-to-end reproduction of transformer-based wireless traffic prediction for Open Radio Access Networks (O-RAN) using the Autoformer architecture.

## 🎯 Project Overview

This project reproduces the research from **Habib et al. (2024)** on transformer-based wireless traffic prediction in O-RAN environments. We implement traffic forecasting models to predict network load and throughput patterns, enabling proactive network optimization through xApps and rApps.

### Key Features

- ✅ **Complete Implementation**: End-to-end pipeline from data loading to model evaluation
- 📊 **Real-World Dataset**: Colosseum O-RAN COMMAG dataset with 5G network traces
- 🤖 **Multiple Models**: Autoformer, LSTM, and baseline models for comparison
- 📈 **Comprehensive Evaluation**: MSE, MAE, RMSE, residual analysis, and visualization
- 📓 **Jupyter Notebook**: Self-contained implementation with inline documentation
- 🔧 **Modular Design**: Reusable components for data processing, training, and evaluation

## 📚 Research Papers

1. **Habib et al. (2024)**: "Transformer-Based Wireless Traffic Prediction and Network Optimization in O-RAN"
2. **Wu et al.**: "Autoformer: Decomposition Transformers with Auto-Correlation for Long-Term Series Forecasting"
3. **Bonati et al. (2021)**: "Intelligence and Learning in O-RAN for Data-Driven NextG Cellular Networks"

## 📊 Dataset

### Colosseum O-RAN COMMAG Dataset

The dataset contains experimental data from a multi-cell, multi-slice 5G network emulated on Colosseum:

- **Infrastructure**: 4 Base Stations (BSs), 40 User Equipments (UEs)
- **Network Slicing**: 3 slices per BS (eMBB, MTC, URLLC)
- **Scheduling Policies**: Round-robin (RR), Waterfilling (WF), Proportionally fair (PF)
- **RF Scenarios**: 
  - `rome_static_close` - UEs within 20m of BSs
  - `rome_static_medium` - UEs within 50m of BSs
  - `rome_static_far` - UEs within 100m of BSs
  - `rome_slow_close` - Mobile UEs at 3 m/s
- **Training Configurations**: 18 configurations (tr0-tr17)
- **Metrics**: Timestamp, number of UEs, downlink/uplink bitrate

### Dataset Structure

```
colosseum-oran-commag-dataset/
├── slice_traffic/          # Traffic-based slicing
│   ├── rome_static_medium/
│   │   ├── tr0/           # Training configuration 0
│   │   │   ├── exp1/      # Experiment 1
│   │   │   │   ├── bs1/   # Base station 1
│   │   │   │   │   ├── bs1.csv          # Aggregate BS metrics
│   │   │   │   │   ├── ue1.csv          # Per-UE metrics
│   │   │   │   │   └── slices_bs1/      # Per-slice metrics
│   │   │   │   ├── bs2/
│   │   │   │   ├── bs3/
│   │   │   │   └── bs4/
│   │   │   ├── exp2/
│   │   │   └── ...
│   │   ├── tr1/
│   │   └── ...
│   ├── rome_static_close/
│   └── ...
└── slice_mixed/            # Mixed slicing
```

## 🏗️ Project Structure

```
traffic-prediction/
├── notebooks/
│   └── 01_traffic_prediction.ipynb    # Main experimental notebook
├── src/                                # Source modules (optional)
│   ├── data/
│   │   ├── dataset_loader.py          # CSV data loading
│   │   ├── preprocessor.py            # Preprocessing & filtering
│   │   └── aggregator.py              # Traffic aggregation
│   ├── models/
│   │   ├── autoformer_wrapper.py      # Autoformer implementation
│   │   └── baseline_models.py         # LSTM and baselines
│   └── utils/
│       ├── metrics.py                 # Evaluation metrics
│       └── visualization.py           # Plotting functions
├── colosseum-oran-commag-dataset/     # Dataset (git submodule)
├── pyproject.toml                     # uv configuration
└── README.md                          # This file
```

## 📄 License

This project follows the licensing of the Colosseum O-RAN COMMAG dataset and associated research papers. See individual paper licenses for details.

## 🙏 Acknowledgments

- **Habib et al.** for the original research paper
- **Wu et al.** for the Autoformer architecture
- **Bonati et al.** for the Colosseum dataset
- **HuggingFace** for Transformers library
- **Colosseum** wireless network emulator team

## 🔗 References

1. Habib, M. A., et al. (2024). "Transformer-Based Wireless Traffic Prediction and Network Optimization in O-RAN"
2. Wu, H., et al. "Autoformer: Decomposition Transformers with Auto-Correlation for Long-Term Series Forecasting"
3. Bonati, L., et al. (2021). "Intelligence and Learning in O-RAN for Data-Driven NextG Cellular Networks," IEEE Communications Magazine
4. Dataset: https://github.com/wineslab/colosseum-oran-commag-dataset
5. Autoformer: https://github.com/thuml/Autoformer
6. HuggingFace: https://huggingface.co/docs/transformers/model_doc/autoformer
