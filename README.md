# CLV Segmentation Engine

**How can we predict customer lifetime value and segment customers to optimize marketing spend?**

ML pipeline that combines RFM analysis, K-Means clustering, and XGBoost regression to segment 93K+ customers and predict their monetary value — built on real Olist e-commerce data.

## Key Results

| Metric | Value |
|--------|-------|
| Dataset | 93,357 customers (Olist) |
| Segmentation | K-Means clustering on RFM |
| CLV Model | XGBoost Regressor |
| MAE | R$ 88.32 |
| RMSE | R$ 167.70 |
| Deploy | FastAPI + Docker |

## Stack

`Python` · `XGBoost` · `Scikit-Learn` · `K-Means` · `Pandas` · `FastAPI` · `Docker`

## Pipeline

1. **RFM Analysis** — Compute Recency, Frequency, and Monetary metrics per customer
2. **K-Means Segmentation** — Cluster customers into behavioral groups
3. **CLV Prediction** — XGBoost model predicts monetary value using RFM features + cluster
4. **API** — FastAPI endpoint for real-time CLV scoring

## Data

This project uses the [Brazilian E-Commerce Public Dataset by Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) (Kaggle). The raw CSV files are **not versioned** — they total ~128 MB and are reproducible from the source.

```bash
pip install kagglehub
python scripts/baixar_dados.py
```

This populates `data/raw/` with the nine source files. The notebooks write their intermediate outputs to `data/processed/`. Both directories are gitignored.

## Project Structure

```
├── notebooks/
│   ├── 01_eda_rfm.ipynb             # Exploratory RFM analysis
│   ├── 02_segmentacao_kmeans.ipynb   # K-Means clustering
│   └── 03_clv_xgboost.ipynb         # XGBoost CLV prediction
├── data/
│   ├── raw/                          # Olist dataset (9 CSV files)
│   └── processed/                    # RFM + cluster outputs
├── models/
│   └── xgboost_clv_model.pkl        # Trained model artifact
├── scripts/
│   └── baixar_dados.py               # Downloads the Olist dataset
├── src/
│   └── api.py                        # FastAPI inference endpoint
├── Dockerfile
└── requirements.txt
```

## How to Run

```bash
git clone https://github.com/guilhermehrsilva/clv-segmentation-engine.git
cd clv-segmentation-engine
pip install -r requirements.txt
python scripts/baixar_dados.py
```

Run notebooks in order (`01` → `02` → `03`), then start the API:

```bash
uvicorn src.api:app --reload
```

### With Docker

```bash
docker build -t clv-engine .
docker run -p 8000:8000 clv-engine
```

## License

MIT — see [LICENSE](LICENSE).
