"""Baixa o Brazilian E-Commerce Public Dataset (Olist) para `data/raw/`.

Os CSV brutos nao sao versionados neste repositorio (~128 MB). Rode este script
uma vez antes de executar os notebooks:

    pip install kagglehub
    python scripts/baixar_dados.py

Requer credenciais do Kaggle em `~/.kaggle/credentials.json` (kagglehub) ou
`~/.kaggle/kaggle.json` (pacote `kaggle`). Ambos sao gerados em
https://www.kaggle.com/settings > API > Create New Token.
"""

import shutil
import sys
from pathlib import Path

DATASET = "olistbr/brazilian-ecommerce"
RAIZ = Path(__file__).resolve().parent.parent
DESTINO = RAIZ / "data" / "raw"

# Os arquivos que os notebooks 01-03 esperam encontrar em data/raw/.
ESPERADOS = [
    "olist_customers_dataset.csv",
    "olist_geolocation_dataset.csv",
    "olist_order_items_dataset.csv",
    "olist_order_payments_dataset.csv",
    "olist_order_reviews_dataset.csv",
    "olist_orders_dataset.csv",
    "olist_products_dataset.csv",
    "olist_sellers_dataset.csv",
    "product_category_name_translation.csv",
]


def baixar() -> Path:
    """Retorna o diretorio de cache onde o dataset foi baixado."""
    try:
        import kagglehub
    except ImportError:
        pass
    else:
        return Path(kagglehub.dataset_download(DATASET))

    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError:
        sys.exit(
            "Nenhuma biblioteca do Kaggle encontrada.\n"
            "Instale uma delas:  pip install kagglehub   (ou)   pip install kaggle"
        )

    api = KaggleApi()
    api.authenticate()
    cache = RAIZ / ".kaggle_cache"
    cache.mkdir(exist_ok=True)
    api.dataset_download_files(DATASET, path=str(cache), unzip=True)
    return cache


def main() -> None:
    DESTINO.mkdir(parents=True, exist_ok=True)

    faltando = [n for n in ESPERADOS if not (DESTINO / n).exists()]
    if not faltando:
        print(f"Os {len(ESPERADOS)} arquivos ja estao em {DESTINO}. Nada a fazer.")
        return

    print(f"Baixando {DATASET} ...")
    origem = baixar()

    copiados = 0
    for nome in ESPERADOS:
        achado = next(origem.rglob(nome), None)
        if achado is None:
            print(f"  AVISO: {nome} nao veio no download")
            continue
        shutil.copy2(achado, DESTINO / nome)
        copiados += 1

    print(f"{copiados}/{len(ESPERADOS)} arquivos em {DESTINO}")
    if copiados < len(ESPERADOS):
        sys.exit(1)


if __name__ == "__main__":
    main()
