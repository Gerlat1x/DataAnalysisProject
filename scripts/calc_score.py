import pandas as pd

from models import create_model


def main():
    df = pd.read_parquet("../data/processed/panel.parquet")
    model = create_model({"type": "Momentum", "window": 5})
    scores = model.score(df)
    print(scores.sort_values(ascending=False).head())


if __name__ == "__main__":
    main()