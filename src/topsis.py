import numpy as np
import pandas as pd

def topsis():
    df = pd.read_csv("results/model_comparison.csv")

    criteria = df[["MSE", "MAE", "R2"]].values
    norm = criteria / np.sqrt((criteria**2).sum(axis=0))

    weights = np.array([0.33, 0.33, 0.34])
    weighted = norm * weights

    ideal_best = [weighted[:,0].min(), weighted[:,1].min(), weighted[:,2].max()]
    ideal_worst = [weighted[:,0].max(), weighted[:,1].max(), weighted[:,2].min()]

    dist_best = np.linalg.norm(weighted - ideal_best, axis=1)
    dist_worst = np.linalg.norm(weighted - ideal_worst, axis=1)

    df["TOPSIS Score"] = dist_worst / (dist_best + dist_worst)
    df["Rank"] = df["TOPSIS Score"].rank(ascending=False)

    df.to_csv("results/topsis_ranking.csv", index=False)
    return df

if __name__ == "__main__":
    topsis()
