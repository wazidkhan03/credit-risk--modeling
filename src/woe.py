"""Weight-of-Evidence encoder with Information Value (numeric + categorical)."""
import numpy as np
import pandas as pd

from .config import CAT_COLS


class WOEBinner:
    def __init__(self, n_bins: int = 8, eps: float = 0.5, min_cat_count: int = 50):
        self.n_bins, self.eps, self.min_cat_count = n_bins, eps, min_cat_count
        self.kind, self.edges, self.woe, self.tables = {}, {}, {}, {}

    # ---- helpers -----------------------------------------------------
    def _woe_iv(self, g):
        bad = g["sum"] + self.eps
        good = (g["count"] - g["sum"]) + self.eps
        woe = np.log((good / good.sum()) / (bad / bad.sum()))
        iv = (good / good.sum() - bad / bad.sum()) * woe
        return woe, iv

    def _assign(self, x: pd.Series, col: str):
        if self.kind[col] == "cat":
            return x.astype(str)
        return pd.cut(x, self.edges[col], labels=False, include_lowest=True)

    # ---- API ---------------------------------------------------------
    def fit(self, X: pd.DataFrame, y: pd.Series):
        for col in X.columns:
            x = X[col]
            if col in CAT_COLS:
                self.kind[col] = "cat"
                b = x.astype(str)
                labels = None
            else:
                self.kind[col] = "num"
                if x.nunique() <= self.n_bins:
                    u = np.sort(x.unique())
                    e = np.r_[-np.inf, (u[:-1] + u[1:]) / 2, np.inf]
                else:
                    q = np.unique(np.quantile(x, np.linspace(0, 1, self.n_bins + 1)[1:-1]))
                    e = np.r_[-np.inf, q, np.inf]
                self.edges[col] = e
                b = pd.cut(x, e, labels=False, include_lowest=True)
                labels = {i: f"({e[i]:.4g}, {e[i + 1]:.4g}]" for i in range(len(e) - 1)}

            g = pd.DataFrame({"b": b, "y": y.values}).groupby("b")["y"].agg(["sum", "count"])
            woe, iv = self._woe_iv(g)
            if self.kind[col] == "cat":          # rare levels -> neutral WOE
                rare = g["count"] < self.min_cat_count
                woe[rare], iv[rare] = 0.0, 0.0
            self.woe[col] = woe.to_dict()
            self.tables[col] = pd.DataFrame({
                "bin": g.index.astype(str),
                "label": [labels.get(i, str(i)) if labels else str(i) for i in g.index],
                "count": g["count"].values,
                "bad_rate": (g["sum"] / g["count"]).values,
                "woe": woe.values, "iv": iv.values})
        return self

    def transform(self, X: pd.DataFrame, cols=None) -> pd.DataFrame:
        cols = cols or list(self.kind)
        out = pd.DataFrame(index=X.index)
        for c in cols:
            b = self._assign(X[c], c)
            if self.kind[c] == "num":
                b = b.astype("Int64")
                mapper = {int(k): v for k, v in self.woe[c].items()}
                out[c] = b.map(mapper).astype(float)
            else:
                out[c] = b.map(self.woe[c]).astype(float)
            out[c] = out[c].fillna(0.0)          # unseen level -> neutral
        return out

    def iv(self) -> pd.Series:
        return pd.Series({c: t["iv"].sum() for c, t in self.tables.items()}
                         ).sort_values(ascending=False)
