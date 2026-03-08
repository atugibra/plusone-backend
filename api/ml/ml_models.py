"""
ML Models for PlusOne Prediction Engine
============================================
EnsemblePredictor: XGBoost + CalibratedClassifier(RandomForest)
soft-vote ensemble producing calibrated probabilities for
Home Win (0) / Draw (1) / Away Win (2).
"""

import os
import numpy as np
import joblib

from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier


LABELS = {0: "Home Win", 1: "Draw", 2: "Away Win"}
MODEL_PATH = os.path.join(os.path.dirname(__file__), "saved_model.joblib")


class EnsemblePredictor:
    """
    Soft-vote ensemble of calibrated XGBoost + RandomForest classifiers.
    Outputs calibrated probabilities that sum to 1.0.
    """

    def __init__(self):
        self.feature_names_ = None
        self.scaler_ = StandardScaler()
        self.is_trained = False
        self.train_accuracy = None
        self.cv_accuracy = None
        self.n_samples = 0
        self.feature_importances_ = {}

        # XGBoost — handles imbalanced classes well, fast, interpretable
        xgb = XGBClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="mlogloss",
            random_state=42,
            n_jobs=-1,
        )

        # RandomForest — diverse learner, good at irregular boundaries
        rf = RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            min_samples_leaf=5,
            class_weight="balanced",   # handles draw class imbalance
            random_state=42,
            n_jobs=-1,
        )

        # Calibrate RF to produce proper probabilities
        rf_cal = CalibratedClassifierCV(rf, cv=3, method="isotonic")

        # Soft vote: weighted toward XGBoost (generally more accurate on tabular data)
        self.model = VotingClassifier(
            estimators=[("xgb", xgb), ("rf", rf_cal)],
            voting="soft",
            weights=[2, 1],      # XGB gets 2x weight
        )

    def train(self, X, y, feature_names=None, cv_folds=5):
        """
        Train ensemble on (X, y).
        X: list of feature vectors
        y: list of labels (0=Home Win, 1=Draw, 2=Away Win)
        feature_names: optional list of feature name strings
        """
        X = np.array(X, dtype=float)
        y = np.array(y, dtype=int)

        # Replace inf/nan with column means
        col_means = np.nanmean(np.where(np.isfinite(X), X, np.nan), axis=0)
        col_means = np.nan_to_num(col_means, nan=0.0)
        inds = np.where(~np.isfinite(X))
        X[inds] = np.take(col_means, inds[1])

        self.feature_names_ = feature_names or [f"f{i}" for i in range(X.shape[1])]
        self.n_samples = len(y)

        # Scale features (helps RF, doesn't hurt XGB)
        X_scaled = self.scaler_.fit_transform(X)

        # Cross-validation accuracy
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        cv_scores = cross_val_score(self.model, X_scaled, y, cv=skf,
                                    scoring="accuracy", n_jobs=-1)
        self.cv_accuracy = float(np.mean(cv_scores))

        # Final fit on all data
        self.model.fit(X_scaled, y)
        y_pred = self.model.predict(X_scaled)
        self.train_accuracy = float(accuracy_score(y, y_pred))
        self.is_trained = True

        # Feature importances from XGBoost sub-model
        try:
            xgb_model = self.model.estimators_[0]
            imps = xgb_model.feature_importances_
            self.feature_importances_ = {
                name: float(imp)
                for name, imp in zip(self.feature_names_, imps)
            }
        except Exception:
            self.feature_importances_ = {}

        return self

    def predict_proba(self, x):
        """
        Predict calibrated probabilities for a single feature vector x.
        Returns dict: {home_win, draw, away_win, predicted_outcome, confidence,
                       confidence_score, label_int}
        """
        if not self.is_trained:
            raise RuntimeError("Model not trained. Call train() or load() first.")

        X = np.array(x, dtype=float).reshape(1, -1)

        # Fix non-finite values
        col_means = np.nan_to_num(
            np.nanmean(np.where(np.isfinite(X), X, np.nan), axis=0), nan=0.0
        )
        inds = np.where(~np.isfinite(X))
        X[inds] = np.take(col_means, inds[1])

        X_scaled = self.scaler_.transform(X)
        proba = self.model.predict_proba(X_scaled)[0]

        # Map to named class probabilities
        classes = self.model.classes_
        prob_map = {int(c): float(p) for c, p in zip(classes, proba)}
        hw = prob_map.get(0, 0.33)
        dr = prob_map.get(1, 0.33)
        aw = prob_map.get(2, 0.34)

        # Normalize to sum exactly to 1
        total = hw + dr + aw
        hw, dr, aw = hw / total, dr / total, aw / total

        predicted_int = int(np.argmax([hw, dr, aw]))
        predicted_label = LABELS[predicted_int]
        confidence_score = max(hw, dr, aw)

        if confidence_score >= 0.60:
            confidence = "High"
        elif confidence_score >= 0.45:
            confidence = "Medium"
        else:
            confidence = "Low"

        return {
            "home_win":         round(hw, 4),
            "draw":             round(dr, 4),
            "away_win":         round(aw, 4),
            "predicted_outcome": predicted_label,
            "label_int":        predicted_int,
            "confidence":       confidence,
            "confidence_score": round(confidence_score, 4),
        }

    def get_top_features(self, x, n=6):
        """
        Return the top n features by importance that influenced this prediction.
        """
        if not self.feature_importances_ or not self.feature_names_:
            return []
        items = sorted(self.feature_importances_.items(),
                       key=lambda kv: kv[1], reverse=True)[:n]
        return [{"feature": k, "importance": round(v, 4)} for k, v in items]

    def save(self, path=None):
        path = path or MODEL_PATH
        joblib.dump(self, path)
        return path

    @classmethod
    def load(cls, path=None):
        path = path or MODEL_PATH
        if not os.path.exists(path):
            return None
        obj = joblib.load(path)
        if isinstance(obj, cls):
            return obj
        return None
