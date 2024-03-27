import pandas as pd
import numpy as np

from typing import Any


def get_outcome(outcome, true_label, pred_label, invert: bool = False) -> str:
    is_equal = true_label == pred_label
    is_outcome_correct = outcome == 'correct'

    if invert:
        is_outcome_correct = not is_outcome_correct

    if is_outcome_correct:
        if is_equal:
            return 'tp'
        else:
            return 'fp'
    else:
        if is_equal:
            return 'tn'
        else:
            return 'fn'


class Evaluation:
    def __init__(self, notifications: pd.DataFrame, labels: pd.DataFrame, predictions: pd.DataFrame,
                 invert: bool = False):

        notification_counts = notifications['notification'].value_counts()
        # transform notifications to lowercase
        notifications['notification'] = notifications['notification'].apply(lambda x: x.lower())

        self.correct = notification_counts.get('correct', 0)
        self.incorrect = notification_counts.get('incorrect', 0)
        self.uncertain = notification_counts.get('uncertain', 0)

        # transform all uncertain notifications to incorrect
        notifications['notification'] = notifications['notification'].apply(lambda x: 'incorrect' if x == 'uncertain' else x)
        labels.rename(columns={'y': 'true_label'}, inplace=True)
        predictions.rename(columns={'y': 'pred_label'}, inplace=True)

        self.results = (notifications.merge(predictions, left_index=True, right_index=True)
                        .merge(labels, left_index=True, right_index=True))

        outcomes = self.results.apply(lambda x: get_outcome(x['notification'], x['true_label'], x['pred_label'], invert),
                                      axis=1)
        outcomes_counts = outcomes.value_counts()

        self.true_pos = outcomes_counts.get('tp', 0)
        self.false_pos = outcomes_counts.get('fp', 0)
        self.true_neg = outcomes_counts.get('tn', 0)
        self.false_neg = outcomes_counts.get('fn', 0)

        self.total = len(notifications)
        self.labels = labels
        self.predictions = predictions

    @property
    def retrieved(self):
        return self.true_pos + self.false_pos

    @property
    def relevant(self):
        return self.true_pos + self.false_neg

    @property
    def precision(self):
        return (self.true_pos / self.retrieved) if self.retrieved > 0 else 0

    @property
    def recall(self):
        return (self.true_pos / self.relevant) if self.relevant > 0 else 0

    @property
    def f1(self):
        recall_precision = self.precision + self.recall
        return (2 * self.precision * self.recall) / recall_precision if recall_precision > 0 else 0

    @property
    def mcc(self):
        covar = self.true_pos * self.true_neg - self.false_pos * self.false_neg
        denom = np.sqrt((self.true_pos + self.false_pos) * (self.true_pos + self.false_neg) *
                        (self.true_neg + self.false_pos) * (self.true_neg + self.false_neg))

        if denom == 0:
            print(f"Warning: MCC denominator is zero. True Pos: {self.true_pos}, False Pos: {self.false_pos}, "
                  f"True Neg: {self.true_neg}, False Neg: {self.false_neg}")

        return covar / denom if denom > 0 else 0

    @property
    def false_positive_rate(self):
        return self.false_pos / (self.false_pos + self.true_neg)

    @property
    def true_positive_rate(self):
        denominator = self.true_pos + self.false_neg
        return (self.true_pos / denominator) if denominator > 0 else 0

    def performance(self):
        return {
            "tpr": round(self.true_positive_rate * 100.0, 2),
            "fpr": round(self.false_positive_rate * 100.0, 2),
            "precision": round(self.precision * 100.0, 2),
            "recall": round(self.recall * 100.0, 2),
            "f1": round(self.f1 * 100.0, 2),
            "mcc": round(self.mcc, 3),
        }

    def to_dict(self):
        return {
            "correct": self.correct,
            "incorrect": self.incorrect,
            "uncertain": self.uncertain,
            "tps": self.true_pos,
            "fps": self.false_pos,
            "tns": self.true_neg,
            "fns": self.false_neg,
        }
