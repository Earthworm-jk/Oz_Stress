import numpy as np


def raw_prediction(pred):
    return np.asarray(pred, dtype=float)


def clip_0_1(pred):
    return np.clip(raw_prediction(pred), 0.0, 1.0)


def clip_round_2(pred):
    return np.round(clip_0_1(pred), 2)


POSTPROCESSORS = {
    "raw": raw_prediction,
    "clip_0_1": clip_0_1,
    "clip_0_1_round2": clip_round_2,
}
