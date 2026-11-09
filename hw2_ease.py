"""
HW2 Q1: HW1의 EASE(RecBole)를 Kaggle 대회 데이터에 적용

실행: (이 파일과 train.csv, test.csv가 RecBole clone 폴더 안에 있어야 함)
    python hw2_ease.py

1) train.csv -> dataset/hw2/hw2.inter (RecBole 형식) 변환
2) 사용자별 시간 순(time_idx) 80/10/10 분할로 reg_weight 튜닝 (valid NDCG@20 기준)
3) 가장 좋은 reg_weight로 train.csv 전체를 다시 학습해 submission.csv 생성
"""
import os

import torch
_orig_load = torch.load
torch.load = lambda *a, **k: _orig_load(*a, **{**k, "weights_only": False})

import numpy as np
import pandas as pd
from recbole.config import Config
from recbole.data import create_dataset
from recbole.model.general_recommender import EASE
from recbole.quick_start import run_recbole

os.chdir(os.path.dirname(os.path.abspath(__file__)))

REG_WEIGHTS = [50, 100, 250, 500, 1000, 2000]
TOPK = 20

BASE_CONFIG = {
    "load_col": {"inter": ["user_id", "item_id", "rating", "timestamp"]},
    # 사용자별로 시간 순 마지막 10%는 valid, 그다음 10%는 test (리더보드와 같은 '미래 예측' 상황)
    "eval_args": {"split": {"RS": [0.8, 0.1, 0.1]}, "order": "TO",
                  "group_by": "user", "mode": "full"},
    "metrics": ["NDCG", "Recall", "Hit"],
    "topk": [TOPK],
    "valid_metric": f"NDCG@{TOPK}",
    "show_progress": False,
    "saved": False,
}


# 1) 데이터 변환: rating은 쓰지 않고 상호작용 여부(implicit)만 사용, time_idx는 정렬용 timestamp
def prepare_dataset():
    df = pd.read_csv("train.csv")
    os.makedirs("dataset/hw2", exist_ok=True)
    df = df.rename(columns={"userId": "user_id:token", "movieId": "item_id:token",
                            "rating": "rating:float", "time_idx": "timestamp:float"})
    df.to_csv("dataset/hw2/hw2.inter", sep="\t", index=False)
    print(f"[data] {len(df)} interactions -> dataset/hw2/hw2.inter")


# 2) 검증 세트로 reg_weight 튜닝
def tune():
    rows = []
    for reg in REG_WEIGHTS:
        r = run_recbole(model="EASE", dataset="hw2",
                        config_dict={**BASE_CONFIG, "reg_weight": reg})
        valid, test = dict(r["best_valid_result"]), dict(r["test_result"])
        rows.append({"reg_weight": reg,
                     "valid_ndcg@20": valid[f"ndcg@{TOPK}"],
                     "test_ndcg@20": test[f"ndcg@{TOPK}"],
                     "test_recall@20": test[f"recall@{TOPK}"],
                     "test_hit@20": test[f"hit@{TOPK}"]})
    res = pd.DataFrame(rows)
    res.to_csv("results_hw2_ease.csv", index=False)
    print("\n[tuning] (valid로 고르고 test는 참고용)")
    print(res.to_string(index=False))
    best = int(res.loc[res["valid_ndcg@20"].idxmax(), "reg_weight"])
    print(f"[tuning] best reg_weight = {best}")
    return best


# 3) train.csv 전체로 EASE 재학습 후 test 사용자별 top-20 추천
def make_submission(reg):
    config = Config(model="EASE", dataset="hw2",
                    config_dict={**BASE_CONFIG, "reg_weight": reg})
    dataset = create_dataset(config)      # 분할 전 전체 데이터
    model = EASE(config, dataset)         # HW1과 같은 RecBole EASE 코드 (B = 아이템-아이템 가중치)

    test = pd.read_csv("test.csv")
    uids = dataset.token2id(dataset.uid_field, test["userId"].astype(str).tolist())

    X = model.interaction_matrix[uids]                 # 각 사용자의 전체 시청 기록
    scores = np.asarray(X @ model.item_similarity)
    scores[X.toarray() > 0] = -np.inf                  # 이미 본 아이템 제외
    scores[:, 0] = -np.inf                             # RecBole의 [PAD] 아이템 제외

    top = np.argsort(-scores, axis=1)[:, :TOPK]
    items = dataset.id2token(dataset.iid_field, top)
    test["recommendations"] = [" ".join(row) for row in items]
    test[["id", "recommendations"]].to_csv("submission.csv", index=False)
    print(f"[submit] submission.csv 저장 ({len(test)} users, reg_weight={reg})")


if __name__ == "__main__":
    prepare_dataset()
    best_reg = tune()
    make_submission(best_reg)
