"""
HW2 Q2: NeuMF (He et al., WWW 2017) - RecBole 구현 사용

실행: (이 파일과 train.csv, test.csv가 RecBole clone 폴더 안에 있어야 함)
    python hw2_neumf.py            # 튜닝 + 제출 파일
    python hw2_neumf.py --quick    # 동작 확인용 (2 epoch, 설정 1개)

1) train.csv -> dataset/hw2/hw2.inter (hw2_ease.py와 같은 변환, implicit feedback)
2) 사용자별 시간 순 80/10/10 분할로 임베딩 크기 튜닝 (valid NDCG@20 기준, early stopping)
3) 가장 좋은 설정과 epoch 수로 train.csv 전체를 다시 학습해 submission_neumf.csv 생성
"""
import os
import sys

import torch
_orig_load = torch.load
torch.load = lambda *a, **k: _orig_load(*a, **{**k, "weights_only": False})

import numpy as np
import pandas as pd
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.data.interaction import Interaction
from recbole.data.utils import create_samplers, get_dataloader
from recbole.model.general_recommender import NeuMF
from recbole.trainer import Trainer
from recbole.utils import init_seed, init_logger

os.chdir(os.path.dirname(os.path.abspath(__file__)))

QUICK = "--quick" in sys.argv
TOPK = 20
# 임베딩 크기 (GMF와 MLP 같은 크기, MLP 은닉층은 [2*d, d])
EMB_SIZES = [16] if QUICK else [16, 32, 64]

BASE_CONFIG = {
    "load_col": {"inter": ["user_id", "item_id", "rating", "timestamp"]},
    "eval_args": {"split": {"RS": [0.8, 0.1, 0.1]}, "order": "TO",
                  "group_by": "user", "mode": "full"},
    "metrics": ["NDCG", "Recall", "Hit"],
    "topk": [TOPK],
    "valid_metric": f"NDCG@{TOPK}",
    # 논문 설정: positive 1개당 negative 4개 샘플링, BCE loss
    "train_neg_sample_args": {"distribution": "uniform", "sample_num": 4,
                              "alpha": 1.0, "dynamic": False, "candidate_num": 0},
    "learning_rate": 0.001,
    "train_batch_size": 1024,
    "dropout_prob": 0.1,
    "epochs": 2 if QUICK else 100,
    "stopping_step": 10,
    "show_progress": False,
}


def model_config(emb):
    return {"mf_embedding_size": emb, "mlp_embedding_size": emb,
            "mlp_hidden_size": [2 * emb, emb]}


def prepare_dataset():
    df = pd.read_csv("train.csv")
    os.makedirs("dataset/hw2", exist_ok=True)
    df = df.rename(columns={"userId": "user_id:token", "movieId": "item_id:token",
                            "rating": "rating:float", "time_idx": "timestamp:float"})
    df.to_csv("dataset/hw2/hw2.inter", sep="\t", index=False)


# 2) 검증 세트로 튜닝: 설정마다 valid NDCG@20으로 early stopping, 최고 epoch 기록
def tune():
    rows = []
    for emb in EMB_SIZES:
        config = Config(model="NeuMF", dataset="hw2",
                        config_dict={**BASE_CONFIG, **model_config(emb)})
        init_seed(config["seed"], config["reproducibility"])
        init_logger(config)
        dataset = create_dataset(config)
        train_data, valid_data, test_data = data_preparation(config, dataset)
        model = NeuMF(config, train_data.dataset).to(config["device"])
        trainer = Trainer(config, model)

        history = []
        trainer.fit(train_data, valid_data, saved=True, show_progress=False,
                    callback_fn=lambda epoch, score: history.append((epoch, score)))
        best_epoch, best_valid = max(history, key=lambda x: x[1])
        test = trainer.evaluate(test_data, load_best_model=True, show_progress=False)
        rows.append({"emb_size": emb, "best_epoch": best_epoch + 1,
                     "valid_ndcg@20": round(best_valid, 4),
                     "test_ndcg@20": test[f"ndcg@{TOPK}"],
                     "test_recall@20": test[f"recall@{TOPK}"],
                     "test_hit@20": test[f"hit@{TOPK}"]})
        print(f"[tune] emb={emb}: {rows[-1]}", flush=True)

    res = pd.DataFrame(rows)
    res.to_csv("results_hw2_neumf.csv", index=False)
    print("\n[tuning] (valid로 고르고 test는 참고용)")
    print(res.to_string(index=False))
    best = res.loc[res["valid_ndcg@20"].idxmax()]
    print(f"[tuning] best emb_size = {int(best.emb_size)}, epochs = {int(best.best_epoch)}")
    return int(best.emb_size), int(best.best_epoch)


# 3) train.csv 전체로 같은 epoch 수만큼 재학습 후 test 사용자별 top-20 추천
def make_submission(emb, epochs):
    config = Config(model="NeuMF", dataset="hw2", config_dict={
        **BASE_CONFIG, **model_config(emb), "epochs": epochs,
        "eval_args": {"split": {"RS": [1.0, 0.0, 0.0]}, "order": "TO",
                      "group_by": "user", "mode": "full"}})
    init_seed(config["seed"], config["reproducibility"])
    dataset = create_dataset(config)
    built = dataset.build()                       # 분할 없이 전체가 train
    train_sampler, _, _ = create_samplers(config, dataset, built)
    train_data = get_dataloader(config, "train")(config, built[0], train_sampler, shuffle=True)

    model = NeuMF(config, built[0]).to(config["device"])
    Trainer(config, model).fit(train_data, valid_data=None, saved=False, show_progress=False)
    model.eval()

    test = pd.read_csv("test.csv")
    uids = dataset.token2id(dataset.uid_field, test["userId"].astype(str).tolist())
    seen = dataset.inter_matrix(form="csr")
    n_items = dataset.item_num
    all_items = torch.arange(n_items, device=config["device"])

    recs = []
    with torch.no_grad():
        for u in uids:
            inter = Interaction({
                dataset.uid_field: torch.full((n_items,), int(u), device=config["device"]),
                dataset.iid_field: all_items})
            scores = model.predict(inter).cpu().numpy()
            scores[seen[u].indices] = -np.inf          # 이미 본 아이템 제외
            scores[0] = -np.inf                        # RecBole의 [PAD] 아이템 제외
            top = np.argsort(-scores)[:TOPK]
            recs.append(" ".join(dataset.id2token(dataset.iid_field, top)))

    test["recommendations"] = recs
    out = "submission_neumf.csv"
    test[["id", "recommendations"]].to_csv(out, index=False)
    print(f"[submit] {out} 저장 ({len(test)} users, emb={emb}, epochs={epochs})")


if __name__ == "__main__":
    prepare_dataset()
    emb, epochs = tune()
    make_submission(emb, epochs)
