import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from feature_extraction import load_dataset, extract_mfcc
from gmm import GMMClassifier
from hmm import HMMClassifier

# CONFIG 
DATA_DIR = "."
GMM_COMPONENTS = 16    
HMM_STATES= 8     
DIGITS= list(range(10))

def accuracy(y_true, y_pred):
    return np.mean(np.array(y_true) == np.array(y_pred))

def confusion_matrix(y_true, y_pred, n=10):
    cm = np.zeros((n, n), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t][p] += 1
    return cm


def per_digit_metrics(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    rows = []
    for d in DIGITS:
        tp = ((y_true == d) & (y_pred == d)).sum()
        fp = ((y_true != d) & (y_pred == d)).sum()
        fn = ((y_true == d) & (y_pred != d)).sum()
        pr = tp / (tp + fp) if (tp + fp) else 0.0
        rc = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * pr * rc / (pr + rc) if (pr + rc) else 0.0
        rows.append((d, tp, tp + fn, pr, rc, f1))
    return rows


def plot_cm(cm, title, path):
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(DIGITS); ax.set_yticks(DIGITS)
    ax.set_xticklabels(DIGITS); ax.set_yticklabels(DIGITS)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(title)
    for i in DIGITS:
        for j in DIGITS:
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() * 0.5 else "black",
                    fontsize=9)
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def print_per_digit(rows, model_name):
    print(f"\n  Per-digit results ({model_name}) -- Speaker 6 test set")
    print(f"  {'Digit':>5}  {'Correct':>7}  {'Total':>5}  "
          f"{'Precision':>10}  {'Recall':>8}  {'F1':>8}")
    print("  " + "-" * 56)
    for d, tp, total, pr, rc, f1 in rows:
        print(f"  {d:>5}  {tp:>7}  {total:>5}  "
              f"{pr:>10.3f}  {rc:>8.3f}  {f1:>8.3f}")


def train_val_split(X_list, y_list, val_frac=0.15, seed=42):
    rng   = np.random.default_rng(seed)
    idx   = rng.permutation(len(X_list))
    n_val = max(1, int(len(idx) * val_frac))
    val_i = idx[:n_val]; tr_i = idx[n_val:]
    sel   = lambda lst, ii: [lst[i] for i in ii]
    return (sel(X_list, tr_i), [y_list[i] for i in tr_i],
            sel(X_list, val_i), [y_list[i] for i in val_i])

def main():
    #Features
    print("\n[1/5] Loading dataset and extracting features ...")
    train_X, train_y, test_X, test_y = load_dataset(DATA_DIR)

    #Val split
    print("\n[2/5] Creating validation split (15%) ...")
    tr_X, tr_y, val_X, val_y = train_val_split(train_X, train_y)
    print(f"  Train: {len(tr_X)}  |  Val: {len(val_X)}  |  Test (Spk6): {len(test_X)}")

    #GMM
    print("\n[3/5] Training GMM (diagonal cov, 16 components/digit) ...")
    gmm = GMMClassifier(n_components=GMM_COMPONENTS)
    gmm.fit(tr_X, tr_y)

    val_pred_gmm  = gmm.predict_all(val_X)
    test_pred_gmm = gmm.predict_all(test_X)
    val_acc_gmm   = accuracy(val_y,  val_pred_gmm)
    test_acc_gmm  = accuracy(test_y, test_pred_gmm)
    gmm_correct   = int(round(test_acc_gmm * len(test_y)))

    plot_cm(confusion_matrix(test_y, test_pred_gmm),
            "GMM Confusion Matrix -- Speaker 6", "confusion_gmm.png")
    gmm_metrics = per_digit_metrics(test_y, test_pred_gmm)
    print_per_digit(gmm_metrics, "GMM")

    #HMM
    print("\n[4/5] Training HMM (left-to-right, 8 states/digit) ...")
    hmm = HMMClassifier(n_states=HMM_STATES)
    hmm.fit(tr_X, tr_y)

    val_pred_hmm  = hmm.predict_all(val_X)
    test_pred_hmm = hmm.predict_all(test_X)
    val_acc_hmm   = accuracy(val_y,  val_pred_hmm)
    test_acc_hmm  = accuracy(test_y, test_pred_hmm)
    hmm_correct   = int(round(test_acc_hmm * len(test_y)))

    plot_cm(confusion_matrix(test_y, test_pred_hmm),
            "HMM Confusion Matrix -- Speaker 6", "confusion_hmm.png")
    hmm_metrics = per_digit_metrics(test_y, test_pred_hmm)
    print_per_digit(hmm_metrics, "HMM")

    #Final summary
    better = "HMM" if test_acc_hmm >= test_acc_gmm else "GMM"
    print("  [5/5] FINAL ACCURACY RESULTS -- Speaker 6 Test Set")
    print(f"  Total test files  : {len(test_y)}")
    print()
    print(f"  MODEL   Val Acc    Test Acc   Correct / Total")
    print(f"  GMM     {val_acc_gmm*100:>7.2f}%   "
          f"{test_acc_gmm*100:>7.2f}%   {gmm_correct}/{len(test_y)}")
    print(f"  HMM     {val_acc_hmm*100:>7.2f}%   "
          f"{test_acc_hmm*100:>7.2f}%   {hmm_correct}/{len(test_y)}")
    print()
    print(f"  Best model on Speaker 6: {better} ")

    import pickle
    with open("gmm_model.pkl", "wb") as f: pickle.dump(gmm, f)
    with open("hmm_model.pkl", "wb") as f: pickle.dump(hmm, f)
    print("\n  Models saved: gmm_model.pkl, hmm_model.pkl")

    return gmm, hmm



def load_models():
    import pickle
    with open("gmm_model.pkl", "rb") as f: gmm = pickle.load(f)
    with open("hmm_model.pkl", "rb") as f: hmm = pickle.load(f)
    return gmm, hmm


def inference(audio_filename, model="both"):
    feats = extract_mfcc(audio_filename)
    gmm, hmm = load_models()
    result = {}
    if model in ("gmm", "both"): result["gmm"] = gmm.predict(feats)
    if model in ("hmm", "both"): result["hmm"] = hmm.predict(feats)
    return result


if __name__ == "__main__":
    main()

