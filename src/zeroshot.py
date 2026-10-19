import argparse
import numpy as np
from transformers import pipeline
from sklearn.metrics import precision_score, recall_score, f1_score
from src.data import get_data
from src.preprocess import remove_artifacts
from src.eval import report, Timer

parser = argparse.ArgumentParser()
parser.add_argument("--part", choices=["b1", "b2"], required=True)
args = parser.parse_args()

train, val, test = get_data()
X_val = [remove_artifacts(t) for t in val.tweet]
X_test = [remove_artifacts(t) for t in test.tweet]


def b1():
    zs = pipeline("zero-shot-classification", device=0,
                  model="MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7")
    # first label = sarcastic
    hypotheses = {
        "ar": (["ساخرة", "جادة"], "هذه التغريدة {}."),
        "fr": (["sarcastique", "sérieux"], "Ce tweet est {}."),
        "en": (["sarcastic", "serious"], "This tweet is {}."),
    }

    def p_sarcastic(texts, labels, template):
        out = zs(texts, candidate_labels=labels,
                 hypothesis_template=template, batch_size=32)
        return np.array([o["scores"][o["labels"].index(labels[0])] for o in out])

    for lang, (labels, template) in hypotheses.items():
        with Timer() as t:
            p_val = p_sarcastic(X_val, labels, template)
        p_test = p_sarcastic(X_test, labels, template)
        report(f"B1_zeroshot_{lang}", val.y.values, p_val, test.y.values, p_test,
               prep="remove_artifacts", extra={"inference_val_s": round(t.s, 1)})


def b2():
    sa = pipeline("text-classification", device=0,
                  model="CAMeL-Lab/bert-base-arabic-camelbert-da-sentiment")
    print("id2label:", sa.model.config.id2label)

    def p_negative(texts):
        out = sa(texts, top_k=None, batch_size=64, truncation=True)
        return np.array([{d["label"]: d["score"] for d in o}["negative"] for o in out])

    p_val, p_test = p_negative(X_val), p_negative(X_test)

    # strict rule: predicted label == negative  =>  sarcastic
    rule = np.array([o["label"] == "negative" for o in sa(
        X_val, batch_size=64, truncation=True)]).astype(int)
    print(f"rule NEG=>sarc (val): P={precision_score(val.y, rule):.3f} "
          f"R={recall_score(val.y, rule):.3f} F1={f1_score(val.y, rule):.3f}")

    # P(negative) used as a score, threshold tuned on val
    report("B2_neg_is_sarc", val.y.values, p_val,
           test.y.values, p_test, prep="remove_artifacts")

    # upper bound: same rule with the GOLD sentiment label
    g = (val.sentiment == "NEG").astype(int)
    print(f"gold NEG=>sarc (val): P={precision_score(val.y, g):.3f} "
          f"R={recall_score(val.y, g):.3f} F1={f1_score(val.y, g):.3f}")


b1() if args.part == "b1" else b2()
