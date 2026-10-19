
import pickle
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from src.data import get_data
from src.preprocess import clean_static
from src.eval import report, Timer


class TfidfLR:
    def __init__(self, **prep):
        self.prep = prep
        self.words = TfidfVectorizer(
            ngram_range=(1, 2), min_df=2, sublinear_tf=True)
        self.chars = TfidfVectorizer(
            analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True)
        self.clf = LogisticRegression(class_weight="balanced", max_iter=3000)

    def features(self, texts, fit=False):
        texts = [clean_static(t, **self.prep) for t in texts]
        if fit:
            return hstack([self.words.fit_transform(texts), self.chars.fit_transform(texts)])
        return hstack([self.words.transform(texts), self.chars.transform(texts)])

    def fit(self, texts, y):
        self.clf.fit(self.features(texts, fit=True), y)
        return self

    def predict_proba(self, texts):
        return self.clf.predict_proba(self.features(texts))[:, 1]


def run(name, train, val, test, **prep):
    with Timer() as t:
        model = TfidfLR(**prep).fit(train.tweet, train.y)
    report(name, val.y.values, model.predict_proba(val.tweet),
           test.y.values, model.predict_proba(test.tweet),
           prep=str(prep), train_time=t.s, n_params=model.clf.coef_.size)
    return model


if __name__ == "__main__":
    train, val, test = get_data()
    model = run("tfidf_lr", train, val, test)
    pickle.dump(model, open("models/tfidf_lr.pkl", "wb"))

    # ablation 4.3: remove one cue at a time
    run("tfidf_no_emoji", train, val, test, emojis=False)
    run("tfidf_no_laugh", train, val, test, laugh=False)
    run("tfidf_no_emoji_no_laugh", train, val, test, emojis=False, laugh=False)
