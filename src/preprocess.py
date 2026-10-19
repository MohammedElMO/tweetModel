import re
import emoji
from arabert.preprocess import ArabertPreprocessor

arabert = ArabertPreprocessor("aubmindlab/bert-base-arabertv02-twitter")
LAUGH = r"ه{3,}|خخخ+"


def remove_artifacts(t):
    # export artifacts found in the audit (quotes, retweet prefix, truncation)
    t = str(t).strip().strip('"')
    t = re.sub(r"^RT @\w+:", "", t)
    return t.replace("…", "").strip()


def clean_plm(t):
    # for AraBERT / MARBERT: minimal cleaning, emojis kept
    return arabert.preprocess(remove_artifacts(t))


def clean_static(t, emojis=True, laugh=True, urls=True):
    # for TF-IDF and word embeddings
    t = remove_artifacts(t)
    # before arabert, which shortens ههههه
    t = re.sub(LAUGH, " ضحك " if laugh else " ", t)
    t = arabert.preprocess(t)
    t = re.sub("[إأآ]", "ا", t).replace("ة", "ه").replace("ى", "ي")
    keep = lambda e, *_: f" {e.group(0) if hasattr(e, 'group') else e} "
    t = emoji.replace_emoji(t, replace=keep if emojis else "")    
    if not urls:
        t = re.sub(r"\[رابط\]|\[مستخدم\]", "", t)
    return " ".join(t.split())


def remove_markers(t):
    # for test F4: remove emojis, laughter, repeated punctuation
    t = emoji.replace_emoji(re.sub(LAUGH, "", t), replace="")
    return re.sub(r"([!؟?])\1+", r"\1", t)
