from src.data import get_data
from src.preprocess import clean_static, remove_artifacts


def test_no_test_tweet_in_train():
    train, val, test = get_data()
    assert not set(train.tweet) & set(test.tweet)


def test_artifacts_removed():
    assert remove_artifacts('"RT @x: hello"') == "hello"


def test_cues_kept():
    out = clean_static("هههههه جميل 😂 !!!")
    assert "ضحك" in out    # laughter kept as a word
    assert "😂" in out      # emoji kept
    assert "!" in out
