def test_total_accuracy():
    true = ["jazz", "rock", "blues"]
    pred = ["jazz", "rock", "jazz"]
    assert total_accuracy(true, pred) == pytest.approx(2/3)

def test_per_genre_accuracy():
    true = ["jazz", "jazz", "rock"]
    pred = ["jazz", "rock", "rock"]
    result = per_genre(true, pred)
    assert result["jazz"] == 0.5
    assert result["rock"] == 1.0

def test_confusion_matrix():
    true = ["jazz", "rock"]
    pred = ["jazz", "jazz"]
    result = confusion_matrix(true, pred)
    assert result["jazz"]["jazz"] == 1
    assert result["rock"]["jazz"] == 1