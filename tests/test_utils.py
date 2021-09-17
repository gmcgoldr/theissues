from theissues import utils


def test_get_id_to_token_should_return_dict():
    vocab = {"a": 1, "b": 2}
    assert utils.get_id_to_token(vocab) == {1: "a", 2: "b"}


def test_docode_token_ids_should_return_str():
    vocab = {"T1 ": 1, "t2": 2, "t3.": 3, "##s ": 4}
    id_to_token = utils.get_id_to_token(vocab)
    result = utils.decode_token_ids([1, 2, 4, 3], id_to_token)
    assert result == "T1 t2s t3."
