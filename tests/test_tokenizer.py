from joey.tokenizer import JoeyTokenizer
from joey.config import MASK_ID, PAD_ID, NUM_SPECIAL


def _toy_corpus(tmp_path):
    p = tmp_path / "corpus.txt"
    p.write_text("\n".join(["the cat sat on the mat"] * 200 +
                            ["a dog ran in the park"] * 200))
    return str(p)


def test_train_and_roundtrip(tmp_path):
    tok = JoeyTokenizer.train([_toy_corpus(tmp_path)], vocab_size=400)
    text = "the cat sat on the mat"
    assert tok.decode(tok.encode(text)) == text


def test_special_ids_reserved(tmp_path):
    tok = JoeyTokenizer.train([_toy_corpus(tmp_path)], vocab_size=400)
    assert tok.mask_id == MASK_ID
    assert tok.pad_id == PAD_ID
    # no normal token collides with a reserved special id
    ids = tok.encode("the cat")
    assert all(i >= NUM_SPECIAL for i in ids)


def test_ids_in_range(tmp_path):
    tok = JoeyTokenizer.train([_toy_corpus(tmp_path)], vocab_size=400)
    ids = tok.encode("a dog ran in the park")
    assert all(0 <= i < tok.vocab_size for i in ids)


def test_save_load(tmp_path):
    tok = JoeyTokenizer.train([_toy_corpus(tmp_path)], vocab_size=400)
    path = str(tmp_path / "tok.json")
    tok.save(path)
    tok2 = JoeyTokenizer.load(path)
    assert tok2.encode("the cat") == tok.encode("the cat")
