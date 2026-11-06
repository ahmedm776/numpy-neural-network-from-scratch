import gzip, hashlib, numpy as np, os
D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
MD5 = {"train-images-idx3-ubyte.gz": "f68b3c2dcbeaaa9fbdd348bbdeb94873", "train-labels-idx1-ubyte.gz": "d53e105ee54ea40749a09fcbcd1e9432",
       "t10k-images-idx3-ubyte.gz": "9fb629c4189551a2d022fa330f9573f3", "t10k-labels-idx1-ubyte.gz": "ec29112dd5afa0611ce80d1b7f02629c"}
def _read(name, img):
    p = os.path.join(D, name); b = open(p, "rb").read()
    assert hashlib.md5(b).hexdigest() == MD5[name], name
    raw = gzip.decompress(b)
    return np.frombuffer(raw, np.uint8, offset=16).reshape(-1, 784) if img else np.frombuffer(raw, np.uint8, offset=8)
def load():
    return (_read("train-images-idx3-ubyte.gz", 1), _read("train-labels-idx1-ubyte.gz", 0),
            _read("t10k-images-idx3-ubyte.gz", 1), _read("t10k-labels-idx1-ubyte.gz", 0))
