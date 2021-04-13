import pdb


def proba():
    a = 5
    pdb.set_trace()
    raise ValueError("Iznimka!")


if __name__ == "__main__":
    proba()
