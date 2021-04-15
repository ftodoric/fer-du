from pathlib import Path


def test(a, z):
    print(a)
    print(z)


WEIGHT_DECAY = 1e-3
SAVE_DIR = Path(__file__).parent / 'out_task3' / \
    'lambda_{:.3f}'.format(WEIGHT_DECAY)


print(SAVE_DIR)
