import argparse

def getParser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--update_interval', type=int, default=5)
    parser.add_argument('--actor_lr', type=float, default=0.0005)
    parser.add_argument('--critic_lr', type=float, default=0.001)

    args = parser.parse_args()
    return args