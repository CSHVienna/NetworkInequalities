from netin import Graph


def run():
    n = 200
    k = 2
    f_m = 0.1
    tc = 0.5
    seed = 1234
    g = Graph(n=n, k=k, f_m=f_m, tc=tc, seed=seed)
    g.generate()
    g.info()


if __name__ == '__main__':
    run()
