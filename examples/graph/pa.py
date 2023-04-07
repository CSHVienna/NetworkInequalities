from netin import PA


def run():
    n = 200
    k = 2
    f_m = 0.1
    seed = 1234
    g = PA(n=n, k=k, f_m=f_m, seed=seed)
    g.generate()
    g.info()


if __name__ == '__main__':
    run()
