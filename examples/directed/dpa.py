from netin import DPA


def run():
    n = 200
    d = 0.1
    f_m = 0.1
    plo_M = 2.0
    plo_m = 2.0
    seed = 1234
    g = DPA(n=n, d=d, f_m=f_m, plo_M=plo_M, plo_m=plo_m, seed=seed)
    g.generate()
    g.info()


if __name__ == '__main__':
    run()
