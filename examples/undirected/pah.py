from netin import PAH


def run():
    n = 1000
    k = 2
    f_m = 0.1
    h_MM = 0.5
    h_mm = 0.5
    seed = 1234
    g = PAH(n=n, k=k, f_m=f_m, h_MM=h_MM, h_mm=h_mm, seed=seed)
    g.generate()
    g.info()


if __name__ == '__main__':
    run()
