from netin import DPAH


def run():
    n = 1000
    d = 0.005
    f_m = 0.1
    plo_M = 2.0
    plo_m = 2.0
    h_MM = 0.5
    h_mm = 0.9
    seed = 1234
    g = DPAH(n=n, d=d, f_m=f_m, plo_M=plo_M, plo_m=plo_m, h_MM=h_MM, h_mm=h_mm, seed=seed)
    g.generate()
    g.info()


if __name__ == '__main__':
    run()
