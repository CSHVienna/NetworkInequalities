from netin import PATCH


def run():
    n = 200
    k = 2
    f_m = 0.1
    h_MM = 0.1
    h_mm = 0.9
    tc = 0.5
    seed = 1234
    g = PATCH(n=n, k=k, f_m=f_m, h_MM=h_MM, h_mm=h_mm, tc=tc, seed=seed)
    g.generate()
    g.info()

if __name__ == '__main__':
    run()
