from netin import Graph


def run():
    n = 200
    k = 2
    f_m = 0.1
    h_MM = 0.1
    h_mm = 0.9
    seed = 1234
    g = Graph(n=n, k=k, f_m=f_m, h_MM=h_MM, h_mm=h_mm, seed=seed)
    g.generate()
    g.info()


if __name__ == '__main__':
    run()
