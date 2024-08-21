from netin.models import DPAHModel

def run():
    n = 1000
    d = 0.005
    f_m = 0.1
    plo_M = 2.0
    plo_m = 2.0
    h_MM = 0.2
    h_mm = 0.9
    seed = 1234
    m = DPAHModel(
        N=n, d=d, f_m=f_m,
        plo_M=plo_M, plo_m=plo_m,
        h_m=h_mm, h_M=h_MM,
        seed=seed)
    m.simulate()

if __name__ == '__main__':
    run()
