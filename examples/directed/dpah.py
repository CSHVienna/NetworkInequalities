from netin.models import DPAHModel
from netin.utils import SimulationTimer


def run():
    N = 1000
    d = 0.005
    f_m = 0.1
    plo_M = 2.0
    plo_m = 2.0
    h_MM = 0.2
    h_mm = 0.9
    seed = 1234

    model = DPAHModel(N=N, d=d, f_m=f_m,
                      plo_M=plo_M, plo_m=plo_m,
                      h_m=h_mm, h_M=h_MM,
                      seed=seed)

    timer = SimulationTimer(model)
    model.simulate()
    print(f"Simulated model {model} in {timer.time:.2f} seconds.")


if __name__ == '__main__':
    run()
