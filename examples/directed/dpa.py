from netin.models import DPAModel
from netin.utils import SimulationTimer


def run():
    n = 1000
    d = 0.005
    f_m = 0.1
    plo_M = 2.0
    plo_m = 2.0
    seed = 1234

    model = DPAModel(n=n, d=d, f_m=f_m,
                      plo_M=plo_M, plo_m=plo_m,
                      seed=seed)

    timer = SimulationTimer(model)
    model.simulate()
    print(f"Simulated model {model} in {timer.time:.2f} seconds.")


if __name__ == '__main__':
    run()
