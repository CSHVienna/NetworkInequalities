from netin.models import PAModel
from netin.utils import SimulationTimer


def run():
    n = 4000
    k = 2
    f_m = 0.1
    seed = 1234

    model = PAModel(n=n,
                    k=k,
                    f_m=f_m,
                    seed=seed)
    timer = SimulationTimer(model)
    model.simulate()
    print(f"Simulated model {model} in {timer.time:.2f} seconds.")


if __name__ == '__main__':
    run()
