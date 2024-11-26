from netin.models import HomophilyModel
from netin.utils import SimulationTimer

def run():
    n = 4000
    k = 2
    f_m = 0.1
    h_MM = 0.5
    h_mm = 0.5
    seed = 1234

    model = HomophilyModel(n=n, k=k, f_m=f_m,
                           h_mm=h_mm, h_MM=h_MM,
                           seed=seed)
    timer = SimulationTimer(model)
    model.simulate()
    print(f"Simulated model {model} in {timer.time:.2f} seconds.")

if __name__ == '__main__':
    run()
