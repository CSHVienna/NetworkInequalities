from netin.models import PAHModel
from netin.utils import SimulationTimer

def run():
    n = 4000
    k = 2
    f_m = 0.1
    h_MM = 0.5
    h_mm = 0.5
    seed = 1234
    m = PAHModel(N=n, m=k, f_m=f_m, h_m=h_mm, h_M=h_MM, seed=seed)
    timer = SimulationTimer(m)
    m.simulate()
    print(f"Simulated model {m} in {timer.time:.2f} seconds.")

if __name__ == '__main__':
    run()
