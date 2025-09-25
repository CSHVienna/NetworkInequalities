from netin.models import PATCHModel, CompoundLFM
from netin.utils import SimulationTimer

def run():
    n = 4000
    k = 2
    f_m = 0.1
    h_MM = 0.9
    h_mm = 0.9
    tau = 0.8
    lfm_l = CompoundLFM.PAH
    lfm_g = CompoundLFM.PAH
    seed = 1234
    model = PATCHModel(n=n, k=k, f_m=f_m,
                       tau=tau,
                       lfm_tc=lfm_l, lfm_global=lfm_g,
                       h_mm=h_mm, h_MM=h_MM,
                       seed=seed)

    timer = SimulationTimer(model)
    model.simulate()
    print(f"Simulated model {model} in {timer.time:.2f} seconds.")


if __name__ == '__main__':
    run()
