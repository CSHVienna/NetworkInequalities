from netin.models import PATCHModel, CompoundLFM
from netin.utils import SimulationTimer

def run():
    n = 100
    k = 2
    f_m = 0.1
    h_M = 0.9
    h_m = 0.9
    tau = 0.8
    lfm_l = CompoundLFM.PAH
    lfm_g = CompoundLFM.PAH
    seed = 1234
    model = PATCHModel(n=n, m=k, f_m=f_m,
                       tau=tau,
                       lfm_tc=lfm_l, lfm_global=lfm_g,
                       h_m=h_m, h_M=h_M,
                       seed=seed)

    timer = SimulationTimer(model)
    model.simulate()
    print(f"Simulated model {model} in {timer.time:.2f} seconds.")


if __name__ == '__main__':
    run()
