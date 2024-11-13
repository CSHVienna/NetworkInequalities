from netin.models import PATCHModel, CompoundLFM
from netin.utils import SimulationTimer

def run():
    n = 100
    k = 2
    f_m = 0.1
    h_MM = 0.9
    h_mm = 0.9
    p_tc = 0.8
    lfm_l = CompoundLFM.PAH
    lfm_g = CompoundLFM.PAH
    seed = 1234

    model = PATCHModel(n=n, m=k, f_m=f_m,
                       p_tc=p_tc,
                       lfm_local=lfm_l, lfm_global=lfm_g,
                       lfm_params={'h_mm': h_mm, 'h_MM': h_MM},
                       seed=seed)

    timer = SimulationTimer(model)
    model.simulate()
    print(f"Simulated model {model} in {timer.time:.2f} seconds.")


if __name__ == '__main__':
    run()
