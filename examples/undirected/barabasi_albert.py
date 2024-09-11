from netin.models import BarabasiAlbertModel
from netin.utils import SimulationTimer

def run():
    N = 200
    m = 2
    seed = 1234

    model = BarabasiAlbertModel(N=N,
                                m=m,
                                seed=seed)
    timer = SimulationTimer(model)
    model.simulate()
    print(f"Simulated model {model} in {timer.time:.2f} seconds.")

if __name__ == '__main__':
    run()
