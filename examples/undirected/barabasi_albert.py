from netin.models import BarabasiAlbertModel


def run():
    N = 200
    m = 2
    seed = 1234

    g = BarabasiAlbertModel(
        N=N, m=m, seed=seed)
    g.simulate()

if __name__ == '__main__':
    run()
