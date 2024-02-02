from netin.models import HomophilyModel, BarabasiAlbertModel, PAH

def main():
    N = 5000
    m = 2
    f = 0.1
    h = 0.5
    model = PAH(N, m, f, h)
    model.simulate()
    print(model)

if __name__ == "__main__":
    main()
