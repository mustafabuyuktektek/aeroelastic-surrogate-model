import numpy as np
import pandas as pd
from scipy.stats import qmc
from mock_solver import solve_flutter
import os


def generate_initial_data(n_samples=200):
    sampler = qmc.LatinHypercube(d=4)
    sample = sampler.random(n=n_samples)

    l_bounds = [0.3, -5.0, 1000, 50]
    u_bounds = [1.2, 15.0, 5000, 200]

    X = qmc.scale(sample, l_bounds, u_bounds)

    y = solve_flutter(X)

    df = pd.DataFrame(X, columns=['Mach', 'AoA', 'Dynamic_Pressure', 'Stiffness'])
    df['Flutter_Load'] = y

    os.makedirs(os.path.dirname(os.path.abspath(__file__)), exist_ok=True)
    csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'initial_doe.csv')
    df.to_csv(csv_path, index=False)
    print(f"{n_samples} adet LHS örneklemi '{csv_path}' konumuna kaydedildi.")


if __name__ == "__main__":
    generate_initial_data()