import pandas as pd
from SALib.sample import saltelli
from SALib.analyze import sobol
from sklearn.gaussian_process import GaussianProcessRegressor
import os


def run_sobol():
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data'))
    df = pd.read_csv(os.path.join(data_dir, 'initial_doe.csv'))

    X = df[['Mach', 'AoA', 'Dynamic_Pressure', 'Stiffness']].values
    y = df['Flutter_Load'].values

    gpr = GaussianProcessRegressor(alpha=0.1)
    gpr.fit(X, y)

    problem = {
        'num_vars': 4,
        'names': ['Mach', 'AoA', 'Dynamic_Pressure', 'Stiffness'],
        'bounds': [[0.3, 1.2], [-5.0, 15.0], [1000, 5000], [50, 200]]
    }

    X_sobol = saltelli.sample(problem, 1024)
    Y_sobol = gpr.predict(X_sobol)

    Si = sobol.analyze(problem, Y_sobol, print_to_console=False)
    print("\n--- Sobol Hassasiyet Analizi Sonuçları ---")
    for i, name in enumerate(problem['names']):
        print(f"{name} Parametresi Toplam Etki: {Si['ST'][i]:.4f}")


if __name__ == '__main__':
    run_sobol()