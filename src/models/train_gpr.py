import pandas as pd
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from scipy.stats import qmc
import sys
import os
data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data'))
sys.path.append(data_dir)
from mock_solver import solve_flutter


def train_active_gpr():
    csv_path = os.path.join(data_dir, 'initial_doe.csv')
    df = pd.read_csv(csv_path)
    X_train = df[['Mach', 'AoA', 'Dynamic_Pressure', 'Stiffness']].values
    y_train = df['Flutter_Load'].values

    kernel = Matern(nu=2.5)
    gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, alpha=0.1)

    print("Başlangıç GPR modeli eğitiliyor...")
    gpr.fit(X_train, y_train)

    print("Aktif Öğrenme (Adaptive Sampling) başlatılıyor...")
    l_bounds = [0.3, -5.0, 1000, 50]
    u_bounds = [1.2, 15.0, 5000, 200]
    sampler = qmc.LatinHypercube(d=4)

    for i in range(5):
        X_pool = qmc.scale(sampler.random(n=1000), l_bounds, u_bounds)

        y_pred, sigma = gpr.predict(X_pool, return_std=True)

        query_idx = np.argmax(sigma)
        X_new = X_pool[query_idx].reshape(1, -1)

        y_new = solve_flutter(X_new)

        X_train = np.vstack([X_train, X_new])
        y_train = np.append(y_train, y_new)
        gpr.fit(X_train, y_train)

        print(
            f"Aktif Öğrenme Adımı {i + 1}: Maksimum belirsizlik {sigma[query_idx]:.4f} olan nokta bulundu ve eklendi.")

    print("GPR Modeli Aktif Öğrenme ile başarıyla güçlendirildi.")


if __name__ == "__main__":
    train_active_gpr()