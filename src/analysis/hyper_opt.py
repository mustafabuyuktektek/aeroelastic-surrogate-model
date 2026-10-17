import optuna
import pandas as pd
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import os


def run_optuna():
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data'))
    df = pd.read_csv(os.path.join(data_dir, 'initial_doe.csv'))

    X = df[['Mach', 'AoA', 'Dynamic_Pressure', 'Stiffness']].values
    y = df['Flutter_Load'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    def objective(trial):
        C = trial.suggest_float('C', 1e-3, 1e3, log=True)
        epsilon = trial.suggest_float('epsilon', 1e-3, 1e1, log=True)
        svr = SVR(C=C, epsilon=epsilon)
        svr.fit(X_train, y_train)
        y_pred = svr.predict(X_test)
        return mean_squared_error(y_test, y_pred)

    study = optuna.create_study(direction='minimize')
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study.optimize(objective, n_trials=10)  # 10 deneme yeterli

    print("\n--- Optuna SVR Hiperparametre Optimizasyonu ---")
    print("Bulunan En İyi Parametreler:", study.best_params)


if __name__ == '__main__':
    run_optuna()