from train_pranet import train
import optuna

def objective(trial):
    lr = trial.suggest_float(
        "lerning_rate", 1e-5, 1e-3, log=True
    )
    
    bs = trial.suggest_int(
        "batch_size", 2, 8, step=2
    )
    
    gc = trial.suggest_float(
        "gclip", 0.3, 0.7, step=0.1
    )
    return train('./polyps_dataset/', './trained_model', lr=lr, epochs=10, batch_size=bs, gclip=gc)


if __name__ == '__main__':
    study = optuna.create_study(
        storage="sqlite:///db.sqlite3",  # Specify the storage URL here.
        study_name="PraNet_mod_minusRes",
        direction='maximize'
        )
    # study = optuna.load_study(
    #     storage="sqlite:///db.sqlite3",  # Specify the storage URL here.
    #     study_name="PraNet_mod_plusRes"
    # )
    study.optimize(objective, n_trials=10)

    print(study.best_params)