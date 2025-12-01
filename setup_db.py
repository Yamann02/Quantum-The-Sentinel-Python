import optuna

def objective(trial):
    x = trial.suggest_float("x", -10, 10)
    return (x - 2) ** 2

# Bu satır db.sqlite3 dosyasını ve gerekli tabloları OLUŞTURUR
study = optuna.create_study(storage="sqlite:///db.sqlite3", study_name="ilk_deneme")
study.optimize(objective, n_trials=10)

print("Veritabanı başarıyla oluşturuldu!")