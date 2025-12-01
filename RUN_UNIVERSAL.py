# ============================================================
# RUN_UNIVERSAL.py
# Tek kod → Google Colab + Oracle + Lokal hepsiyle uyumlu
# ============================================================

import os
import subprocess
import sys
import platform

# Colab kontrolü
IN_COLAB = "google.colab" in sys.modules

# HTML görüntüleme sadece COLAB’da aktif
if IN_COLAB:
    from IPython.display import IFrame, display
    from google.colab import files


print("\n🚀 UNIVERSAL LAUNCHER BAŞLADI\n")

# ------------------------------------------------------------
# 1️⃣ GitHub Repo Ayarları
# ------------------------------------------------------------
REPO_URL = "https://github.com/Yamann02/Quantum-The-Sentinel-Python.git"
REPO_NAME = "Quantum-The-Sentinel-Python"
MAIN_SCRIPT = "DENEME2.py"
HTML_FILE = "dashboard.html"
STUDY_FILE = "optuna_study.db"


# ------------------------------------------------------------
# 2️⃣ Repo İndir / Güncelle
# ------------------------------------------------------------
if not os.path.exists(REPO_NAME):
    print(f"📥 Repo bulunamadı → indiriliyor: {REPO_URL}")
    subprocess.run(["git", "clone", REPO_URL])
else:
    print(f"🔄 Repo bulundu → güncelleniyor")
    subprocess.run(["git", "-C", REPO_NAME, "pull"])

os.chdir(REPO_NAME)
print(f"\n📁 Çalışma dizini: {os.getcwd()}\n")


# ------------------------------------------------------------
# 3️⃣ Gereken paketleri yükle
# ------------------------------------------------------------
print("📦 Paketler yükleniyor...\n")
subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
subprocess.run([
    sys.executable, "-m", "pip", "install",
    "optuna", "pandas", "numpy", "matplotlib"
])


# ------------------------------------------------------------
# 4️⃣ Ana scripti çalıştır
# ------------------------------------------------------------
print(f"\n▶️ Ana Python dosyası çalıştırılıyor: {MAIN_SCRIPT}\n")
subprocess.run([sys.executable, MAIN_SCRIPT])


# ------------------------------------------------------------
# 5️⃣ Optuna sonuçlarını göster
# ------------------------------------------------------------
print("\n📊 Optuna sonucu kontrol ediliyor...")

import optuna
try:
    study = optuna.load_study(
        study_name="my_study",
        storage=f"sqlite:///{STUDY_FILE}"
    )
    print("\n⭐ En iyi değer:", study.best_value)
    print("🧠 En iyi parametreler:", study.best_params)
except Exception as e:
    print(f"⚠️ Optuna study bulunamadı: {e}")


# ------------------------------------------------------------
# 6️⃣ HTML Rapor (Colab veya Oracle)
# ------------------------------------------------------------
print("\n🌐 HTML rapor kontrol ediliyor...")

if os.path.exists(HTML_FILE):
    print(f"✔️ HTML bulundu → {HTML_FILE}")

    if IN_COLAB:
        # COLAB: direkt göster + download
        display(IFrame(HTML_FILE, width=1000, height=600))
        print("\n⬇️ HTML indiriliyor...")
        files.download(HTML_FILE)
    else:
        # ORACLE / LOKAL
        SAVE_DIR = "/home/opc/html_reports"
        os.makedirs(SAVE_DIR, exist_ok=True)

        new_path = os.path.join(SAVE_DIR, HTML_FILE)
        os.system(f"cp {HTML_FILE} {new_path}")

        print(f"\n📄 HTML dosyası buraya kopyalandı:")
        print(f"👉 {new_path}")
        print("\n🔗 Oracle üzerinde görüntülemek için Nginx kullanılabilir.")

else:
    print("⚠️ HTML dosyası bulunamadı.\n")

print("\n🎉 UNIVERSAL LAUNCHER TAMAMLANDI\n")
