# Bu dosyanın adı: veri_indir.py

import ccxt
import pandas as pd
import time
from datetime import datetime

# --- AYARLAR: BURAYI DÜZENLEYİN ---
symbol = '1000SHIBUSDT'  # <-- 1. DEĞİŞİKLİK: '1000SHIB/USDT' yerine '1000SHIBUSDT' (slash YOK)
timeframe = '5m'         
start_date = '2024-12-01T00:00:00Z' 
filename = '1000SHIB_5d_verisi.csv'   # Dosya adını aynı bırakabiliriz

# Borsa doğru, 'ccxt.binanceusdm' olarak kalmalı
exchange = ccxt.binanceusdm({   
    'rateLimit': 1200,
    'enableRateLimit': True
})
# -----------------------------------

def load_data(filename):
    """
    Belirtilen CSV dosyasından veriyi yükler ve backtesting için hazırlar.
    """
    if not os.path.exists(filename):
        print(f"Hata: '{filename}' dosyası bulunamadı. Lütfen önce veriyi indirin.")
        return None
    
    print(f"'{filename}' dosyasından veri yükleniyor...")
    df = pd.read_csv(filename, index_col='datetime', parse_dates=True)
    print("Veri başarıyla yüklendi.")
    return df


def fetch_all_ohlcv(symbol, timeframe, since):
    """
    Belirtilen sembol ve zaman aralığı için tüm geçmiş verileri çeker.
    """
    print(f"{symbol} için {timeframe} verisi {start_date} tarihinden itibaren indiriliyor...")
    
    # 'since' parametresini milisaniye timestamp'e çevir
    since_timestamp = exchange.parse8601(since)
    all_candles = []
    
    while True:
        try:
            # API'den 1000 mumluk veriyi çek
            candles = exchange.fetch_ohlcv(symbol, timeframe, since=since_timestamp, limit=1000)
            
            if len(candles) == 0:
                print("Veri indirme tamamlandı.")
                break
                
            all_candles.extend(candles)
            
            # Bir sonraki istek için başlangıç zamanını güncelle
            # Son mumun timestamp'ini al + 1
            since_timestamp = candles[-1][0] + 1 
            
            # API limitlerine takılmamak için kısa bir süre bekle
            time.sleep(exchange.rateLimit / 1000) # saniye cinsinden bekleme
            
            # Kullanıcıya ilerleme hakkında bilgi ver
            print(f"Toplam {len(all_candles)} mum indirildi. En son tarih: {exchange.iso8601(candles[-1][0])}")

        except ccxt.NetworkError as e:
            print(f"Ağ hatası: {e}. 5 saniye bekleyip tekrar deniyorum...")
            time.sleep(5)
        except ccxt.ExchangeError as e:
            print(f"Borsa hatası: {e}. Script durduruluyor.")
            return None
        except Exception as e:
            print(f"Bilinmeyen bir hata oluştu: {e}. Script durduruluyor.")
            return None
            
    return all_candles

if __name__ == "__main__":
    data = fetch_all_ohlcv(symbol, timeframe, start_date)
    
    if data:
        # Veriyi pandas DataFrame'e çevir
        df = pd.DataFrame(data, columns=['datetime', 'Open', 'High', 'Low', 'Close', 'Volume'])
        
        # Timestamp'i okunabilir tarihe çevir
        df['datetime'] = pd.to_datetime(df['datetime'], unit='ms')
        
        # Backtesting kütüphanesinin anlayacağı format:
        # 'datetime' sütununu index yap
        df.set_index('datetime', inplace=True)
        
        # Sütun isimlerini backtesting.py'nin sevdiği formata getir (İlk harf büyük)
        df.rename(columns={
            'open': 'Open',
            'high': 'High',
            'low': 'Low',
            'close': 'Close',
            'volume': 'Volume'
        }, inplace=True)
        
        # Eksik veri olup olmadığını kontrol et
        print(f"\nVeri {df.index.min()} ile {df.index.max()} arasındadır.")
        
        # CSV olarak kaydet
        df.to_csv(filename)
        print(f"\nBaşarılı! Veri '{filename}' dosyasına kaydedildi.")
