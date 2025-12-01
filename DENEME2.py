"""
WHALE STOP STRATEGY V5 - PINE SCRIPT %100 UYUMLU
V4'ün doğru hesaplamaları + V3'ün detaylı görselleştirmesi
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')


class WhaleStopStrategy:
    def __init__(self, df):
        """Pine Script stratejisinin %100 uyumlu Python karşılığı"""
        self.df = df.copy()
        self.df.reset_index(drop=True, inplace=True)
        self.df['bar_index'] = self.df.index
        
        # Temel Ayarlar
        self.use_long_step_entry = True
        self.use_bullish_entry = True
        self.use_long_approach_entry = True
        self.use_long_alt_entry = True
        self.use_short_step_entry = True
        self.use_bearish_entry = True
        self.use_short_approach_entry = True
        self.use_short_alt_entry = True
        
        # Parametreler
        self.long_islem_aktif = True
        self.short_islem_aktif = False
        self.enable_mum_sayisi = True
        self.mum_sayisi = 10
        self.enable_position_multiplier = True
        self.position_multiplier = 2.0

        # TP
        self.enable_take_profit = False
        self.take_profit_perc = 5.0
        self.tp_type = "Fixed"  # Hata veren eksik özellik eklendi
        
        # USD Hesaplama
        self.manual_lot_usd = 100.0
        self.enable_min_lot = True
        self.min_lot = 1.0
        self.max_qty = 10.0
        self.enable_leverage = False
        self.leverage = 10
        
        # Trailing Stop
        self.enable_trailing_stop = True
        self.trailing_perc = 0.5 / 100  # %0.5
        self.enable_min_position_for_ts = True
        self.min_position_for_ts = 3
        # Trailing sonrası extra TP (yeni eklendi)
        self.enable_extra_tp_after_ts = False
        self.extra_tp_perc = 1.0
        
        # Maliyet Kontrol
        self.enable_maliyet_exit = True
        self.min_entry_for_maliyet = 5
        self.maliyet_return_perc = 0.1
        # Miniloss
        self.enable_mini_stop = True
        self.min_entry_for_mini_loss = 5
        self.maliyet_stop_perc = 2.0
        self.mini_loss_cooldown_bars = 10
        
        # DCA
        self.enable_dca = True
        self.max_dca = 3
        self.dca_drop_perc = 3.0 * 0.01
        self.enable_dca_mum_delay = True
        self.dca_mum_delay = 4
        
        # Whale Stop
        self.enable_drop_speed = True
        self.drop_speed_perc = 2.0
        self.drop_speed_bars = 1
        self.enable_panic_drop = True
        self.panic_drop_perc = 10.0
        self.panic_drop_bars = 1
        self.panic_cooldown_bars = 100
        
        # İndikatör Parametreleri
        self.bb_length = 20
        self.bb_std_dev = 2.0
        self.rsi_periodu = 14
        self.rsi_taban_degeri = 20
        self.rsi_tavan_degeri = 80
        
        # Stochastic RSI 1
        self.stoch_rsi1_k_length = 3
        self.stoch_rsi1_d_length = 3
        self.stoch_rsi1_rsi_length = 14
        self.stoch_rsi1_stoch_length = 14
        
        # Stochastic RSI 2
        self.stoch_rsi2_k_length = 3
        self.stoch_rsi2_d_length = 3
        self.stoch_rsi2_rsi_length = 8
        self.stoch_rsi2_stoch_length = 10
        
        # Diğer İndikatörler
        self.williams_r_length = 14
        self.rsi_length = 14
        self.smoothing_length = 7
        self.upper_band_level = 70
        self.lower_band_level = 30
        self.adx_length = 14
        self.domcycle = 20
        self.vibration = 10
        self.leveling = 10.0
        self.fisher_len = 9
        self.fast_length = 12
        self.slow_length = 26
        self.signal_smoothing = 9
        
        # State Variables
        self.step = 0
        self.reset_bar = None
        self.step_short = 0
        self.reset_bar_short = None
        self.mum_sayaci = 0
        self.sinyal_verildi = False
        self.long_signal_count = 0
        self.short_signal_count = 0
        self.prev_step = 0
        self.prev_step_short = 0
        
        # Önceki durum değişkenleri
        self.prev_bullish_kosul = False
        self.prev_bearish_kosul = False
        self.prev_long_approach = False
        self.prev_short_approach = False
        self.prev_long_alt = False
        self.prev_short_alt = False
        
        # Position Variables
        self.position_size = 0
        self.avg_long_price = None
        self.avg_short_price = None
        self.next_long_size = None
        self.next_short_size = None
        
        # Trailing Stop
        self.entry_bar_long = None
        self.entry_bar_short = None
        self.ts_active_long = False
        self.ts_active_short = False
        self.trail_stop_long = None
        self.trail_stop_short = None
        self.plot_trail_stop_long = None
        self.plot_trail_stop_short = None
        
        # DCA
        self.last_entry_price_long = None
        self.last_entry_size_long = None
        self.dca_count_long = 0
        self.dca_bar_count_long = 0
        self.last_entry_price_short = None
        self.last_entry_size_short = None
        self.dca_count_short = 0
        self.dca_bar_count_short = 0
        
        # Cooldown
        self.is_panic_cooldown_active = False
        self.panic_cooldown_start_bar = 0
        self.long_signal_triggered_this_bar = False
        self.short_signal_triggered_this_bar = False
        self.is_mini_loss_cooldown_active = False
        self.mini_loss_cooldown_start_bar = 0
        
        # Mini Loss
        self.mini_loss_level = None
        self.can_draw_mini_loss = False
        
        # Exit flag
        self.exit_occurred_this_bar = False
        
        # Results
        self.trades = []
        self.signals = []
    
    # ==================== PINE SCRIPT UYUMLU İNDİKATÖRLER (V4'TEN) ====================
    
    def ta_rma(self, series, length):
        """Pine Script ta.rma (Wilder's smoothing) - V4"""
        alpha = 1.0 / length
        rma = pd.Series(index=series.index, dtype=float)
        
        for i in range(len(series)):
            if i == 0:
                rma.iloc[i] = series.iloc[i] if not pd.isna(series.iloc[i]) else 0.0
            elif pd.isna(series.iloc[i]):
                rma.iloc[i] = rma.iloc[i-1]
            else:
                rma.iloc[i] = alpha * series.iloc[i] + (1 - alpha) * rma.iloc[i-1]
        
        return rma
    
    def ta_rsi(self, source, length):
        """Pine Script ta.rsi - Wilder's RSI - V4"""
        delta = source.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
        
        avg_gain = self.ta_rma(gain, length)
        avg_loss = self.ta_rma(loss, length)
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        rsi.iloc[:length] = np.nan
        
        return rsi
    
    def ta_stoch(self, source, high, low, length):
        """Pine Script ta.stoch - V4"""
        lowest = low.rolling(window=length).min()
        highest = high.rolling(window=length).max()
        
        range_val = highest - lowest
        stoch = pd.Series(0.0, index=source.index)
        
        mask = range_val != 0
        stoch[mask] = 100 * (source[mask] - lowest[mask]) / range_val[mask]
        
        return stoch
    
    def calculate_bollinger_bands(self):
        """Bollinger Bands"""
        basis = self.df['close'].rolling(window=self.bb_length).mean()
        std = self.df['close'].rolling(window=self.bb_length).std(ddof=0)
        upper = basis + self.bb_std_dev * std
        lower = basis - self.bb_std_dev * std
        return basis, upper, lower
    
    def calculate_stochastic_rsi(self, rsi_length, stoch_length, k_length, d_length):
        """Stochastic RSI - V4 Hesaplama"""
        rsi_value = self.ta_rsi(self.df['close'], rsi_length)
        stoch_rsi = self.ta_stoch(rsi_value, rsi_value, rsi_value, stoch_length)
        stoch_rsi_k = stoch_rsi.rolling(window=k_length).mean()
        stoch_rsi_d = stoch_rsi_k.rolling(window=d_length).mean()
        return stoch_rsi_k, stoch_rsi_d, rsi_value
    
    def calculate_williams_r(self):
        """Williams %R"""
        highest_high = self.df['high'].rolling(window=self.williams_r_length).max()
        lowest_low = self.df['low'].rolling(window=self.williams_r_length).min()
        williams_r = -100 * (highest_high - self.df['close']) / (highest_high - lowest_low)
        return williams_r
    
    def calculate_smoothed_rsi(self):
        """Smoothed RSI - V4 Hesaplama"""
        rsi = self.ta_rsi(self.df['close'], self.rsi_length)
        smoothed_rsi = rsi.ewm(span=self.smoothing_length, adjust=False).mean()
        return smoothed_rsi
    
    def calculate_adx(self):
        """
        ADX - Pine Script ile TAM UYUMLU
        Pine Script ta.sma kullanır (RMA DEĞİL!)
        """
        high = self.df['high']
        low = self.df['low']
        close = self.df['close']
        
        # True Range
        high_low = high - low
        high_close = (high - close.shift(1)).abs()
        low_close = (low - close.shift(1)).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        
        # Directional Movement
        up_move = high - high.shift(1)
        down_move = low.shift(1) - low
        
        plus_dm = pd.Series(0.0, index=close.index)
        minus_dm = pd.Series(0.0, index=close.index)
        
        # Pine Script mantığı: high - highPrev > lowPrev - low
        plus_dm[up_move > down_move] = up_move[up_move > down_move].clip(lower=0)
        minus_dm[down_move > up_move] = down_move[down_move > up_move].clip(lower=0)
        
        # ⭐ KRİTİK: Pine Script SMA kullanır, RMA DEĞİL!
        tr_smooth = tr.rolling(window=self.adx_length).mean()  # SMA
        plus_dm_smooth = plus_dm.rolling(window=self.adx_length).mean()  # SMA
        minus_dm_smooth = minus_dm.rolling(window=self.adx_length).mean()  # SMA
        
        # +DI ve -DI
        plus_di = 100 * plus_dm_smooth / tr_smooth
        minus_di = 100 * minus_dm_smooth / tr_smooth
        
        # DX
        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
        
        # ⭐ KRİTİK: ADX smoothing de SMA ile yapılır
        adx = dx.rolling(window=self.adx_length).mean()  # SMA
        
        return adx


    def calculate_cyclic_rsi(self):
        """
        Cyclic RSI - Pine Script ile %100 uyumlu
        """
        cyclelen = self.domcycle // 2
        src = self.df['close']
        delta = src.diff()
        
        up = delta.where(delta > 0, 0.0)
        down = -delta.where(delta < 0, 0.0)
        
        # RMA kullanımı doğru
        up_avg = self.ta_rma(up, cyclelen)
        down_avg = self.ta_rma(down, cyclelen)
        
        # Cycle RSI hesaplama
        cycle_rsi = pd.Series(100.0, index=src.index)
        mask = (down_avg != 0) & (up_avg != 0)
        cycle_rsi[mask] = 100 - 100 / (1 + up_avg[mask] / down_avg[mask])
        cycle_rsi[up_avg == 0] = 0
        
        # Phasing ve smoothing
        torque = 2.0 / (self.vibration + 1)
        phasing_lag = int((self.vibration - 1) / 2.0)
        
        crsi = pd.Series(index=src.index, dtype=float)
        for i in range(len(src)):
            if i < phasing_lag:
                crsi.iloc[i] = cycle_rsi.iloc[i]
            else:
                prev_crsi = crsi.iloc[i-1] if i > 0 and not pd.isna(crsi.iloc[i-1]) else 0
                crsi.iloc[i] = torque * (2 * cycle_rsi.iloc[i] - cycle_rsi.iloc[i - phasing_lag]) + (1 - torque) * prev_crsi
        
        # dB ve uB hesaplama (Pine Script mantığı)
        cyclicmemory = self.domcycle * 2
        aperc = self.leveling / 100
        
        db = pd.Series(index=src.index, dtype=float)
        ub = pd.Series(index=src.index, dtype=float)
        
        for i in range(len(src)):
            if i < cyclicmemory:
                db.iloc[i] = np.nan
                ub.iloc[i] = np.nan
                continue
            
            # Pine Script: crsi[i], crsi[i-1], ..., crsi[i-cyclicmemory+1]
            # Python: crsi.iloc[i-cyclicmemory+1:i+1]
            window_vals = [crsi.iloc[j] for j in range(i - cyclicmemory + 1, i + 1) if not pd.isna(crsi.iloc[j])]
            if not window_vals:
                db.iloc[i] = np.nan
                ub.iloc[i] = np.nan
                continue
            
            lmax = max(window_vals)
            lmin = min(window_vals)
            mstep = (lmax - lmin) / 100 if lmax != lmin else 0
            
            # dB calculation
            db_val = lmin
            if mstep > 0:
                for steps in range(101):
                    testvalue = lmin + mstep * steps
                    below = sum(1 for v in window_vals if v < testvalue)
                    ratio = below / len(window_vals)
                    if ratio >= aperc:
                        db_val = testvalue
                        break
            db.iloc[i] = db_val
            
            # uB calculation
            ub_val = lmax
            if mstep > 0:
                for steps in range(101):
                    testvalue = lmax - mstep * steps
                    above = sum(1 for v in window_vals if v >= testvalue)
                    ratio = above / len(window_vals)
                    if ratio >= aperc:
                        ub_val = testvalue
                        break
            ub.iloc[i] = ub_val
        
        return crsi, db, ub


    def calculate_fisher_transform(self):
        """
        Fisher Transform - Pine Script ile %100 uyumlu
        """
        hl2 = (self.df['high'] + self.df['low']) / 2
        high_ = hl2.rolling(window=self.fisher_len).max()
        low_ = hl2.rolling(window=self.fisher_len).min()
        
        value = pd.Series(0.0, index=hl2.index)
        fish1 = pd.Series(0.0, index=hl2.index)
        
        for i in range(1, len(hl2)):
            range_val = high_.iloc[i] - low_.iloc[i]
            
            if pd.notna(range_val) and range_val != 0:
                val = 0.66 * ((hl2.iloc[i] - low_.iloc[i]) / range_val - 0.5) + 0.67 * value.iloc[i-1]
            else:
                val = value.iloc[i-1]
            
            # Pine Script: round_ fonksiyonu
            val = max(min(val, 0.999), -0.999)
            value.iloc[i] = val
            
            # Fish1 hesaplama
            fish1.iloc[i] = 0.5 * np.log((1 + val) / (1 - val)) + 0.5 * fish1.iloc[i-1]
        
        fish2 = fish1.shift(1)
        return fish1, fish2

    
    def calculate_dema(self, series, length):
        """DEMA"""
        ema1 = series.ewm(span=length, adjust=False).mean()
        ema2 = ema1.ewm(span=length, adjust=False).mean()
        dema = 2 * ema1 - ema2
        return dema
    
    def calculate_macd_dema(self):
        """MACD DEMA"""
        macd_line = self.calculate_dema(self.df['close'], self.fast_length) - self.calculate_dema(self.df['close'], self.slow_length)
        signal_line = macd_line.ewm(span=self.signal_smoothing, adjust=False).mean()
        macd_histogram = macd_line - signal_line
        return macd_line, signal_line, macd_histogram
    
    def calculate_obv_rsi(self):
        """OBV RSI - V4"""
        price_change = self.df['close'].diff()
        obv_value = (self.df['volume'] * np.sign(price_change)).cumsum()
        obv_rsi = self.ta_rsi(obv_value, 14)
        return obv_rsi
    
    def run_strategy(self):
        """Stratejiyi çalıştır"""
        print("="*70)
        print("☄️ QUANTUM THE SENTİNEL ☄️")
        print("="*70)
        # Optimizasyon için indikatör hesaplamaları bu fonksiyondan çıkarıldı.
        # Hesaplamalar artık strateji çalıştırılmadan ÖNCE bir kez yapılıyor.
        # Bu, her Optuna denemesinde tekrar tekrar hesaplama yapılmasını önler.
        
        print("\n📊 İndikatörler hesaplanıyor (V4 Hesaplama Motor)...")
        
        self.df['bb_basis'], self.df['bb_upper'], self.df['bb_lower'] = self.calculate_bollinger_bands()
        print("   ✅ Bollinger Bands")
        
        self.df['stoch_rsi1_k'], self.df['stoch_rsi1_d'], self.df['rsi_value1'] = self.calculate_stochastic_rsi(
            self.stoch_rsi1_rsi_length, self.stoch_rsi1_stoch_length, 
            self.stoch_rsi1_k_length, self.stoch_rsi1_d_length
        )
        print("   ✅ Stochastic RSI 1")
        
        self.df['stoch_rsi2_k'], self.df['stoch_rsi2_d'], self.df['rsi_value2'] = self.calculate_stochastic_rsi(
            self.stoch_rsi2_rsi_length, self.stoch_rsi2_stoch_length,
            self.stoch_rsi2_k_length, self.stoch_rsi2_d_length
        )
        print("   ✅ Stochastic RSI 2")
        
        self.df['williams_r'] = self.calculate_williams_r()
        print("   ✅ Williams %R")
        
        self.df['smoothed_rsi'] = self.calculate_smoothed_rsi()
        print("   ✅ Smoothed RSI")
        
        self.df['adx'] = self.calculate_adx()
        print("   ✅ ADX")
        
        self.df['crsi'], self.df['db'], self.df['ub'] = self.calculate_cyclic_rsi()
        print("   ✅ Cyclic RSI")
        
        self.df['fish1'], self.df['fish2'] = self.calculate_fisher_transform()
        print("   ✅ Fisher Transform")
        
        self.df['macd_line'], self.df['signal_line'], self.df['macd_histogram'] = self.calculate_macd_dema()
        print("   ✅ MACD DEMA")
        
        self.df['rsi_degeri'] = self.ta_rsi(self.df['close'], self.rsi_periodu)
        print("   ✅ RSI")
        
        self.df['sma200'] = self.df['close'].rolling(window=50).mean()
        self.df['ema1'] = self.df['close'].ewm(span=14, adjust=False).mean()
        print("   ✅ SMA & EMA")
        
        self.df['obv_rsi'] = self.calculate_obv_rsi()
        print("   ✅ OBV RSI")
        
        print("\n✅ Tüm indikatörler hazır!")
        
        # Trade USD hesapla
        lev = max(1, self.leverage) if self.enable_leverage else 1
        trade_amount_usd = (self.min_lot * self.manual_lot_usd) / lev
        self.next_long_size = self.min_lot
        self.next_short_size = self.min_lot
        
        print(f"\n💰 Trade Ayarları:")
        print(f"   • Trade Amount USD: ${trade_amount_usd:.2f}")
        print(f"   • Initial Size: {self.min_lot:.4f}")
        
        # Her bar için strateji
        # print(f"\n⚙️ Strateji çalıştırılıyor... ({len(self.df)} bar)") # Optimizasyon sırasında konsolu kirletmemek için kapatıldı
        for i in range(len(self.df)):
            # if i % 500 == 0 and i > 0:
            #     print(f"   📍 Bar {i}/{len(self.df)} işlendi...")
            self.process_bar(i)
        # print(f"\n✅ Strateji tamamlandı!")
        print(f"\n✅ Strateji tamamlandı!")
        print(f"\n📈 Sonuçlar:")
        print(f"   • Toplam Sinyal: {len(self.signals)}")
        print(f"   • Toplam İşlem: {len(self.trades)}")
        print("="*70)
        
        return self.df, self.trades, self.signals
    
    def process_bar(self, i):
        """Her bar için strateji logiğini işle"""
        bar_index = i
        
        if i < self.bb_length:
            return
        
        current_price = self.df['close'].iloc[i]
        self.exit_occurred_this_bar = False
        
        # Cooldown kontrolü
        if self.is_panic_cooldown_active and (bar_index - self.panic_cooldown_start_bar >= self.panic_cooldown_bars):
            self.is_panic_cooldown_active = False
        
        if self.is_mini_loss_cooldown_active and (bar_index - self.mini_loss_cooldown_start_bar >= self.mini_loss_cooldown_bars):
            self.is_mini_loss_cooldown_active = False
        
        long_allowed = not self.is_panic_cooldown_active and not self.is_mini_loss_cooldown_active
        short_allowed = long_allowed
        
        self.check_signals(i, bar_index, current_price, long_allowed, short_allowed)
        self.process_entries(i, bar_index, current_price, long_allowed, short_allowed)
        self.process_dca(i, bar_index, current_price, long_allowed, short_allowed)
        self.process_exits(i, bar_index, current_price)
        self.update_trailing_stops(i, bar_index, current_price)
        
        self.mum_sayaci += 1
        if self.enable_mum_sayisi and self.mum_sayaci >= self.mum_sayisi:
            self.mum_sayaci = 0
            self.sinyal_verildi = False
    
    def check_signals(self, i, bar_index, current_price, long_allowed, short_allowed):
        """Sinyalleri kontrol et - V3'ün detaylı yaklaşımı"""
        close = self.df['close'].iloc[i]
        high = self.df['high'].iloc[i]
        low = self.df['low'].iloc[i]
        bb_lower = self.df['bb_lower'].iloc[i]
        bb_upper = self.df['bb_upper'].iloc[i]
        
        if pd.isna(bb_lower):
            return
        
        prev_step = self.prev_step
        prev_step_short = self.prev_step_short

        # === LONG STEP ===
        if self.step == 0 and close < bb_lower:
            self.step = 1
            self.reset_bar = bar_index

        elif self.step == 1 and self.df['stoch_rsi1_k'].iloc[i] <= self.df['stoch_rsi1_d'].iloc[i]:
            self.step = 2
            self.reset_bar = bar_index

        elif self.step == 2 and self.df['stoch_rsi2_k'].iloc[i] <= self.df['stoch_rsi2_d'].iloc[i]:
            self.step = 3
            self.reset_bar = bar_index

        elif self.step == 3 and self.df['williams_r'].iloc[i] <= -80:
            self.step = 4
            self.reset_bar = bar_index

        elif self.step == 4 and self.df['smoothed_rsi'].iloc[i] <= self.lower_band_level:
            self.step = 5
            self.reset_bar = bar_index

        elif self.step == 5 and self.df['fish1'].iloc[i] <= self.df['fish2'].iloc[i]:
            self.step = 6
            self.reset_bar = bar_index

        elif self.step == 6 and self.df['crsi'].iloc[i] <= 30:
            self.step = 7
            self.reset_bar = bar_index

        elif self.step == 7 and self.df['adx'].iloc[i] >= 20:
            self.step = 8
            self.reset_bar = bar_index

        elif self.step == 8 and self.df['macd_line'].iloc[i] < self.df['signal_line'].iloc[i]:
            self.step = 9
            self.reset_bar = bar_index

        # --- Pine Script'e birebir eşdeğer kısım ---
        # Mum kapanışında step == 9 olduğunda sadece 1 kez sinyal ver
        long_condition = False
        if self.step == 9 and (i == len(self.df) - 1 or self.df['bar_index'].iloc[i+1] != bar_index):
            # yani bar kapanmışsa (bir sonraki bar’a geçildiğinde)
            long_condition = True
            self.step = 0
            self.reset_bar = None
            self.long_signal_triggered_this_bar = True # Sinyal tetiklendi bayrağını ayarla
            # self.step = 0 # Sıfırlama işlemi process_entries'e taşındı
            # self.reset_bar = None # Sıfırlama işlemi process_entries'e taşındı

        # Timeout kontrolü (>= 10 bar geçmişse sıfırla)
        if self.reset_bar is not None and (bar_index - self.reset_bar >= 10):
            self.step = 0
            self.reset_bar = None




        # === SHORT STEP ===
        if self.step_short == 0 and close > bb_upper:
            self.step_short = 1
            self.reset_bar_short = bar_index

        elif self.step_short == 1 and self.df['stoch_rsi1_k'].iloc[i] >= self.df['stoch_rsi1_d'].iloc[i]:
            self.step_short = 2
            self.reset_bar_short = bar_index

        elif self.step_short == 2 and self.df['stoch_rsi2_k'].iloc[i] >= self.df['stoch_rsi2_d'].iloc[i]:
            self.step_short = 3
            self.reset_bar_short = bar_index

        elif self.step_short == 3 and self.df['williams_r'].iloc[i] >= -20:
            self.step_short = 4
            self.reset_bar_short = bar_index

        elif self.step_short == 4 and self.df['smoothed_rsi'].iloc[i] >= self.upper_band_level:
            self.step_short = 5
            self.reset_bar_short = bar_index

        elif self.step_short == 5 and self.df['fish1'].iloc[i] >= self.df['fish2'].iloc[i]:
            self.step_short = 6
            self.reset_bar_short = bar_index

        elif self.step_short == 6 and self.df['crsi'].iloc[i] >= 70:
            self.step_short = 7
            self.reset_bar_short = bar_index

        elif self.step_short == 7 and self.df['adx'].iloc[i] >= 20:
            self.step_short = 8
            self.reset_bar_short = bar_index

        elif self.step_short == 8 and self.df['macd_line'].iloc[i] > self.df['signal_line'].iloc[i]:
            self.step_short = 9
            self.reset_bar_short = bar_index

        # --- Pine Script mantığıyla kapanışta sinyal ---
        short_condition = False
        if self.step_short == 9 and (i == len(self.df) - 1 or self.df['bar_index'].iloc[i+1] != bar_index):
            short_condition = True
            self.step_short = 0
            self.reset_bar_short = None
            self.short_signal_triggered_this_bar = True # Sinyal tetiklendi bayrağını ayarla
            # self.step_short = 0 # Sıfırlama işlemi process_entries'e taşındı
            # self.reset_bar_short = None # Sıfırlama işlemi process_entries'e taşındı

        # Timeout kontrolü (>=10 bar geçtiyse sıfırla)
        if self.reset_bar_short is not None and (bar_index - self.reset_bar_short >= 10):
            self.step_short = 0
            self.reset_bar_short = None


        
        # Bullish Kosul
        current_bullish_kosul = False
        if not pd.isna(self.df['rsi_degeri'].iloc[i]) and not pd.isna(self.df['crsi'].iloc[i]) and not pd.isna(self.df['db'].iloc[i]):
            current_bullish_kosul = (
                self.df['rsi_degeri'].iloc[i] < self.rsi_taban_degeri and
                self.df['crsi'].iloc[i] < self.df['db'].iloc[i] and
                low < bb_lower
            )
        bullish_kosul = current_bullish_kosul and not self.prev_bullish_kosul
        self.prev_bullish_kosul = current_bullish_kosul

        # Bearish Kosul
        current_bearish_kosul = False
        if not pd.isna(self.df['rsi_degeri'].iloc[i]) and not pd.isna(self.df['crsi'].iloc[i]) and not pd.isna(self.df['ub'].iloc[i]):
            current_bearish_kosul = (
                self.df['rsi_degeri'].iloc[i] > self.rsi_tavan_degeri and
                self.df['crsi'].iloc[i] > self.df['ub'].iloc[i] and
                high > bb_upper
            )
        bearish_kosul = current_bearish_kosul and not self.prev_bearish_kosul
        self.prev_bearish_kosul = current_bearish_kosul

        # Long Approach
        current_long_approach = False
        if (not pd.isna(self.df['rsi_value2'].iloc[i]) and 
            not pd.isna(self.df['stoch_rsi1_k'].iloc[i]) and 
            not pd.isna(self.df['williams_r'].iloc[i]) and
            not pd.isna(self.df['sma200'].iloc[i]) and
            not pd.isna(self.df['ema1'].iloc[i]) and
            not pd.isna(self.df['obv_rsi'].iloc[i])):
            
            current_long_approach = (
                (self.df['rsi_value2'].iloc[i] < 30) and
                (self.df['stoch_rsi1_k'].iloc[i] - self.df['stoch_rsi1_d'].iloc[i] < 0.002) and
                (self.df['stoch_rsi2_k'].iloc[i] - self.df['stoch_rsi2_d'].iloc[i] < 0.002) and
                (close < bb_lower + (bb_lower * 0.01)) and
                (self.df['williams_r'].iloc[i] > -80) and
                (self.df['volume'].iloc[i] > 1000) and
                (close < self.df['sma200'].iloc[i]) and
                (close < self.df['ema1'].iloc[i]) and
                (self.df['obv_rsi'].iloc[i] < 35)
            )
        long_approach_condition = current_long_approach and not self.prev_long_approach
        self.prev_long_approach = current_long_approach

        # Short Approach
        current_short_approach = False
        if (not pd.isna(self.df['rsi_value2'].iloc[i]) and 
            not pd.isna(self.df['stoch_rsi1_k'].iloc[i]) and 
            not pd.isna(self.df['williams_r'].iloc[i]) and
            not pd.isna(self.df['sma200'].iloc[i]) and
            not pd.isna(self.df['ema1'].iloc[i]) and
            not pd.isna(self.df['obv_rsi'].iloc[i])):
            
            current_short_approach = (
                (self.df['rsi_value2'].iloc[i] > 70) and
                (self.df['stoch_rsi1_k'].iloc[i] - self.df['stoch_rsi1_d'].iloc[i] < 0.002) and
                (self.df['stoch_rsi2_k'].iloc[i] - self.df['stoch_rsi2_d'].iloc[i] < 0.002) and
                (close > bb_upper - (bb_upper * 0.01)) and
                (self.df['williams_r'].iloc[i] < -20) and
                (self.df['volume'].iloc[i] > 1000) and
                (close > self.df['sma200'].iloc[i]) and
                (close > self.df['ema1'].iloc[i]) and
                (self.df['obv_rsi'].iloc[i] > 65)
            )
        short_approach_condition = current_short_approach and not self.prev_short_approach
        self.prev_short_approach = current_short_approach

        # Long Alt
        current_long_alt = False
        if (not pd.isna(self.df['stoch_rsi1_k'].iloc[i]) and
            not pd.isna(self.df['williams_r'].iloc[i]) and
            not pd.isna(self.df['smoothed_rsi'].iloc[i]) and
            not pd.isna(self.df['rsi_value1'].iloc[i]) and
            not pd.isna(self.df['crsi'].iloc[i])):
            
            current_long_alt = (
                (close > bb_lower) and
                (self.df['stoch_rsi1_k'].iloc[i] >= self.df['stoch_rsi1_d'].iloc[i]) and
                (self.df['stoch_rsi2_k'].iloc[i] >= self.df['stoch_rsi2_d'].iloc[i]) and
                (self.df['williams_r'].iloc[i] <= -80) and
                (self.df['smoothed_rsi'].iloc[i] <= self.lower_band_level) and
                (self.df['rsi_value1'].iloc[i] <= 30) and
                (self.df['crsi'].iloc[i] <= 30)
            )
        long_condition1 = current_long_alt and not self.prev_long_alt
        self.prev_long_alt = current_long_alt
        
        # Short Alt
        current_short_alt = False
        if (not pd.isna(self.df['stoch_rsi1_k'].iloc[i]) and
            not pd.isna(self.df['williams_r'].iloc[i]) and
            not pd.isna(self.df['smoothed_rsi'].iloc[i]) and
            not pd.isna(self.df['rsi_value1'].iloc[i]) and
            not pd.isna(self.df['crsi'].iloc[i])):
            
            current_short_alt = (
                (close < bb_upper) and
                (self.df['stoch_rsi1_k'].iloc[i] <= self.df['stoch_rsi1_d'].iloc[i]) and
                (self.df['stoch_rsi2_k'].iloc[i] <= self.df['stoch_rsi2_d'].iloc[i]) and
                (self.df['williams_r'].iloc[i] >= -20) and
                (self.df['smoothed_rsi'].iloc[i] >= self.upper_band_level) and
                (self.df['rsi_value1'].iloc[i] >= 70) and
                (self.df['crsi'].iloc[i] >= 70)
            )
        short_condition1 = current_short_alt and not self.prev_short_alt
        self.prev_short_alt = current_short_alt
        
        # Sinyalleri kaydet
        if long_condition and self.use_long_step_entry:
            self.signals.append({
                'bar': i,
                'type': 'LONG_STEP',
                'price': close,
                'label': 'AL',
                'reason': '9-Step Long Complete'
            })
        
        if short_condition and self.use_short_step_entry:
            self.signals.append({
                'bar': i,
                'type': 'SHORT_STEP',
                'price': close,
                'label': 'SAT',
                'reason': '9-Step Short Complete'
            })
        
        if bullish_kosul and self.use_bullish_entry:
            self.signals.append({
                'bar': i,
                'type': 'BULLISH_OS',
                'price': close,
                'label': 'OS',
                'reason': 'Oversold Signal'
            })
        
        if bearish_kosul and self.use_bearish_entry:
            self.signals.append({
                'bar': i,
                'type': 'BEARISH_OB',
                'price': close,
                'label': 'OB',
                'reason': 'Overbought Signal'
            })
        
        if long_approach_condition and self.use_long_approach_entry:
            self.signals.append({
                'bar': i,
                'type': 'LONG_APPROACH',
                'price': close,
                'label': 'BOĞA',
                'reason': 'Bull Approach Signal'
            })
        
        if short_approach_condition and self.use_short_approach_entry:
            self.signals.append({
                'bar': i,
                'type': 'SHORT_APPROACH',
                'price': close,
                'label': 'AYI',
                'reason': 'Bear Approach Signal'
            })
        
        if long_condition1 and self.use_long_alt_entry:
            self.signals.append({
                'bar': i,
                'type': 'LONG_ALT',
                'price': close,
                'label': '🚀',
                'reason': 'Alternative Long'
            })
        
        if short_condition1 and self.use_short_alt_entry:
            self.signals.append({
                'bar': i,
                'type': 'SHORT_ALT',
                'price': close,
                'label': '🌗',
                'reason': 'Alternative Short'
            })
    
    def process_entries(self, i, bar_index, current_price, long_allowed, short_allowed):
        """Entry işlemleri"""
        close = self.df['close'].iloc[i]
        
        long_entry_signals = [s for s in self.signals if s['bar'] == i and s['type'] in ['LONG_STEP', 'BULLISH_OS', 'LONG_APPROACH', 'LONG_ALT']]
        short_entry_signals = [s for s in self.signals if s['bar'] == i and s['type'] in ['SHORT_STEP', 'BEARISH_OB', 'SHORT_APPROACH', 'SHORT_ALT']]
        
        # LONG GİRİŞ
        if self.long_islem_aktif and long_allowed and not self.exit_occurred_this_bar and len(long_entry_signals) > 0:
            if (not self.sinyal_verildi or (self.enable_mum_sayisi and self.mum_sayaci >= self.mum_sayisi)):
                
                if self.position_size < 0:
                    self.close_position(i, close, 'SHORT', 'Kısa Kapat')
                    self.short_signal_count = 0
                    self.next_short_size = self.min_lot
                    self.avg_short_price = None
                
                if self.long_signal_count == 0:
                    position_size_long = self.min_lot
                else:
                    position_size_long = min(self.next_long_size, self.max_qty)
                
                if position_size_long > 0:
                    entry_label = long_entry_signals[0]['label']
                    
                    self.open_position(i, bar_index, close, position_size_long, 'LONG', f'{self.long_signal_count + 1}. ENTER-LONG ({entry_label})')
                    
                    if self.long_signal_count == 0:
                        self.entry_bar_long = bar_index
                    
                    if self.avg_long_price is None:
                        self.avg_long_price = close
                    else:
                        total_size = abs(self.position_size) + position_size_long
                        self.avg_long_price = (self.avg_long_price * abs(self.position_size) + close * position_size_long) / total_size
                    
                    self.position_size += position_size_long
                    self.next_long_size = min(position_size_long * self.position_multiplier, self.max_qty)
                    self.long_signal_count += 1
                    self.sinyal_verildi = True
                    self.mum_sayaci = 0
                    
                    self.last_entry_price_long = close
                    self.last_entry_size_long = position_size_long
                    self.dca_count_long = 0
                    self.dca_bar_count_long = 0
                    
                    if self.enable_mini_stop:
                        if self.long_signal_count == self.min_entry_for_mini_loss and self.avg_long_price is not None:
                            self.can_draw_mini_loss = True
                        
                        if self.can_draw_mini_loss and self.mini_loss_level is None and self.position_size > 0:
                            self.mini_loss_level = self.avg_long_price * (1 - self.maliyet_stop_perc / 100)
        
                    # LONG_STEP sinyali bir pozisyon açtıysa step sayacını sıfırla
                    if self.long_signal_triggered_this_bar:
                        self.step = 0
                        self.reset_bar = None
                        self.long_signal_triggered_this_bar = False # Bayrağı sıfırla
        # SHORT GİRİŞ
        if self.short_islem_aktif and short_allowed and not self.exit_occurred_this_bar and len(short_entry_signals) > 0:
            if (not self.sinyal_verildi or (self.enable_mum_sayisi and self.mum_sayaci >= self.mum_sayisi)):
                
                if self.position_size > 0:
                    self.close_position(i, close, 'LONG', 'Long Kapat')
                    self.long_signal_count = 0
                    self.next_long_size = self.min_lot
                    self.avg_long_price = None
                
                if self.short_signal_count == 0:
                    position_size_short = self.min_lot
                else:
                    position_size_short = min(self.next_short_size, self.max_qty)
                
                if position_size_short > 0:
                    entry_label = short_entry_signals[0]['label']
                    
                    self.open_position(i, bar_index, close, -position_size_short, 'SHORT', f'{self.short_signal_count + 1}. ENTER-SHORT ({entry_label})')
                    
                    if self.short_signal_count == 0:
                        self.entry_bar_short = bar_index
                    
                    if self.avg_short_price is None:
                        self.avg_short_price = close
                    else:
                        total_size = abs(self.position_size) + position_size_short
                        self.avg_short_price = (self.avg_short_price * abs(self.position_size) + close * position_size_short) / total_size
                    
                    self.position_size -= position_size_short
                    self.next_short_size = min(position_size_short * self.position_multiplier, self.max_qty)
                    self.short_signal_count += 1
                    self.sinyal_verildi = True
                    self.mum_sayaci = 0
                    
                    self.last_entry_price_short = close
                    self.last_entry_size_short = position_size_short
                    self.dca_count_short = 0
                    self.dca_bar_count_short = 0

                    # SHORT_STEP sinyali pozisyon açtıysa step sayacını sıfırla
                    if self.short_signal_triggered_this_bar:
                        self.step_short = 0
                        self.reset_bar_short = None
                        self.short_signal_triggered_this_bar = False
    
    
    def process_dca(self, i, bar_index, current_price, long_allowed, short_allowed):
        """DCA işlemleri"""
        close = self.df['close'].iloc[i]
        
        # LONG DCA
        if self.enable_dca and long_allowed and self.position_size > 0 and self.last_entry_price_long is not None and self.dca_count_long < self.max_dca:
            self.dca_bar_count_long += 1
            
            if not self.enable_dca_mum_delay or self.dca_bar_count_long >= self.dca_mum_delay:
                drop_from_last = (self.avg_long_price - close) / self.avg_long_price
                
                if drop_from_last >= self.dca_drop_perc:
                    self.open_position(i, bar_index, close, self.last_entry_size_long, 'LONG', f'{self.long_signal_count + 1}. DCA-LONG')
                    
                    total_size = abs(self.position_size) + self.last_entry_size_long
                    self.avg_long_price = (self.avg_long_price * abs(self.position_size) + close * self.last_entry_size_long) / total_size
                    
                    self.position_size += self.last_entry_size_long
                    self.last_entry_price_long = close
                    self.dca_count_long += 1
                    self.long_signal_count += 1
                    self.dca_bar_count_long = 0
        
        # SHORT DCA
        if self.enable_dca and short_allowed and self.position_size < 0 and self.last_entry_price_short is not None and self.dca_count_short < self.max_dca:
            self.dca_bar_count_short += 1
            
            if not self.enable_dca_mum_delay or self.dca_bar_count_short >= self.dca_mum_delay:
                rise_from_last = (close - self.avg_short_price) / self.avg_short_price
                
                if rise_from_last >= self.dca_drop_perc:
                    self.open_position(i, bar_index, close, -self.last_entry_size_short, 'SHORT', f'{self.short_signal_count + 1}. DCA-SHORT')
                    
                    total_size = abs(self.position_size) + self.last_entry_size_short
                    self.avg_short_price = (self.avg_short_price * abs(self.position_size) + close * self.last_entry_size_short) / total_size
                    
                    self.position_size -= self.last_entry_size_short
                    self.last_entry_price_short = close
                    self.dca_count_short += 1
                    self.short_signal_count += 1
                    self.dca_bar_count_short = 0
    
    def process_exits(self, i, bar_index, current_price):
        """Exit işlemleri - Pine Script mantığı"""
        close = self.df['close'].iloc[i]
        high = self.df['high'].iloc[i]
        low = self.df['low'].iloc[i]
        
        exit_reason = ""
        
        # ==================== LONG EXIT ====================
        if self.position_size > 0 and self.avg_long_price is not None:
            
            # 1) Take Profit
            if self.enable_take_profit and self.tp_type == "Fixed":
                tp_level_long = self.avg_long_price * (1 + self.take_profit_perc / 100)
                if high >= tp_level_long:
                    exit_reason = "TP"
            
            # 1.5) Trailing Sonrası Ekstra Take Profit (YENİ EKLENDİ)
            if exit_reason == "" and self.enable_extra_tp_after_ts and self.ts_active_long and self.trail_stop_long is not None:
                # TP seviyesi, trailing'in aktif olduğu andaki fiyata göre değil, güncel trailing stop seviyesine göre hesaplanır.
                # Bu, fiyat yükseldikçe TP'nin de dinamik olarak yükselmesini sağlar.
                extra_tp_level = self.trail_stop_long / (1 - self.trailing_perc) * (1 + self.extra_tp_perc / 100)
                if high >= extra_tp_level:
                    exit_reason = "Extra TP"
            
            # 2) Trailing Stop Trigger
            short_close_signals = [s for s in self.signals if s['bar'] == i and s['type'] in ['SHORT_STEP', 'BEARISH_OB', 'SHORT_APPROACH', 'SHORT_ALT']]
            
            if exit_reason == "" and len(short_close_signals) > 0:
                if self.enable_trailing_stop and self.entry_bar_long is not None and bar_index > self.entry_bar_long and self.long_signal_count >= self.min_position_for_ts:
                    if not self.ts_active_long:
                        self.ts_active_long = True
                        self.trail_stop_long = close * (1 - self.trailing_perc)
                elif not self.enable_trailing_stop:
                    exit_reason = "Signal - No trailing"
            
            # 3) Trailing Stop Kontrol
            trailing_hit_long = self.enable_trailing_stop and self.ts_active_long and self.trail_stop_long is not None and close <= self.trail_stop_long
            if exit_reason == "" and trailing_hit_long:
                exit_reason = "Trailing Stop"
            
            # 4) Maliyet Çıkışı
            if exit_reason == "" and self.enable_maliyet_exit and self.long_signal_count >= self.min_entry_for_maliyet and not self.ts_active_long:
                maliyet_level = self.avg_long_price * (1 + self.maliyet_return_perc / 100)
                if high >= maliyet_level:
                    exit_reason = "Maliyet"
            
            # 5) MiniLoss
            if exit_reason == "" and self.enable_mini_stop and self.long_signal_count >= self.min_entry_for_mini_loss and self.position_size > 0:
                if self.mini_loss_level is not None and low <= self.mini_loss_level:
                    exit_reason = "MiniLoss"
            
            # 6) Whale Panic
            if exit_reason == "" and self.check_whale_panic(i):
                exit_reason = "Whale Panic"
            
            # TEK KAPAMA
            if exit_reason != "":
                if exit_reason == "MiniLoss":
                    self.is_mini_loss_cooldown_active = True
                    self.mini_loss_cooldown_start_bar = bar_index
                
                if exit_reason == "Whale Panic":
                    self.is_panic_cooldown_active = True
                    self.panic_cooldown_start_bar = bar_index
                
                self.close_position(i, close, 'LONG', f'EXIT LONG ({exit_reason})')
                self.exit_occurred_this_bar = True
                self.long_signal_count = 0 # Reset
                self.next_long_size = self.calculate_qty_from_usd(self.calculate_trade_amount_usd(), close)
                self.avg_long_price = None
                self.entry_bar_long = None
                self.ts_active_long = False
                self.trail_stop_long = None
                self.plot_trail_stop_long = None
                self.mini_loss_level = None
                self.can_draw_mini_loss = False
                self.position_size = 0
        
        # ==================== SHORT EXIT ====================
        if self.position_size < 0 and self.avg_short_price is not None:
            
            # 1) Take Profit
            if self.enable_take_profit and self.tp_type == "Fixed":
                tp_level_short = self.avg_short_price * (1 - self.take_profit_perc / 100)
                if low <= tp_level_short:
                    exit_reason = "TP"
            
            # 2) Trailing Stop Trigger
            long_close_signals = [s for s in self.signals if s['bar'] == i and s['type'] in ['LONG_STEP', 'BULLISH_OS', 'LONG_APPROACH', 'LONG_ALT']]
            
            if exit_reason == "" and len(long_close_signals) > 0:
                if self.enable_trailing_stop and self.entry_bar_short is not None and bar_index > self.entry_bar_short and self.short_signal_count >= self.min_position_for_ts:
                    if not self.ts_active_short:
                        self.ts_active_short = True
                        self.trail_stop_short = close * (1 + self.trailing_perc)
                elif not self.enable_trailing_stop:
                    exit_reason = "Signal - No trailing"
            
            # 3) Trailing Stop Kontrol
            trailing_hit_short = self.enable_trailing_stop and self.ts_active_short and self.trail_stop_short is not None and close >= self.trail_stop_short
            if exit_reason == "" and trailing_hit_short:
                exit_reason = "Trailing Stop"
            
            # 4) Maliyet Çıkışı
            if exit_reason == "" and self.enable_maliyet_exit and self.short_signal_count >= self.min_entry_for_maliyet and not self.ts_active_short:
                maliyet_level_s = self.avg_short_price * (1 - self.maliyet_return_perc / 100)
                if low <= maliyet_level_s:
                    exit_reason = "Maliyet"
            
            # 5) MiniLoss
            if exit_reason == "" and self.enable_mini_stop and self.short_signal_count >= self.min_entry_for_mini_loss and self.position_size < 0:
                short_mini_level = self.avg_short_price * (1 + self.maliyet_stop_perc / 100)
                if close >= short_mini_level:
                    exit_reason = "MiniLoss"
            
            # 6) Whale Panic
            if exit_reason == "" and self.check_whale_panic(i):
                exit_reason = "Whale Panic"
            
            # TEK KAPAMA
            if exit_reason != "":
                if exit_reason == "MiniLoss":
                    self.is_mini_loss_cooldown_active = True
                    self.mini_loss_cooldown_start_bar = bar_index
                
                if exit_reason == "Whale Panic":
                    self.is_panic_cooldown_active = True
                    self.panic_cooldown_start_bar = bar_index
                
                self.close_position(i, close, 'SHORT', f'EXIT SHORT ({exit_reason})')
                self.exit_occurred_this_bar = True
                self.short_signal_count = 0 # Reset
                self.next_short_size = self.calculate_qty_from_usd(self.calculate_trade_amount_usd(), close)
                self.avg_short_price = None
                self.entry_bar_short = None
                self.ts_active_short = False
                self.trail_stop_short = None
                self.plot_trail_stop_short = None
                self.position_size = 0
    
    def update_trailing_stops(self, i, bar_index, current_price):
        """Trailing stop güncelle - Pine Script mantığı"""
        close = self.df['close'].iloc[i]
        high = self.df['high'].iloc[i]
        low = self.df['low'].iloc[i]
        
        # LONG TRAILING
        if self.position_size > 0 and self.enable_trailing_stop:
            min_pos_check_long = not self.enable_min_position_for_ts or self.long_signal_count >= self.min_position_for_ts
            
            # Kapanma sinyali geldi mi?
            short_close_signals = [s for s in self.signals if s['bar'] == i and s['type'] in ['SHORT_STEP', 'BEARISH_OB', 'SHORT_APPROACH', 'SHORT_ALT']]
            
            if len(short_close_signals) > 0:
                if min_pos_check_long:
                    if not self.ts_active_long and self.trail_stop_long is None:
                        self.ts_active_long = True
                        self.trail_stop_long = close * (1 - self.trailing_perc)
                else:
                    # min pozisyon sağlanmıyorsa direkt kapat
                    self.close_position(i, close, 'LONG', 'Long EXIT (MinPos Not Met)')
                    self.long_signal_count = 0
                    self.avg_long_price = None
                    self.ts_active_long = False
                    self.trail_stop_long = None
                    self.plot_trail_stop_long = None
                    self.position_size = 0
            
            # Trailing aktifse
            if self.ts_active_long and self.trail_stop_long is not None:
                self.trail_stop_long = max(self.trail_stop_long, close * (1 - self.trailing_perc))
                self.plot_trail_stop_long = self.trail_stop_long

                if low <= self.trail_stop_long:
                    self.close_position(i, self.trail_stop_long, 'LONG', 'TS LONG EXIT')
                    self.long_signal_count = 0
                    self.avg_long_price = None
                    self.ts_active_long = False
                    self.trail_stop_long = None
                    self.plot_trail_stop_long = None
                    self.position_size = 0
        else:
            self.plot_trail_stop_long = None
        
        # SHORT TRAILING
        if self.position_size < 0 and self.enable_trailing_stop:
            min_pos_check_short = not self.enable_min_position_for_ts or self.short_signal_count >= self.min_position_for_ts
            
            # Kapanma sinyali geldi mi?
            long_close_signals = [s for s in self.signals if s['bar'] == i and s['type'] in ['LONG_STEP', 'BULLISH_OS', 'LONG_APPROACH', 'LONG_ALT']]
            
            if len(long_close_signals) > 0:
                if min_pos_check_short:
                    if not self.ts_active_short and self.trail_stop_short is None:
                        self.ts_active_short = True
                        self.trail_stop_short = close * (1 + self.trailing_perc)
                else:
                    # min pozisyon sağlanmıyorsa direkt kapat
                    self.close_position(i, close, 'SHORT', 'Short EXIT (MinPos Not Met)')
                    self.short_signal_count = 0
                    self.avg_short_price = None
                    self.ts_active_short = False
                    self.trail_stop_short = None
                    self.plot_trail_stop_short = None
                    self.position_size = 0
            
            # Trailing aktifse
            if self.ts_active_short and self.trail_stop_short is not None:
                self.trail_stop_short = min(self.trail_stop_short, close * (1 + self.trailing_perc))
                self.plot_trail_stop_short = self.trail_stop_short

                if high >= self.trail_stop_short:
                    self.close_position(i, self.trail_stop_short, 'SHORT', 'TS SHORT EXIT')
                    self.short_signal_count = 0
                    self.avg_short_price = None
                    self.ts_active_short = False
                    self.trail_stop_short = None
                    self.plot_trail_stop_short = None
                    self.position_size = 0
        else:
            self.plot_trail_stop_short = None
    
    def check_whale_panic(self, i):
        """Whale Panic kontrolü - Pine Script mantığı"""
        if i < max(self.drop_speed_bars, self.panic_drop_bars):
            return False
        
        close = self.df['close'].iloc[i]
        
        # Drop Speed
        if self.enable_drop_speed and i >= self.drop_speed_bars:
            close_prev = self.df['close'].iloc[i - self.drop_speed_bars]
            if close < close_prev * (1 - self.drop_speed_perc / 100):
                return True
        
        # Panic Drop
        if self.enable_panic_drop and i >= self.panic_drop_bars:
            close_prev = self.df['close'].iloc[i - self.panic_drop_bars]
            if close < close_prev * (1 - self.panic_drop_perc / 100):
                return True
        
        return False
    
    def open_position(self, bar, bar_index, price, size, direction, comment):
        """Pozisyon aç"""
        self.trades.append({
            'bar': bar,
            'bar_index': bar_index,
            'type': 'ENTRY',
            'direction': direction,
            'price': price,
            'size': abs(size),
            'comment': comment,
            'timestamp': self.df.index[bar] if 'timestamp' not in self.df.columns else self.df['timestamp'].iloc[bar]
        })
    
    def close_position(self, bar, price, direction, comment):
        """Pozisyonu kapat"""
        if self.position_size != 0:
            self.trades.append({
                'bar': bar,
                'type': 'EXIT',
                'direction': direction,
                'price': price,
                'size': abs(self.position_size),
                'comment': comment,
                'timestamp': self.df.index[bar] if 'timestamp' not in self.df.columns else self.df['timestamp'].iloc[bar]
            })
            
    def calculate_trade_amount_usd(self):
        """Hesaplanan trade miktarını USD olarak döndürür."""
        lev = max(1, self.leverage) if self.enable_leverage else 1
        return (self.min_lot * self.manual_lot_usd) / lev

    def calculate_qty_from_usd(self, amount_usd, price):
        """USD miktarını ve fiyatı kullanarak lot miktarını hesaplar."""
        if price == 0: return 0
        return amount_usd / price




# ==================== VİZUALİZER CLASS (V3'TEN TAM AYNI) ====================
class TradingVisualizer:
    """Pine Script görsellerine %100 uyumlu görselleştirme"""
    
    def __init__(self, df, trades, signals):
        self.df = df.copy()
        self.trades = trades
        self.signals = signals
        
        # Tarih formatını kontrol et ve datetime nesnesine çevir
        if 'datetime' in self.df.columns and not pd.api.types.is_datetime64_any_dtype(self.df['datetime']):
            self.df['datetime'] = pd.to_datetime(self.df['datetime'])
    
    def create_interactive_chart(self, output_file='whale_strategy_v5.html', show_browser=True):
        """Pine Script görsellerine uyumlu detaylı grafik"""
        
        print("\n" + "="*70)
        print("🎨 PINE SCRIPT UYUMLU GRAFİK OLUŞTURULUYOR")
        print("="*70)
        
        fig = make_subplots(
            rows=1, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            subplot_titles=('💹 Fiyat & Sinyaller',)
        )
        
        # Candlestick
        fig.add_trace(
            go.Candlestick(
                x=self.df['datetime'],
                open=self.df['open'],
                high=self.df['high'],
                low=self.df['low'],
                close=self.df['close'],
                name='Price',
                increasing_line_color='#26a69a',
                decreasing_line_color='#ef5350',
                increasing_fillcolor='#26a69a',
                decreasing_fillcolor='#ef5350'
            ),
            row=1, col=1
        )
        
        # Bollinger Bands
        if 'bb_upper' in self.df.columns:
            fig.add_trace(
                go.Scatter(
                    x=self.df['datetime'],
                    y=self.df['bb_lower'],
                    name='BB Lower',
                    line=dict(color='rgba(33, 150, 243, 0.3)', width=1),
                    fill='tonexty',
                    fillcolor='rgba(33, 150, 243, 0.05)',
                    showlegend=False
                ),
                row=1, col=1
            )
        
        # Sinyaller
        signal_configs = {
            'LONG_STEP': {
                'color': 'green',
                'symbol': 'triangle-up',
                'text': 'AL',
                'size': 12,
                'position': 'bottom center'
            },
            'SHORT_STEP': {
                'color': 'red',
                'symbol': 'triangle-down',
                'text': 'SAT',
                'size': 12,
                'position': 'top center'
            },
            'BULLISH_OS': {
                'color': 'rgb(68, 255, 249)',
                'symbol': 'triangle-up',
                'text': 'OS',
                'size': 10,
                'position': 'bottom center'
            },
            'BEARISH_OB': {
                'color': 'rgb(195, 0, 0)',
                'symbol': 'triangle-down',
                'text': 'OB',
                'size': 10,
                'position': 'top center'
            },
            'LONG_APPROACH': {
                'color': 'rgb(103, 160, 240)',
                'symbol': 'circle',
                'text': 'BOĞA',
                'size': 8,
                'position': 'bottom center'
            },
            'SHORT_APPROACH': {
                'color': 'rgb(237, 230, 44)',
                'symbol': 'circle',
                'text': 'AYI',
                'size': 8,
                'position': 'top center'
            },
            'LONG_ALT': {
                'color': 'blue',
                'symbol': 'circle',
                'text': '🚀',
                'size': 10,
                'position': 'bottom center'
            },
            'SHORT_ALT': {
                'color': 'red',
                'symbol': 'circle',
                'text': '🌗',
                'size': 10,
                'position': 'top center'
            }
        }
        
        # Sinyalleri grupla ve çiz
        signal_types = {}
        for signal in self.signals:
            signal_type = signal['type']
            if signal_type not in signal_types:
                signal_types[signal_type] = []
            signal_types[signal_type].append(signal)
        
        for signal_type, signals_list in signal_types.items():
            if signal_type in signal_configs:
                config = signal_configs[signal_type]
                
                bars = [self.df['datetime'].iloc[s['bar']] for s in signals_list]
                prices = [self.df['low'].iloc[s['bar']] if 'LONG' in s['type'] else self.df['high'].iloc[s['bar']] for s in signals_list]
                reasons = [s.get('reason', 'N/A') for s in signals_list]
                
                if not bars or not prices:
                    continue
                
                fig.add_trace(
                    go.Scatter(
                        x=bars,
                        y=prices,
                        mode='markers+text',
                        name=config['text'],
                        marker=dict(
                            symbol=config['symbol'],
                            size=config['size'],
                            color=config['color'],
                            line=dict(width=1, color='white')
                        ),
                        text=[config['text']] * len(signals_list),
                        textposition=config['position'],
                        textfont=dict(size=8, color='black'),
                        hovertemplate='<b>' + config['text'] + '</b><br>' +
                                    'Bar: %{x}<br>' +
                                    'Price: $%{y:.4f}<br>' +
                                    'Reason: %{customdata}<br>' +
                                    '<extra></extra>',
                        customdata=reasons
                    ),
                    row=1, col=1
                )
        
        # Entry/Exit işaretleri
        entries = [t for t in self.trades if t['type'] == 'ENTRY']
        exits = [t for t in self.trades if t['type'] == 'EXIT']
        
        for entry in entries:
            direction = entry['direction']
            bar = entry['bar']
            price = entry['price']
            comment = entry['comment']
            
            if direction == 'LONG':
                fig.add_annotation(
                    x=self.df['datetime'].iloc[bar],
                    y=price * 0.998,
                    text=comment,
                    showarrow=True,
                    arrowhead=2,
                    arrowsize=1,
                    arrowwidth=2,
                    arrowcolor='green',
                    ax=0,
                    ay=30,
                    bgcolor='rgba(0, 255, 0, 0.2)',
                    bordercolor='green',
                    borderwidth=1,
                    font=dict(size=8, color='darkgreen'),
                    row=1, col=1
                )
            else:
                fig.add_annotation(
                    x=self.df['datetime'].iloc[bar],
                    y=price * 1.002,
                    text=comment,
                    showarrow=True,
                    arrowhead=2,
                    arrowsize=1,
                    arrowwidth=2,
                    arrowcolor='red',
                    ax=0,
                    ay=-30,
                    bgcolor='rgba(255, 0, 0, 0.2)',
                    bordercolor='red',
                    borderwidth=1,
                    font=dict(size=8, color='darkred'),
                    row=1, col=1
                )
        
        for exit_trade in exits:
            direction = exit_trade['direction']
            bar = exit_trade['bar']
            price = exit_trade['price']
            comment = exit_trade['comment']
            
            if direction == 'LONG':
                fig.add_annotation(
                    x=self.df['datetime'].iloc[bar],
                    y=price * 1.002,
                    text=comment,
                    showarrow=True,
                    arrowhead=2,
                    arrowsize=1,
                    arrowwidth=2,
                    arrowcolor='orange',
                    ax=0,
                    ay=-30,
                    bgcolor='rgba(255, 165, 0, 0.2)',
                    bordercolor='orange',
                    borderwidth=1,
                    font=dict(size=8, color='darkorange'),
                    row=1, col=1
                )
            else:
                fig.add_annotation(
                    x=self.df['datetime'].iloc[bar],
                    y=price * 0.998,
                    text=comment,
                    showarrow=True,
                    arrowhead=2,
                    arrowsize=1,
                    arrowwidth=2,
                    arrowcolor='orange',
                    ax=0,
                    ay=30,
                    bgcolor='rgba(255, 165, 0, 0.2)',
                    bordercolor='orange',
                    borderwidth=1,
                    font=dict(size=8, color='darkorange'),
                    row=1, col=1
                )
        
        # Layout
        fig.update_layout(
            title={
                'text': '🐋 Whale Stop Strategy V5 - İyileştirilmiş Hesaplama + Tam Görselleştirme',
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 22, 'color': '#1565C0', 'family': 'Arial Black'}
            },
            height=800,
            showlegend=True,
            xaxis_rangeslider_visible=False,
            hovermode='x unified',
            template='plotly_white',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor="gray",
                borderwidth=1
            ),
            font=dict(family="Arial", size=10)
        )
        
        fig.update_xaxes(title_text="Tarih", row=1, col=1, showgrid=True, gridwidth=0.5, gridcolor='lightgray')
        fig.update_yaxes(title_text="Price (USDT)", row=1, col=1, showgrid=True, gridwidth=0.5, gridcolor='lightgray')
        
        fig.write_html(output_file)
        print(f"\n✅ Grafik kaydedildi: {output_file}")
        
        self.print_performance_summary()
        
        if show_browser:
            import webbrowser
            import os
            webbrowser.open('file://' + os.path.realpath(output_file))
            print(f"🌐 Grafik tarayıcınızda açıldı!")
        
        print("="*70)
        
        return fig
    
    def print_performance_summary(self):
        """Performans özetini yazdır"""
        entries = [t for t in self.trades if t['type'] == 'ENTRY']
        exits = [t for t in self.trades if t['type'] == 'EXIT']
        
        print(f"\n📊 PERFORMANS ÖZETİ:")
        print(f"   • Toplam Sinyal: {len(self.signals)}")
        print(f"   • Toplam Entry: {len(entries)}")
        print(f"   • Toplam Exit: {len(exits)}")
        
        signal_types = {}
        for signal in self.signals:
            stype = signal['type']
            signal_types[stype] = signal_types.get(stype, 0) + 1
        
        print(f"\n   📍 Sinyal Dağılımı:")
        for stype, count in sorted(signal_types.items()):
            print(f"      • {stype}: {count}")
        
        if len(exits) > 0:
            exit_reasons = {}
            for exit_trade in exits:
                reason = exit_trade['comment']
                exit_reasons[reason] = exit_reasons.get(reason, 0) + 1
            
            print(f"\n   🚪 Exit Nedenleri:")
            for reason, count in sorted(exit_reasons.items()):
                print(f"      • {reason}: {count}")


def precompute_indicators(df):
    """Tüm indikatörleri bir kerede hesaplayıp DataFrame'e ekler."""
    print("\n" + "="*70)
    print("📊 ÖN HESAPLAMA: Tüm indikatörler hesaplanıyor...")
    print("="*70)
    
    # Geçici bir strateji nesnesi oluşturarak parametreleri ve hesaplama fonksiyonlarını kullan
    temp_strategy = WhaleStopStrategy(df)
    
    df_out = df.copy()

    df_out['bb_basis'], df_out['bb_upper'], df_out['bb_lower'] = temp_strategy.calculate_bollinger_bands()
    print("   ✅ Bollinger Bands")
    
    df_out['stoch_rsi1_k'], df_out['stoch_rsi1_d'], df_out['rsi_value1'] = temp_strategy.calculate_stochastic_rsi(
        temp_strategy.stoch_rsi1_rsi_length, temp_strategy.stoch_rsi1_stoch_length, 
        temp_strategy.stoch_rsi1_k_length, temp_strategy.stoch_rsi1_d_length
    )
    print("   ✅ Stochastic RSI 1")
    
    df_out['stoch_rsi2_k'], df_out['stoch_rsi2_d'], df_out['rsi_value2'] = temp_strategy.calculate_stochastic_rsi(
        temp_strategy.stoch_rsi2_rsi_length, temp_strategy.stoch_rsi2_stoch_length,
        temp_strategy.stoch_rsi2_k_length, temp_strategy.stoch_rsi2_d_length
    )
    print("   ✅ Stochastic RSI 2")
    
    df_out['williams_r'] = temp_strategy.calculate_williams_r()
    print("   ✅ Williams %R")
    
    df_out['smoothed_rsi'] = temp_strategy.calculate_smoothed_rsi()
    print("   ✅ Smoothed RSI")
    
    df_out['adx'] = temp_strategy.calculate_adx()
    print("   ✅ ADX")
    
    df_out['crsi'], df_out['db'], df_out['ub'] = temp_strategy.calculate_cyclic_rsi()
    print("   ✅ Cyclic RSI")
    
    df_out['fish1'], df_out['fish2'] = temp_strategy.calculate_fisher_transform()
    print("   ✅ Fisher Transform")
    
    df_out['macd_line'], df_out['signal_line'], df_out['macd_histogram'] = temp_strategy.calculate_macd_dema()
    print("   ✅ MACD DEMA")
    
    df_out['rsi_degeri'] = temp_strategy.ta_rsi(df_out['close'], temp_strategy.rsi_periodu)
    print("   ✅ RSI")
    
    df_out['sma200'] = df_out['close'].rolling(window=50).mean()
    df_out['ema1'] = df_out['close'].ewm(span=14, adjust=False).mean()
    print("   ✅ SMA & EMA")
    
    df_out['obv_rsi'] = temp_strategy.calculate_obv_rsi()
    print("   ✅ OBV RSI")
    
    print("\n✅ Tüm indikatörler önceden hesaplandı ve hazır!")
    return df_out









# ==================== KULLANIM ====================
def run_whale_strategy(csv_file='1000SHIB_5d_verisi.csv'):
    """Whale Strategy'yi CSV dosyasından çalıştırır"""
    print("\n" + "="*70)
    print("☄️ QUANTUM THE SENTİNEL ☄️")
    print("="*70)
    
    try:
        print(f"\n📂 CSV dosyası okunuyor: {csv_file}")
        df = pd.read_csv(csv_file)
        print(f"✅ Dosya yüklendi: {len(df)} satır")

        df.columns = [c.lower().strip() for c in df.columns]
        required = ['open', 'high', 'low', 'close', 'volume']
        for r in required:
            if r not in df.columns:
                raise ValueError(f"'{r}' sütunu eksik!")

        strategy = WhaleStopStrategy(df)
        df_result, trades, signals = strategy.run_strategy()

        print("\n" + "="*70)
        print("✅ ANALİZ BAŞARIYLA TAMAMLANDI!")
        print("="*70)

        return df_result, trades, signals

    except Exception as e:
        print(f"\n❌ HATA: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None


if __name__ == "__main__":
    df_result, trades, signals = run_whale_strategy('1000SHIB_5d_verisi.csv')
    
    if df_result is not None and trades is not None and signals is not None:
        print("\nGrafik oluşturuluyor...")
        visualizer = TradingVisualizer(df_result, trades, signals)
        visualizer.create_interactive_chart(show_browser=True)














# =============================================================================
# WHALE STRATEGY - OPTUNA OPTİMİZASYON SİSTEMİ (TAMAMEN DÜZELTİLMİŞ)
# =============================================================================

import optuna
import json
import traceback
import sys
import webbrowser
import os
from datetime import datetime
import numpy as np
import pandas as pd
from pandas.tseries.offsets import DateOffset
from pathlib import Path
import optuna.visualization as vis
import plotly.express as px
import plotly.graph_objects as go
import logging

# Optuna'nın çok fazla log basmasını engelle
optuna.logging.set_verbosity(optuna.logging.WARNING)

# =============================================================================
# GÖRSELLEŞTİRME SINIFI
# =============================================================================
class TradingVisualizer:
    """
    Gerçek işlem geçmişini grafik üzerine çizen sınıf.
    """
    def __init__(self, df, trades, signals=None):
        self.df = df
        self.trades = trades
        # signals parametresi eklendi (uyumluluk için)
        self.signals = signals

    def create_interactive_chart(self, filename="en_iyi_strateji.html", show_browser=True):
        print(f"📈 Grafik oluşturuluyor: {filename} ...")
        
        # 1. Mum Grafiği (Candlestick)
        fig = go.Figure(data=[go.Candlestick(
            x=self.df.index,
            open=self.df['open'],
            high=self.df['high'],
            low=self.df['low'],
            close=self.df['close'],
            name='Fiyat'
        )])

        # 2. Alış ve Satış İşaretleri
        buy_x = []
        buy_y = []
        sell_x = []
        sell_y = []
        
        for t in self.trades:
            # Datetime kontrolü (DataFrame indexinden al)
            dt = t.get('datetime')
            if dt is None and 'bar' in t:
                dt = self.df.index[t['bar']]

            if t['type'] == 'ENTRY':
                buy_x.append(dt)
                buy_y.append(t['price'])
            elif t['type'] == 'EXIT':
                sell_x.append(dt)
                sell_y.append(t['price'])

        # Alış İşaretleri (Yeşil Yukarı Ok)
        if buy_x:
            fig.add_trace(go.Scatter(
                x=buy_x, y=buy_y,
                mode='markers',
                name='Giriş (Entry)',
                marker=dict(symbol='triangle-up', size=12, color='green', line=dict(width=2, color='black'))
            ))

        # Satış İşaretleri (Kırmızı Aşağı Ok)
        if sell_x:
            fig.add_trace(go.Scatter(
                x=sell_x, y=sell_y,
                mode='markers',
                name='Çıkış (Exit)',
                marker=dict(symbol='triangle-down', size=12, color='red', line=dict(width=2, color='black'))
            ))

        # 3. Ayarlar ve Kaydetme
        fig.update_layout(
            title='En İyi Strateji Sonucu - İşlem Yerleri',
            yaxis_title='Fiyat',
            xaxis_title='Tarih',
            template='plotly_dark',
            xaxis_rangeslider_visible=False
        )
        
        fig.write_html(filename)
        print(f"✅ Grafik kaydedildi: {filename}")

        if show_browser:
            try:
                path = os.path.abspath(filename)
                webbrowser.open(f'file://{path}')
                print("🌍 Tarayıcı otomatik açılıyor...")
            except Exception as e:
                print("Otomatik açma başarısız oldu, klasörden elle açabilirsin.")

# =============================================================================
# GELİŞTİRİLMİŞ ISTATISTIK HESAPLAMA FONKSİYONU (DÜZELTİLDİ)
# =============================================================================

def calculate_backtest_stats(trades, initial_equity=10000):
    """
    Backtest istatistiklerini, her işlemin yatırım maliyetine göre getirisini
    hesaplayarak ve bileşik hale getirerek daha gerçekçi bir şekilde ölçer.
    """
    equity = initial_equity
    peak_equity = initial_equity
    max_drawdown = 0.0
    wins = 0
    losses = 0
    closed_trades = []
    
    # Pozisyon durumu (Long ve Short için ortak)
    position_active = False
    position_direction = None
    position_size = 0.0
    position_cost = 0.0
    avg_entry_price = 0.0

    for trade in trades:
        trade_type = trade.get('type')
        price = trade.get('price')
        size = float(trade.get('size', 0.0))
        direction = trade.get('direction')
        
        if not price or price <= 0 or size <= 0.0:
            continue

        if trade_type == 'ENTRY':
            # Eğer ters yönde bir pozisyon varsa, önce onu kapat (bu senaryo normalde olmamalı ama güvenlik için)
            if position_active and direction != position_direction:
                # Bu durumu şimdilik görmezden geliyoruz, strateji mantığı bunu engellemeli.
                pass

            new_cost = price * size
            total_cost = position_cost + new_cost
            total_size = position_size + size
            
            if total_size > 0:
                avg_entry_price = total_cost / total_size
            
            position_size = total_size
            position_cost = total_cost
            position_active = True
            position_direction = direction

        elif trade_type == 'EXIT':
            if not position_active or direction != position_direction:
                continue # Eşleşmeyen bir çıkış, atla

            pnl = 0.0
            if direction == 'LONG':
                pnl = (price - avg_entry_price) * position_size
            elif direction == 'SHORT':
                pnl = (avg_entry_price - price) * position_size

            # En önemli değişiklik: Getiri oranını pozisyon maliyetine göre hesapla
            return_on_investment = (pnl / position_cost) if position_cost > 0 else 0.0

            closed_trades.append({
                'pnl': pnl,
                'return_on_investment': return_on_investment
            })

            # İstatistikleri güncelle
            if pnl > 0:
                wins += 1
            else:
                losses += 1
            
            # Bileşik bakiye güncellemesi
            equity *= (1 + return_on_investment)

            # Drawdown hesaplaması için
            if equity > peak_equity:
                peak_equity = equity
            
            drawdown = ((peak_equity - equity) / peak_equity) * 100 if peak_equity > 0 else 0
            max_drawdown = max(max_drawdown, drawdown)

            # Pozisyonu sıfırla
            position_active = False
            position_direction = None
            position_size = 0.0
            position_cost = 0.0
            avg_entry_price = 0.0

    num_closed_trades = len(closed_trades)
    win_rate = (wins / num_closed_trades * 100) if num_closed_trades > 0 else 0
    total_return_abs = sum(t['pnl'] for t in closed_trades)
    # Bileşik getiri yüzdesi
    total_return_pct = ((equity - initial_equity) / initial_equity) * 100
    
    return {
        'total_return': total_return_abs, # Mutlak kar/zarar
        'return_pct': total_return_pct,   # Bileşik getiri yüzdesi
        'win_rate': win_rate,
        'max_drawdown': max_drawdown,
        'num_trades': num_closed_trades,
        'wins': wins,      # EKLENDİ
        'losses': losses   # EKLENDİ
    }

# =============================================================================
# OPTUNA OBJECTIVE
# =============================================================================

def objective_whale(trial, data_slice):

    """Whale Strategy için Optuna optimizasyon fonksiyonu"""
    # Kullanıcının isteği: Hangi denemede olduğunu görmek için print ekle
    print(f"🚀 Optimizasyon Adımı #{trial.number}...")

    params = {
        'enable_take_profit': trial.suggest_categorical('enable_take_profit', [False]),
        'enable_trailing_stop': trial.suggest_categorical('enable_trailing_stop', [True]),
        'enable_maliyet_exit': trial.suggest_categorical('enable_maliyet_exit', [True]),
        'enable_mini_stop': trial.suggest_categorical('enable_mini_stop', [True]),
        'enable_dca': trial.suggest_categorical('enable_dca', [True]),
        'enable_drop_speed': trial.suggest_categorical('enable_drop_speed', [True]),
        'enable_panic_drop': trial.suggest_categorical('enable_panic_drop', [True]),
    }

    if params['enable_take_profit']:
        params['take_profit_perc'] = trial.suggest_float('take_profit_perc', 3, 5, step=1)
    
    if params['enable_trailing_stop']:
        params['trailing_perc'] = trial.suggest_float('trailing_perc', 0.2, 2.0, step=0.2)
        params['min_position_for_ts'] = trial.suggest_int('min_position_for_ts', 2, 5, step=1)
        # Trailing sonrası extra TP optimizasyonu (yeni eklendi)
        params['enable_extra_tp_after_ts'] = trial.suggest_categorical('enable_extra_tp_after_ts', [True, False])
        if params['enable_extra_tp_after_ts']:
            params['extra_tp_perc'] = trial.suggest_float('extra_tp_perc', 0.5, 5.0, step=0.5)

    
    if params['enable_maliyet_exit']:
        params['maliyet_return_perc'] = trial.suggest_float('maliyet_return_perc', 0.0, 1.0, step=0.2)
        params['min_entry_for_maliyet'] = trial.suggest_int('min_entry_for_maliyet', 3, 7, step=1)
    
    if params['enable_mini_stop']:
        params['maliyet_stop_perc'] = trial.suggest_float('maliyet_stop_perc', 1.0, 5.0, step=0.5)
        params['mini_loss_cooldown_bars'] = trial.suggest_int('mini_loss_cooldown_bars', 10, 50, step=10)
        params['min_entry_for_mini_loss'] = trial.suggest_int('min_entry_for_mini_loss', 3, 7, step=1)
    
    if params['enable_dca']:
        params['max_dca'] = trial.suggest_int('max_dca', 1, 4, step=1)
        params['dca_drop_perc'] = trial.suggest_float('dca_drop_perc', 2, 5, step=1) * 0.01
        params['dca_mum_delay'] = trial.suggest_int('dca_mum_delay', 2, 6, step=1)
    
    if params['enable_panic_drop']:
        params['panic_drop_perc'] = trial.suggest_float('panic_drop_perc', 5.0, 13.0, step=2.0)
        params['panic_drop_bars'] = trial.suggest_int('panic_drop_bars', 1, 3, step=1)
        params['panic_cooldown_bars'] = trial.suggest_int('panic_cooldown_bars', 20, 100, step=20)
    
    if params['enable_drop_speed']:
        params['drop_speed_perc'] = trial.suggest_float('drop_speed_perc', 2.0, 6.0, step=1.0)
        params['drop_speed_bars'] = trial.suggest_int('drop_speed_bars', 1, 3, step=1)
    
    params['position_multiplier'] = trial.suggest_float('position_multiplier', 1, 2, step=1)


    try:
        data_copy = data_slice.copy()
        data_copy.reset_index(drop=True, inplace=True)
        
        strategy = WhaleStopStrategy(data_copy)
        
        for key, value in params.items():
            if hasattr(strategy, key):
                setattr(strategy, key, value)
        
        df_result, trades, signals = strategy.run_strategy()
        
        stats = calculate_backtest_stats(trades)
        
        num_trades = stats['num_trades']
        win_rate = stats['win_rate']
        return_pct = stats['return_pct']
        max_drawdown = stats['max_drawdown']

        if num_trades < 2:
            return -10.0 + num_trades 

        score_win = 4.0 * ((win_rate - 50) / 50.0)
        capped_dd = min(max_drawdown, 60.0) 
        score_dd = 3.0 * (1 - (capped_dd / 30.0))
        score_profit = 3.0 * np.tanh(return_pct / 25.0)

        final_score = score_win + score_dd + score_profit
        final_score = max(min(final_score, 10.0), -10.0)

        if np.isnan(final_score):
            return -10.0
            
        return final_score
        
    except Exception as e:
        trial.set_user_attr("error", str(e))
        return -10.0

# =============================================================================
# KAPSAMLI DASHBOARD OLUŞTURUCU (HATA GİDERİLDİ)
# =============================================================================

def create_comprehensive_results_dashboard(best_params, stats, trades, timestamp):
    """
    En iyi parametreleri ve backtest sonuçlarını içeren kapsamlı HTML dashboard
    """
    filename = f'whale_dashboard_{timestamp}.html'
    print(f"\n📋 Kapsamlı sonuç dashboard'u oluşturuluyor: {filename}")

    # Win/Loss Ratio Güvenli Hesaplama
    wins = stats.get('wins', 0)
    losses = stats.get('losses', 0)
    if losses > 0:
        win_loss_ratio = f"{wins / losses:.2f}"
    else:
        win_loss_ratio = "∞"
    
    # Renkleri belirle (f-string içinde if/else karışıklığını önlemek için)
    color_ret = '#4CAF50' if stats['total_return'] > 0 else '#f44336'
    color_ret_pct = '#4CAF50' if stats['return_pct'] > 0 else '#f44336'
    color_win = '#4CAF50' if stats['win_rate'] > 50 else '#f44336'
    color_dd = '#4CAF50' if stats['max_drawdown'] < 20 else '#f44336'
    color_w = '#4CAF50'
    color_l = '#f44336'
    # Düzeltme: stats sözlüğünde anahtarın varlığını kontrol et
    stats_total_return = stats.get('total_return', 0)
    stats_return_pct = stats.get('return_pct', 0)
    stats_win_rate = stats.get('win_rate', 0)
    stats_max_drawdown = stats.get('max_drawdown', 0)

    color_ret = '#4CAF50' if stats_total_return > 0 else '#f44336'
    color_ret_pct = '#4CAF50' if stats_return_pct > 0 else '#f44336'
    color_win = '#4CAF50' if stats_win_rate > 50 else '#f44336'
    color_dd = '#4CAF50' if stats_max_drawdown < 20 else '#f44336'
    color_w = '#4CAF50' # Kazananlar için her zaman yeşil
    color_l = '#f44336' # Kaybedenler için her zaman kırmızı


    # --- CSS KODU (AYRI DEĞİŞKENDE - HATA ÖNLEYİCİ) ---
    css_style = """
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%); color: #fff; padding: 20px; min-height: 100vh; }
        .container { max-width: 1400px; margin: 0 auto; }
        .header { text-align: center; padding: 40px 20px; background: rgba(255,255,255,0.1); border-radius: 20px; margin-bottom: 30px; backdrop-filter: blur(10px); box-shadow: 0 8px 32px rgba(0,0,0,0.3); }
        .header h1 { font-size: 3em; margin-bottom: 10px; text-shadow: 2px 2px 4px rgba(0,0,0,0.3); }
        .header .emoji { font-size: 4em; margin-bottom: 20px; }
        .header .timestamp { font-size: 1.1em; opacity: 0.8; margin-top: 10px; }
        .grid, .summary-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin-bottom: 30px; }
        .card, .summary-card { background: rgba(255,255,255,0.1); border-radius: 15px; padding: 20px; backdrop-filter: blur(10px); box-shadow: 0 8px 32px rgba(0,0,0,0.2); transition: transform 0.3s ease, box-shadow 0.3s ease; text-align: center; }
        .card:hover, .summary-card:hover { transform: translateY(-5px); box-shadow: 0 12px 40px rgba(0,0,0,0.3); }
        .card h2 { font-size: 1.5em; margin-bottom: 20px; border-bottom: 2px solid rgba(255,255,255,0.3); padding-bottom: 10px; text-align: left; }
        .metric-value, .value { font-size: 1.8em; font-weight: bold; margin: 10px 0; }
        .metric-label, .label { font-size: 1em; opacity: 0.8; }
        .param-table { width: 100%; border-collapse: collapse; margin-top: 15px; }
        .param-table th, .param-table td { padding: 12px; text-align: left; border-bottom: 1px solid rgba(255,255,255,0.1); }
        .param-table th { background: rgba(255,255,255,0.1); font-weight: bold; }
        .param-table tr:hover { background: rgba(255,255,255,0.05); }
        .badge { display: inline-block; padding: 5px 15px; border-radius: 20px; font-size: 0.9em; font-weight: bold; }
        .badge.success { background: #4CAF50; }
        .badge.warning { background: #FFC107; color: #000; }
        .badge.error { background: #f44336; }
        .badge.info { background: #2196F3; }
        .trades-section { margin-top: 30px; }
        .trade-item { background: rgba(255,255,255,0.05); padding: 15px; border-radius: 10px; margin: 10px 0; border-left: 4px solid #4CAF50; text-align: left; }
        .trade-item.exit { border-left-color: #FFC107; }
        .trade-item.short { border-left-color: #f44336; }
        .progress-bar { width: 100%; height: 30px; background: rgba(255,255,255,0.1); border-radius: 15px; overflow: hidden; margin: 10px 0; }
        .progress-fill { height: 100%; background: linear-gradient(90deg, #4CAF50, #8BC34A); display: flex; align-items: center; justify-content: center; font-weight: bold; transition: width 0.3s ease; }
        .icon { font-size: 2em; margin-bottom: 10px; }
    """

    # HTML GÖVDE
    html_content = f"""
<!DOCTYPE html>
<html lang="tr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Whale Strategy - Optimization Results</title>
    <style>{css_style}</style>
</head>
<body>
    <div class="container">
        <div class="header">
            <div class="emoji">🐋</div>
            <h1>Whale Strategy Optimization Results</h1>
            <div class="timestamp">⏰ {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}</div>
        </div>
        
        <div class="summary-grid">
            <div class="summary-card">
                <div class="icon">💰</div>
                <div class="value" style="color: {color_ret}">${stats['total_return']:,.2f}</div>
                <div class="label">Toplam Kar/Zarar</div>
            </div>
            
            <div class="summary-card">
                <div class="icon">📈</div>
                <div class="value" style="color: {color_ret_pct}">%{stats['return_pct']:.2f}</div>
                <div class="label">Getiri Oranı</div>
            </div>
            
            <div class="summary-card">
                <div class="icon">🎯</div>
                <div class="value" style="color: {color_win}">%{stats['win_rate']:.2f}</div>
                <div class="label">Kazanma Oranı</div>
            </div>
            
            <div class="summary-card">
                <div class="icon">📉</div>
                <div class="value" style="color: {color_dd}">%{stats['max_drawdown']:.2f}</div>
                <div class="label">Max Drawdown</div>
            </div>
            
            <div class="summary-card">
                <div class="icon">🔢</div>
                <div class="value">{stats['num_trades']}</div>
                <div class="label">Toplam İşlem</div>
            </div>
            
            <div class="summary-card">
                <div class="icon">✅</div>
                <div class="value" style="color: {color_w}">{wins}</div>
                <div class="label">Kazanan</div>
            </div>
            
            <div class="summary-card">
                <div class="icon">❌</div>
                <div class="value" style="color: {color_l}">{losses}</div>
                <div class="label">Kaybeden</div>
            </div>
            
            <div class="summary-card">
                <div class="icon">⚖️</div>
                <div class="value">{win_loss_ratio}</div>
                <div class="label">Win/Loss Ratio</div>
            </div>
        </div>
        
        <div class="card">
            <h2>📊 Kazanma Oranı Göstergesi</h2>
            <div class="progress-bar">
                <div class="progress-fill" style="width: {stats['win_rate']}%">
                    %{stats['win_rate']:.1f}
                </div>
            </div>
        </div>
        
        <div class="card">
            <h2>⚙️ En İyi Çıkış Parametreleri</h2>
            <table class="param-table">
                <thead>
                    <tr>
                        <th>Parametre</th>
                        <th>Değer</th>
                        <th>Durum</th>
                    </tr>
                </thead>
                <tbody>"""
    
    # Parametreleri kategorilere ayır ve güzel göster
    param_categories = {
        'Take Profit': ['enable_take_profit', 'take_profit_perc'],
        'Trailing Stop': ['enable_trailing_stop', 'trailing_perc', 'min_position_for_ts', 'enable_extra_tp_after_ts', 'extra_tp_perc'],
        'Pozisyon Büyütme': ['position_multiplier'],
        'DCA': ['enable_dca', 'max_dca', 'dca_drop_perc', 'dca_mum_delay'],
        'Maliyet Çıkışı': ['enable_maliyet_exit', 'min_entry_for_maliyet', 'maliyet_return_perc'],
        'Mini Loss': ['enable_mini_stop', 'min_entry_for_mini_loss', 'maliyet_stop_perc', 'mini_loss_cooldown_bars'],
        'Whale Stop': ['enable_drop_speed', 'drop_speed_perc', 'drop_speed_bars', 'enable_panic_drop', 'panic_drop_perc', 'panic_drop_bars', 'panic_cooldown_bars']
    }

    
    param_labels = {
        'enable_take_profit': 'Take Profit Aktif',
        'take_profit_perc': 'TP Yüzdesi',
        'enable_trailing_stop': 'Trailing Stop Aktif',
        'trailing_perc': 'Trailing Yüzdesi',
        'enable_min_position_for_ts': 'Min Pozisyon Kontrolü',
        'min_position_for_ts': 'Min Pozisyon Sayısı',
        'enable_extra_tp_after_ts': 'Trailing Sonrası Ekstra TP',
        'extra_tp_perc': 'Ekstra TP Yüzdesi',
        'position_multiplier': 'Pozisyon Çarpanı',
        'enable_dca': 'DCA Aktif',
        'max_dca': 'Maksimum DCA Sayısı',
        'dca_drop_perc': 'DCA Düşüş Yüzdesi',
        'dca_mum_delay': 'DCA Mum Gecikmesi',
        'enable_maliyet_exit': 'Maliyet Çıkışı Aktif',
        'min_entry_for_maliyet': 'Maliyet İçin Min Entry',
        'maliyet_return_perc': 'Maliyet Return %',
        'enable_mini_stop': 'Mini Loss Aktif',
        'min_entry_for_mini_loss': 'Mini Loss İçin Min Entry',
        'maliyet_stop_perc': 'Mini Loss Stop %',
        'mini_loss_cooldown_bars': 'Mini Loss Cooldown',
        'enable_drop_speed': 'Drop Speed Aktif',
        'drop_speed_perc': 'Drop Speed %',
        'drop_speed_bars': 'Drop Speed Barlar',
        'enable_panic_drop': 'Panic Drop Aktif',
        'panic_drop_perc': 'Panic Drop %',
        'panic_drop_bars': 'Panic Drop Barlar',
        'panic_cooldown_bars': 'Panic Cooldown'
    }
    
    for category, params in param_categories.items():
        html_content += f"""
                    <tr>
                        <td colspan="3" style="background: rgba(33,150,243,0.2); font-weight: bold; font-size: 1.1em;">
                            {category}
                        </td>
                    </tr>"""
        
        for param in params:
            if param in best_params:
                value = best_params[param]
                label = param_labels.get(param, param)
                
                # Değer formatı
                if isinstance(value, bool):
                    display_value = '✅ Aktif' if value else '❌ Pasif'
                    badge_class = 'success' if value else 'error'
                elif isinstance(value, float):
                    display_value = f'{value:.2f}'
                    badge_class = 'info'
                else:
                    display_value = str(value)
                    badge_class = 'info'
                
                html_content += f"""
                    <tr>
                        <td>{label}</td>
                        <td><strong>{display_value}</strong></td>
                        <td><span class="badge {badge_class}">Optimize Edildi</span></td>
                    </tr>"""
    
    html_content += """
                </tbody>
            </table>
        </div>
        
        <div class="card trades-section">
            <h2>📋 İşlem Geçmişi (Son 20 İşlem)</h2>"""
    
    # Son 20 işlemi göster
    recent_trades = trades[-20:] if len(trades) > 20 else trades
    
    for idx, trade in enumerate(reversed(recent_trades), 1):
        trade_class = 'trade-item'
        if trade['type'] == 'EXIT':
            trade_class += ' exit'
        elif trade.get('direction') == 'SHORT':
            trade_class += ' short'
        
        price = trade.get('price', 0)
        size = trade.get('size', 0)
        comment = trade.get('comment', 'N/A')
        
        html_content += f"""
            <div class="{trade_class}">
                <strong>#{len(trades) - idx + 1}</strong> - 
                {trade['type']} {trade.get('direction', '')} @ 
                <strong>${price:.4f}</strong> | 
                Size: {size:.4f} | 
                {comment}
            </div>"""
    
    html_content += """
        </div>
        
        <div class="card" style="text-align: center; margin-top: 30px;">
            <h3>🎯 Optimizasyon Başarıyla Tamamlandı!</h3>
            <p style="margin-top: 15px; opacity: 0.8;">
                Bu sonuçlar Optuna optimizasyon algoritması kullanılarak elde edilmiştir.
            </p>
        </div>
    </div>
</body>
</html>"""
    
    # HTML dosyasını kaydet
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"✅ Kapsamlı dashboard kaydedildi: {filename}")
    
    # Otomatik aç
    try:
        webbrowser.open('file://' + os.path.abspath(filename))
    except Exception as e:
        print(f"Otomatik açma başarısız: {e}")
    
    return filename

# =============================================================================
# MAIN (ÇALIŞTIRMA KISMI)
# =============================================================================
if __name__ == "__main__":
    # Veri yükleme
    try:
        DATA_FILE = '1000SHIB_5d_verisi.csv'
        data = pd.read_csv(DATA_FILE)
        data.columns = [c.lower().strip() for c in data.columns]
        
        if 'datetime' in data.columns:
            data['datetime'] = pd.to_datetime(data['datetime'])
            data.set_index('datetime', inplace=True)
        else:
            print("HATA: Veride 'datetime' sütunu bulunamadı.")
            sys.exit(1)
            
        if not isinstance(data.index, pd.DatetimeIndex):
            print("HATA: Veri indeksi DatetimeIndex tipinde değil.")
            sys.exit(1)
        
            
        # =============================================================================
        # ADIM 0: TÜM İNDİKATÖRLERİ ÖNCEDEN HESAPLA (PERFORMANS ARTIŞI)
        # =============================================================================
        # NOT: precompute_indicators fonksiyonunun tanımı user input'ta yoktu.
        # Eğer bu fonksiyon başka bir yerde tanımlıysa sorun yok.
        # Eğer tanımlı değilse hata verebilir. Buraya bir placeholder ekliyorum:
        if 'precompute_indicators' in globals():
            data = precompute_indicators(data)
        else:
            print("Uyarı: 'precompute_indicators' fonksiyonu bulunamadı, bu adım atlanıyor.")
        
        print(f"Veri başarıyla yüklendi: {DATA_FILE} (Toplam {len(data)} bar)")
        print(f"Veri Periyodu: {data.index.min()} -> {data.index.max()}")
        
    except FileNotFoundError:
        print(f"HATA: '{DATA_FILE}' dosyası bulunamadı.")
        print("Lütfen veri dosyasının bu script ile aynı klasörde olduğundan emin olun.")
        sys.exit(1)
    except Exception as e:
        print("VERİ YÜKLEME HATASI:", e)
        traceback.print_exc()
        sys.exit(1)
    
    # ... (HTML STYLE değişkeni artık kullanılmıyor, dashboard kendi içinde hallediyor) ...
    
    # -----------------------------------------------------
    # BÖLÜM 1: TAM VERİ OPTİMİZASYONU
    # -----------------------------------------------------
    
    print("\n" + "="*50)
    print(" BÖLÜM 1: TAM VERİ ÜZERİNDE OPTİMİZASYON")
    print("="*50)
    
    study_full = optuna.create_study(
        storage="sqlite:///db.sqlite3",       # Aynı veritabanına bağlanır
        study_name="WFO_Genel_Sonuc",         # Dashboard'da ayrı bir başlık olur
        direction="maximize", 
        sampler=optuna.samplers.TPESampler(),
        load_if_exists=True
    )
    
    n_trials_full = 20  # DENEME SAYISI

    print(f"\nAI Optimizasyonu (Tam Veri) başlıyor... Toplam {n_trials_full} deneme yapılacak.")
    
    try:
        study_full.optimize(lambda trial: objective_whale(trial, data), n_trials=n_trials_full, n_jobs=1)
    except KeyboardInterrupt:
        print("\nOptimizasyon kullanıcı tarafından durduruldu.")
    except Exception as e:
        print("Optimizasyon sırasında hata:", e)
        traceback.print_exc()
    
    print("\nTam Veri Optimizasyonu Tamamlandı!")
    
    if study_full.best_trial and study_full.best_trial.value is not None:
        best_params_full = study_full.best_trial.params
        best_value_full = study_full.best_trial.value
        print("\n--- EN İYİ (TAM VERİ) SONUÇLARI ---")
        print(f"En İyi Deneme: #{study_full.best_trial.number} | Skor: {best_value_full:.4f}")
        print("\nEn İyi Parametreler (Tam Veri):")
        print(json.dumps(best_params_full, indent=4))
        
        try:
            print("\n🧪 En iyi parametrelerle test ediliyor...")
            
            test_data_full = data.copy()
            test_data_full.reset_index(drop=True, inplace=True)
            
            strategy_best = WhaleStopStrategy(test_data_full)
            
            for key, value in best_params_full.items():
                if hasattr(strategy_best, key):
                    setattr(strategy_best, key, value)
            
            df_result, trades, signals = strategy_best.run_strategy()
            
            print("\n📊 TEST SONUÇLARI:")
            final_stats = calculate_backtest_stats(trades)
            print("--- Tam Veri Strateji Performansı (Geliştirilmiş Hesaplama) ---")
            print(f"   • Toplam Kapalı İşlem: {final_stats['num_trades']}")
            print(f"   • Getiri: {final_stats['return_pct']:.2f}%")
            print(f"   • Win Rate: {final_stats['win_rate']:.2f}%")
            print(f"   • Max DD: {final_stats['max_drawdown']:.2f}%")

            if 'datetime' not in df_result.columns:
                df_result['datetime'] = data.index[:len(df_result)]
            
            visualizer = TradingVisualizer(df_result, trades, signals)
            visualizer.create_interactive_chart('tam_veri_whale_strategy.html', show_browser=False)

            # --- DÜZELTME: KAPSAMLI DASHBOARD BURADA ÇAĞRILIYOR ---
            create_comprehensive_results_dashboard(
                best_params_full,
                final_stats,
                trades,
                datetime.now().strftime("%Y%m%d_%H%M%S")
            )
            
        except Exception as e:
            print("Final test hatası:")
            traceback.print_exc()
    else:
        print("\nTam Veri Optimizasyonunda anlamlı bir sonuç bulunamadı.")
    
    # -----------------------------------------------------
    # BÖLÜM 2: WALK-FORWARD OPTIMIZATION (WFO) - DÜZELTME
    # -----------------------------------------------------
    
    print("\n" + "="*50)
    print(" BÖLÜM 2: WALK-FORWARD OPTİMİZASYON (WFO) BAŞLIYOR")
    print("="*50)
    
    # -----------------------------------------------------
    #               DENEME GÜNÜ
    # -----------------------------------------------------
    IN_SAMPLE_DAYS = 270       # Öğrenme periyodu (GÜN) - Daha fazla WFO adımı için düşürüldü
    OUT_OF_SAMPLE_DAYS = 30    # Test periyodu (GÜN)
    N_TRIALS_PER_STEP = 20    # Her WFO adımındaki Optuna deneme sayısı
    # -----------------------------------------------------

    print(f"WFO Ayarları: {IN_SAMPLE_DAYS} GÜN öğren, SON {OUT_OF_SAMPLE_DAYS} GÜN test et.")
    print(f"Her adımda {N_TRIALS_PER_STEP} deneme yapılacak.")
    
    start_date = data.index.min()
    end_date = data.index.max()
    
    all_oos_results = []
    wfo_step_count = 0
    
    current_start = start_date
    
    while current_start + DateOffset(days=IN_SAMPLE_DAYS + OUT_OF_SAMPLE_DAYS) <= end_date:
        wfo_step_count += 1
        
        train_start = current_start
        train_end = train_start + DateOffset(days=IN_SAMPLE_DAYS)
        test_start = train_end + DateOffset(microseconds=1) 
        test_end = test_start + DateOffset(days=OUT_OF_SAMPLE_DAYS)
        
        train_data = data.loc[train_start:train_end]
        test_data = data.loc[test_start:test_end]
        
        if train_data.empty or test_data.empty or len(test_data) < 20:
            print(f"\n⚠️ Adım {wfo_step_count} atlanıyor (yetersiz veri)")
            current_start += DateOffset(days=OUT_OF_SAMPLE_DAYS)
            continue
        
        print(f"\n{'='*70}")
        print(f"📊 WFO ADIM {wfo_step_count}")
        print(f"{'='*70}")
        print(f"🎓 Öğrenme:  {train_data.index.min().strftime('%Y-%m-%d')} → {train_data.index.max().strftime('%Y-%m-%d')} ({len(train_data)} bar)")
        print(f"🧪 Test:     {test_data.index.min().strftime('%Y-%m-%d')} → {test_data.index.max().strftime('%Y-%m-%d')} ({len(test_data)} bar)")
        
        study_step = optuna.create_study(
            storage="sqlite:///db.sqlite3",       # Veritabanı bağlantısı
            study_name="WFO_Adim_{i}",       # Dashboard'da görünecek isim
            direction="maximize", 
            sampler=optuna.samplers.TPESampler(),
            load_if_exists=True                   # Hata almamak için şart
        )
        
        try:
            print(f"\n🔍 {N_TRIALS_PER_STEP} deneme yapılıyor...")
            study_step.optimize(lambda trial: objective_whale(trial, train_data), n_trials=N_TRIALS_PER_STEP, n_jobs=1)
            
            if study_step.best_trial and study_step.best_trial.value > -9999:
                best_params_step = study_step.best_trial.params
                print(f"\n✅ Öğrenme tamamlandı! En iyi skor: {study_step.best_trial.value:.4f}")
                
                print(f"🧪 Test verisinde çalıştırılıyor...")
                
                test_data_copy = test_data.copy()
                test_data_copy.reset_index(drop=True, inplace=True)
                
                strategy_test = WhaleStopStrategy(test_data_copy)
                
                for key, value in best_params_step.items():
                    if hasattr(strategy_test, key):
                        setattr(strategy_test, key, value)
                
                df_test, trades_test, signals_test = strategy_test.run_strategy()
                
                test_stats = calculate_backtest_stats(trades_test)
                
                result = {
                    'step': wfo_step_count,
                    'train_start': train_data.index.min().strftime('%Y-%m-%d'),
                    'train_end': train_data.index.max().strftime('%Y-%m-%d'),
                    'test_start': test_data.index.min().strftime('%Y-%m-%d'),
                    'test_end': test_data.index.max().strftime('%Y-%m-%d'),
                    'test_return': test_stats['return_pct'],
                    'test_win_rate': test_stats['win_rate'],
                    'test_max_dd': test_stats['max_drawdown'],
                    'test_trades': test_stats['num_trades'],
                    'best_params': best_params_step
                }
                
                all_oos_results.append(result)
                
                print(f"\n📈 TEST SONUÇLARI:")
                print(f"   • Getiri: {test_stats['return_pct']:.2f}%")
                print(f"   • Win Rate: {test_stats['win_rate']:.2f}%")
                print(f"   • Max DD: {test_stats['max_drawdown']:.2f}%")
                print(f"   • İşlem: {test_stats['num_trades']}")
                
            else:
                print(f"\n⚠️ Adım {wfo_step_count}: Geçerli strateji bulunamadı")
                
        except Exception as e:
            print(f"\n❌ Adım {wfo_step_count} HATA: {e}")
            traceback.print_exc()
        
        current_start += DateOffset(days=OUT_OF_SAMPLE_DAYS)
    
    print("\n" + "="*50)
    print("WFO OPTİMİZASYONU TAMAMLANDI")
    print("="*50)
    
    # WFO Raporlaması (Konsol)
    total_trades_wfo = 0
    final_wfo_stats = {}
    if not all_oos_results:
        print("Hiç geçerli WFO adımı tamamlanamadı. Rapor oluşturulmuyor.")
    else:
        print("Tüm Out-of-Sample (OOS) Periyotların Özeti:")
        
        for result in all_oos_results:
            print(f"Adım {result['step']}: Test [{result['test_start']} → {result['test_end']}] → Getiri: {result['test_return']:.2f}%, WinRate: {result['test_win_rate']:.2f}%, İşlem: {result['test_trades']}")
            total_trades_wfo += result['test_trades']
        print(f"\nTüm OOS periyotları boyunca toplam işlem sayısı: {total_trades_wfo}")

    
    # -----------------------------------------------------
    # BÖLÜM 3 (YENİ): WFO GELİŞMİŞ ANALİZ VE BİRLEŞİK RAPOR
    # -----------------------------------------------------
    
    print("\n" + "="*50)
    print(" BÖLÜM 4: WFO GELİŞMİŞ ANALİZ VE BİRLEŞİK RAPOR")
    print("="*50)
    
    # HATA DÜZELTMESİ: Tüm rapor değişkenlerini try bloğunun dışında başlat.
    # Bu, herhangi bir hata durumunda NameError'ı engeller.
    HTML_STYLE = """
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif; background-color: #f4f6f9; color: #333; margin: 0; padding: 25px; }
        .container { max-width: 95%; margin: auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 5px 15px rgba(0,0,0,0.08); }
        .header { color: #2c3e50; text-align: center; margin-bottom: 30px; font-weight: 600; font-size: 2em; border-bottom: 2px solid #dfe4ea; padding-bottom: 20px; }
        h2 { color: #4a69bd; margin-top: 30px; }
        table { width: 100%; border-collapse: collapse; margin-top: 20px; }
        th, td { padding: 12px; border: 1px solid #dfe4ea; text-align: left; }
        th { background-color: #f8f9fa; }
        tr:nth-child(even) { background-color: #f2f2f2; }
        .footer { text-align: center; margin-top: 40px; padding-top: 20px; border-top: 1px solid #dfe4ea; color: #576574; font-size: 0.9em; }
    </style>
    """
    html_stitched_equity = "<p>Birleşik bakiye eğrisi için yeterli WFO adımı yok.</p>"
    html_params_over_time = "<p>Parametre değişimi için yeterli WFO adımı yok.</p>"
    summary_html = "<p>WFO özeti için yeterli WFO adımı yok.</p>"
    robust_params_html = "<p>Sağlam parametre analizi için yeterli WFO adımı yok.</p>"
    total_return_wfo = 0.0
    robust_params = {} # Final testi için robust_params'ı burada başlat

    if all_oos_results and len(all_oos_results) > 1:
        try:
            # 1. BİRLEŞİK BAKİYE EĞRİSİ (EQUITY STITCHING)
            print("Birleşik bakiye eğrisi oluşturuluyor...")
            initial_equity = 10000
            current_equity = initial_equity
            
            # Başlangıç noktasını oluştur
            start_date = pd.to_datetime(all_oos_results[0]['test_start'])
            
            stitched_equity_data = {start_date: current_equity}

            for result in all_oos_results:
                period_return_pct = result['test_return']
                # Her periyodun sonunda bakiyeyi güncelle
                current_equity *= (1 + period_return_pct / 100)
                end_date = pd.to_datetime(result['test_end'])
                stitched_equity_data[end_date] = current_equity
            
            # DataFrame'e çevir
            stitched_equity_curve = pd.Series(stitched_equity_data, name="Equity").reset_index()
            stitched_equity_curve.columns = ['Date', 'Equity']
            
            total_return_wfo = (current_equity - initial_equity) / initial_equity * 100
            
            # Final WFO istatistiklerini de sakla
            final_wfo_stats = {
                'total_return_wfo': total_return_wfo,
                'total_trades_wfo': total_trades_wfo,
                'num_wfo_steps': len(all_oos_results)
            }

            fig_equity = px.line(stitched_equity_curve, x='Date', y='Equity', 
                                 title="Birleşik WFO Bakiye Eğrisi (Out-of-Sample)")
            fig_equity.update_layout(xaxis_title="Tarih", yaxis_title="Bakiye")
            html_stitched_equity = fig_equity.to_html(include_plotlyjs='cdn', full_html=False)
            print(f"Birleşik Bakiye Eğrisi Toplam Getiri: {total_return_wfo:.2f}%")


            # 2. PARAMETRE DEĞİŞİM GRAFİĞİ
            # Bu, sizin 'numeric_cols' hatanızın düzeltilmiş halidir.
            print("Parametre değişim grafiği oluşturuluyor...")
            params_list = [res['best_params'] for res in all_oos_results]
            params_df = pd.DataFrame(params_list)
            
            # Tarihleri ekle
            params_df['test_end_date'] = pd.to_datetime([res['test_end'] for res in all_oos_results])
            
            # Sadece sayısal olan (numeric) parametreleri seç
            numeric_params_df = params_df.select_dtypes(include=[np.number])
            numeric_params_df['Date'] = params_df['test_end_date']
            
            # Uzun formata (long format) çevir
            params_df_long = numeric_params_df.melt(id_vars='Date', var_name='Parameter', value_name='Value')

            if not params_df_long.empty:
                fig_params = px.line(params_df_long, x='Date', y='Value', color='Parameter',
                                     title="Optimize Edilen Parametrelerin Zaman İçinde Değişimi (WFO)",
                                     markers=True)
                fig_params.update_layout(xaxis_title="Tarih", yaxis_title="Parametre Değeri")
                html_params_over_time = fig_params.to_html(include_plotlyjs=False, full_html=False)
            else:
                print("⚠️ Parametre değişim grafiği için çizilecek sayısal parametre bulunamadı.")
                html_params_over_time = "<p>Grafik oluşturulamadı: Optimize edilen sayısal parametre bulunamadı.</p>"

            # 3. EN SAĞLAM (ROBUST) ORTAK PARAMETRELERİ BULMA
            print("\n" + "-"*30)
            print("En Sağlam WFO Parametreleri Hesaplanıyor...")
            print("-" * 30)
            
            wfo_params_df = pd.DataFrame([res['best_params'] for res in all_oos_results])
            
            for param in wfo_params_df.columns:
                if wfo_params_df[param].dtype == 'object' or wfo_params_df[param].dtype == 'bool':
                    robust_params[param] = bool(wfo_params_df[param].mode().iloc[0])
                elif pd.api.types.is_numeric_dtype(wfo_params_df[param]):
                    median_val = wfo_params_df[param].median()
                    if pd.api.types.is_integer_dtype(wfo_params_df[param].dropna()):
                         robust_params[param] = int(round(median_val))
                    else:
                         robust_params[param] = median_val

            print("✅ En Sağlam Parametre Seti:")
            print(json.dumps(robust_params, indent=4))

            # HTML Raporu için tablo oluştur
            robust_params_html = "<h2>En Sağlam WFO Parametreleri (Tüm Adımların Ortak Aklı)</h2>"
            robust_params_html += "<p>Bu tablo, tüm WFO adımlarında bulunan en iyi parametrelerin istatistiksel olarak (medyan/mod) birleştirilmesiyle oluşturulmuştur. Bu set, stratejinin en genel geçer ve sağlam ayarlarını temsil eder.</p>"
            robust_params_html += "<table border='1'><tr><th>Parametre</th><th>Değer</th></tr>"
            for key, value in robust_params.items():
                if isinstance(value, bool):
                    display_value = '✅ Aktif' if value else '❌ Pasif'
                else:
                    display_value = f"{value:.4f}" if isinstance(value, float) else value
                robust_params_html += f"<tr><td>{key}</td><td><strong>{display_value}</strong></td></tr>"
            robust_params_html += "</table>"

        except Exception as e:
            print(f"WFO Analiz grafikleri oluşturulurken hata: {e}")
            traceback.print_exc()
            html_stitched_equity = f"<p>Bakiye eğrisi hatası: {e}</p>"
            html_params_over_time = f"<p>Parametre grafiği hatası: {e}</p>"
            robust_params_html = f"<p>Sağlam parametre analizi hatası: {e}</p>"

        # 4. WFO ÖZET TABLOSU
        summary_html = "<h2>WFO Adım Özetleri (Out-of-Sample Performans)</h2>"
        summary_html += "<table border='1'><tr><th>Adım</th><th>Öğrenme Periyodu</th><th>Test Periyodu</th><th>Getiri [%]</th><th>Max. Drawdown [%]</th><th>Win Rate [%]</th><th># Trades</th><th>En İyi Parametreler</th></tr>"
        
        for result in all_oos_results:
            params_str = json.dumps(result['best_params'], indent=2)
            params_html = f"<pre style='font-size: 11px; text-align: left; margin: 0; padding: 5px; background-color: #fdfdfd;'>{params_str}</pre>"
            summary_html += f"<tr><td>{result['step']}</td><td>{result['train_start']} → {result['train_end']}</td><td>{result['test_start']} → {result['test_end']}</td><td>{result['test_return']:.2f}</td><td>{result['test_max_dd']:.2f}</td><td>{result['test_win_rate']:.2f}</td><td>{result['test_trades']}</td><td>{params_html}</td></tr>"
        
        summary_html += "</table>"
        
        # 5. WFO HTML RAPORUNU OLUŞTUR
        wfo_html_content = f"""
        <!DOCTYPE html><html lang="tr"><head><meta charset="UTF-8"><title>Whale WFO Raporu (Gelişmiş)</title>{HTML_STYLE}</head>
        <body><div class="container"><div class="header">Whale Strategy - Gelişmiş Walk-Forward Raporu</div>
        <div class="content">
            <p><strong>Mantık:</strong> {IN_SAMPLE_DAYS} gün öğren → SON {OUT_OF_SAMPLE_DAYS} gün test et → Pencereyi kaydır</p>
            
            <h2>Birleşik Bakiye Eğrisi (Out-of-Sample)</h2>
            <p>Bu grafik, tüm "bilinmeyen" test periyotlarının getirilerini birleştirerek stratejinin kümülatif performansını gösterir.</p>
            {html_stitched_equity}
            
            <h2>Parametrelerin Zaman İçinde Değişimi (Out-of-Sample)</h2>
            <p>Bu grafik, optimizasyonun her WFO adımında hangi parametreleri seçtiğini gösterir. Stabil çizgiler, stratejinin farklı piyasa koşullarında benzer ayarlara yöneldiğini gösterir.</p>
            {html_params_over_time}
            
            <hr>
            {robust_params_html}

            <hr>
            {summary_html}
            <hr>
            
            <h3>Genel Özet (Out-of-Sample)</h3>
            <p><strong>Toplam İşlem (Tüm OOS periyotları):</strong> {total_trades_wfo}</p>
            <p><strong>Kümülatif Getiri (Birleşik Bakiye):</strong> {total_return_wfo:.2f}%</p>
        </div>
        <div class="footer">Rapor Oluşturma Tarihi: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>
        </div></body></html>
        """
        
        wfo_filename = "whale_wfo_GELISMIS_raporu.html"
        with open(wfo_filename, "w", encoding="utf-8") as f:
            f.write(wfo_html_content)
        
        print(f"WFO Gelişmiş Raporu '{wfo_filename}' olarak kaydedildi.")
        webbrowser.open(f"file://{os.path.abspath(wfo_filename)}")

    else:
        print("Gelişmiş WFO analizi için yeterli veri yok (en az 2 adım gerekir).")
        # Eski JSON kaydını buraya taşı
        if all_oos_results:
             wfo_results_file = "whale_wfo_results.json"
             try:
                 with open(wfo_results_file, 'w') as f:
                     json.dump(all_oos_results, f, indent=4)
                 print(f"WFO sonuçları '{wfo_results_file}' dosyasına kaydedildi.")
             except Exception as e:
                 print(f"WFO sonuç kaydetme hatası: {e}")

    # -----------------------------------------------------
    # FİNAL BÖLÜM : EN SAĞLAM PARAMETRELERLE FİNAL TESTİ
    # -----------------------------------------------------
    print("\n" + "="*60)
    print(" BÖLÜM 6: EN SAĞLAM PARAMETRELERLE FİNAL TESTİ (TÜM VERİ)")
    print("="*60)

    # 'robust_params' WFO analizinde başarıyla oluşturulduysa ve boş değilse bu bloğu çalıştır
    if robust_params:
        try:
            print("\n🧪 En sağlam (robust) WFO parametreleri ile tüm veri üzerinde son bir backtest yapılıyor...")
            
            # Stratejiyi tüm veri ile yeniden başlat
            final_test_data = data.copy()
            # HATA DÜZELTMESİ: Final test için index'i resetleme, datetime index kalsın.
            # final_test_data.reset_index(drop=True, inplace=True) 
            strategy_final_robust = WhaleStopStrategy(final_test_data)
            
            # En sağlam parametreleri stratejiye uygula
            for key, value in robust_params.items():
                if hasattr(strategy_final_robust, key):
                    setattr(strategy_final_robust, key, value)
            
            # Stratejiyi çalıştır
            df_final, trades_final, signals_final = strategy_final_robust.run_strategy()
            
            # Sonuçları hesapla
            stats_final_robust = calculate_backtest_stats(trades_final)
            
            print("\n📊 ROBUST STRATEJİ FİNAL PERFORMANSI:")
            print(f"   • Toplam Kapalı İşlem: {stats_final_robust['num_trades']}")
            print(f"   • Getiri: {stats_final_robust['return_pct']:.2f}%")
            print(f"   • Win Rate: {stats_final_robust['win_rate']:.2f}%")
            print(f"   • Max DD: {stats_final_robust['max_drawdown']:.2f}%")

            # Ayrı bir HTML raporu oluştur
            timestamp_robust = datetime.now().strftime("%Y%m%d_%H%M%S")
            create_comprehensive_results_dashboard(
                robust_params,
                stats_final_robust,
                trades_final,
                f"ROBUST_{timestamp_robust}"
            )

        except Exception as e:
            print(f"❌ En sağlam parametrelerle final testi sırasında hata oluştu: {e}")
            traceback.print_exc()
    else:
        print("⚠️ En sağlam parametreler bulunamadığı veya WFO adımları yetersiz olduğu için final testi atlandı.")
    
    # -----------------------------------------------------
    # BÖLÜM 5: GELİŞMİŞ OPTUNA DASHBOARD (TAM VERİ İÇİN)
    # -----------------------------------------------------
    
    print("\n" + "="*50)
    print(" BÖLÜM 5: GELİŞMİŞ OPTUNA DASHBOARD (TAM VERİ)")
    print("="*50)
    
    def generate_optuna_dashboard(study, best_params, best_stats, filename="whale_optuna_dashboard.html"):
        """Gelişmiş, sekmeli Optuna dashboard'u oluşturur."""
        
        # Eğer hiç başarılı deneme yoksa iptal et
        if study is None or len(study.trials) == 0 or study.best_trial is None:
            print("❌ Optuna Dashboard oluşturulamadı: Geçerli çalışma (study) verisi yok.")
            return

        print(f"\n📊 Gelişmiş Optuna Dashboard hazırlanıyor... ({len(study.trials)} deneme analiz ediliyor)")

        # --- Grafik HTML'lerini oluştur ---
        html_plots = {}
        
        # 1. Optimizasyon Geçmişi (Hızlı)
        try:
            print("   • Grafik 1/4: Optimizasyon Geçmişi çiziliyor...")
            html_plots['history'] = vis.plot_optimization_history(study).to_html(include_plotlyjs=False, full_html=False)
        except Exception as e:
            print(f"     ⚠️ Geçmiş grafiği hatası: {e}")
            html_plots['history'] = "<p>Grafik oluşturulamadı.</p>"

        # 2. Paralel Koordinatlar (Orta Hız)
        try:
            print("   • Grafik 2/4: Paralel Koordinatlar çiziliyor...")
            html_plots['parallel'] = vis.plot_parallel_coordinate(study).to_html(include_plotlyjs=False, full_html=False)
        except Exception as e:
            print(f"     ⚠️ Paralel koordinat hatası: {e}")
            html_plots['parallel'] = "<p>Grafik oluşturulamadı.</p>"

        # 3. Parametre Dilimleri (Hızlı)
        try:
            print("   • Grafik 3/4: Dilim Grafikleri çiziliyor...")
            html_plots['slice'] = vis.plot_slice(study).to_html(include_plotlyjs=False, full_html=False)
        except Exception as e:
            print(f"     ⚠️ Dilim grafiği hatası: {e}")
            html_plots['slice'] = "<p>Grafik oluşturulamadı.</p>"
            
        # 4. Parametre Önemi (EN YAVAŞ KISIM BURASI)
        try:
            print("   • Grafik 4/4: Parametre Önemi hesaplanıyor (Bu işlem Scikit-Learn kullanır, biraz sürebilir)...")
            import sklearn # Kontrol amaçlı
            # Not: Eğer çok yavaşsa bu satırı devre dışı bırakabilirsin
            html_plots['importance'] = vis.plot_param_importances(study).to_html(include_plotlyjs=False, full_html=False)
        except ImportError:
            html_plots['importance'] = "<p><b>Parametre önemi grafiği için 'scikit-learn' kütüphanesi gerekli.</b><br><code>pip install scikit-learn</code> komutu ile kurabilirsiniz.</p>"
            print("     ⚠️ Scikit-learn yüklü değil, önem grafiği atlandı.")
        except Exception as e:
            html_plots['importance'] = f"<p><b>Parametre önemi grafiği oluşturulamadı.</b><br><small>Hata Detayı: {e}</small></p>"
            print(f"     ⚠️ Önem grafiği hatası (Normal olabilir): {e}")

        # --- HTML İçeriğini Oluştur ---
        print("   • HTML şablonu birleştiriliyor...")
        
        # Stil ve Script
        html_head = f"""
        <head>
            <meta charset="utf-8"><title>Whale Strategy Optuna Dashboard</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <style>
                body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif; background-color: #f4f6f9; color: #333; margin: 0; padding: 25px; }}
                .container {{ max-width: 1400px; margin: auto; }}
                h1 {{ color: #2c3e50; text-align: center; margin-bottom: 30px; font-weight: 600; }}
                .tabs {{ display: flex; border-bottom: 2px solid #dfe4ea; margin-bottom: 25px; }}
                .tab-button {{ background: none; border: none; padding: 15px 25px; cursor: pointer; font-size: 16px; font-weight: 500; color: #576574; position: relative; transition: color 0.3s; }}
                .tab-button.active {{ color: #4a69bd; }}
                .tab-button.active::after {{ content: ''; position: absolute; bottom: -2px; left: 0; right: 0; height: 2px; background-color: #4a69bd; }}
                .tab-content {{ display: none; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 5px 15px rgba(0,0,0,0.08); }}
                .tab-content.active {{ display: block; }}
                .summary-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 25px; }}
                .summary-card {{ background-color: #f8f9fa; border: 1px solid #e9ecef; border-radius: 8px; padding: 20px; }}
                .summary-card h3 {{ margin-top: 0; color: #4a69bd; border-bottom: 1px solid #dfe4ea; padding-bottom: 10px; }}
                .summary-card pre {{ background: #e9ecef; padding: 15px; border-radius: 5px; font-size: 13px; white-space: pre-wrap; word-wrap: break-word; }}
                .metrics-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 15px; text-align: center; }}
                .metric-item {{ background: white; padding: 15px; border-radius: 8px; box-shadow: 0 2px 5px rgba(0,0,0,0.05); }}
                .metric-item .value {{ font-size: 24px; font-weight: 600; color: #2c3e50; }}
                .metric-item .label {{ font-size: 14px; color: #576574; margin-top: 5px; }}
            </style>
        </head>
        """

        # Özet Sekmesi
        best_params_str = json.dumps(best_params, indent=2)
        summary_tab = f"""
        <div class="summary-grid">
            <div class="summary-card">
                <h3>🏆 En İyi Performans Metrikleri (Tam Veri)</h3>
                <div class="metrics-grid">
                    <div class="metric-item"><div class="value">{best_stats.get('return_pct', 0):.2f}%</div><div class="label">Getiri</div></div>
                    <div class="metric-item"><div class="value">{best_stats.get('win_rate', 0):.2f}%</div><div class="label">Win Rate</div></div>
                    <div class="metric-item"><div class="value">{best_stats.get('max_drawdown', 0):.2f}%</div><div class="label">Max Drawdown</div></div>
                    <div class="metric-item"><div class="value">{best_stats.get('num_trades', 0)}</div><div class="label">İşlem Sayısı</div></div>
                </div>
            </div>
            <div class="summary-card">
                <h3>⚙️ En İyi Parametreler</h3>
                <pre>{best_params_str}</pre>
            </div>
        </div>
        """

        # HTML Body
        html_body = f"""
        <body>
            <div class="container">
                <h1>🐋 Whale Strategy - Gelişmiş Optuna Dashboard (Tam Veri)</h1>
                <div class="tabs">
                    <button class="tab-button active" onclick="openTab(event, 'Summary')">Özet</button>
                    <button class="tab-button" onclick="openTab(event, 'History')">Optimizasyon Geçmişi</button>
                    <button class="tab-button" onclick="openTab(event, 'Importance')">Parametre Önemi</button>
                    <button class="tab-button" onclick="openTab(event, 'Parallel')">Paralel Koordinat</button>
                    <button class="tab-button" onclick="openTab(event, 'Slice')">Parametre Dilimleri</button>
                </div>

                <div id="Summary" class="tab-content active">{summary_tab}</div>
                <div id="History" class="tab-content">{html_plots.get('history', '')}</div>
                <div id="Importance" class="tab-content">{html_plots.get('importance', '')}</div>
                <div id="Parallel" class="tab-content">{html_plots.get('parallel', '')}</div>
                <div id="Slice" class="tab-content">{html_plots.get('slice', '')}</div>
            </div>

            <script>
                function openTab(evt, tabName) {{
                    var i, tabcontent, tablinks;
                    tabcontent = document.getElementsByClassName("tab-content");
                    for (i = 0; i < tabcontent.length; i++) {{ tabcontent[i].style.display = "none"; }}
                    tablinks = document.getElementsByClassName("tab-button");
                    for (i = 0; i < tablinks.length; i++) {{ tablinks[i].className = tablinks[i].className.replace(" active", ""); }}
                    document.getElementById(tabName).style.display = "block";
                    evt.currentTarget.className += " active";
                    var plotDivs = document.getElementById(tabName).getElementsByClassName('plotly-graph-div');
                    if(plotDivs.length > 0) {{ Plotly.Plots.resize(plotDivs[0]); }}
                }}
            </script>
        </body>
        """

        # Dosyaya yaz ve aç
        try:
            html_content = f"<!DOCTYPE html><html lang='tr'>{html_head}{html_body}</html>"
            dashboard_path = Path(filename)
            dashboard_path.write_text(html_content, encoding="utf-8")
            print(f"✅ Gelişmiş Dashboard kaydedildi: {filename}")
            print("🚀 Tarayıcıda açılıyor...")
            webbrowser.open(f"file://{os.path.abspath(str(dashboard_path))}")
        except Exception as e:
            print(f"❌ Dashboard dosyası yazılırken hata: {e}")

    # --- Dashboard'u Çalıştır ---
    # Not: Bölüm 1'de 'study_full' ve 'best_params_full' tanımlanmıştı.
    if 'study_full' in locals() and 'best_params_full' in locals() and 'final_stats' in locals():
        generate_optuna_dashboard(study_full, best_params_full, final_stats)
    else:
        print("⚠️ Dashboard için gerekli Bölüm 1 sonuçları bulunamadı (study_full eksik).")

















def create_comprehensive_results_dashboard(best_params, stats, trades, timestamp):
    """
    En iyi parametreleri ve backtest sonuçlarını içeren kapsamlı HTML dashboard
    """
    filename = f'optimization_dashboard_{timestamp}.html'
    print(f"\n📋 Kapsamlı sonuç dashboard'u oluşturuluyor: {filename}")

    # Calculate win/loss ratio safely to avoid f-string formatting errors
    win_loss_ratio = (
        f"{stats['wins'] / stats['losses']:.2f}" if stats['losses'] > 0 else "∞"
    )
    
    # HTML içeriği oluştur
    html_content = f"""
<!DOCTYPE html>
<html lang="tr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Whale Strategy - Optimization Results</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
            color: #fff;
            padding: 20px;
            min-height: 100vh;
        }}
        
        .container {{
            max-width: 1400px;
            margin: 0 auto;
        }}
        
        .header {{
            text-align: center;
            padding: 40px 20px;
            background: rgba(255,255,255,0.1);
            border-radius: 20px;
            margin-bottom: 30px;
            backdrop-filter: blur(10px);
            box-shadow: 0 8px 32px rgba(0,0,0,0.3);
        }}
        
        .header h1 {{
            font-size: 3em;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }}
        
        .header .emoji {{
            font-size: 4em;
            margin-bottom: 20px;
        }}
        
        .header .timestamp {{
            font-size: 1.1em;
            opacity: 0.8;
            margin-top: 10px;
        }}
        
        .grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        
        .card {{
            background: rgba(255,255,255,0.1);
            border-radius: 15px;
            padding: 25px;
            backdrop-filter: blur(10px);
            box-shadow: 0 8px 32px rgba(0,0,0,0.2);
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }}
        
        .card:hover {{
            transform: translateY(-5px);
            box-shadow: 0 12px 40px rgba(0,0,0,0.3);
        }}
        
        .card h2 {{
            font-size: 1.5em;
            margin-bottom: 20px;
            border-bottom: 2px solid rgba(255,255,255,0.3);
            padding-bottom: 10px;
        }}
        
        .metric {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 12px 0;
            border-bottom: 1px solid rgba(255,255,255,0.1);
        }}
        
        .metric:last-child {{
            border-bottom: none;
        }}
        
        .metric-label {{
            font-size: 1.1em;
            opacity: 0.9;
        }}
        
        .metric-value {{
            font-size: 1.3em;
            font-weight: bold;
            color: #4CAF50;
        }}
        
        .metric-value.negative {{
            color: #f44336;
        }}
        
        .metric-value.neutral {{
            color: #FFC107;
        }}
        
        .stat-big {{
            text-align: center;
            padding: 30px;
            background: linear-gradient(135deg, rgba(76,175,80,0.3) 0%, rgba(33,150,243,0.3) 100%);
            border-radius: 15px;
            margin: 10px 0;
        }}
        
        .stat-big .value {{
            font-size: 3em;
            font-weight: bold;
            margin: 10px 0;
        }}
        
        .stat-big .label {{
            font-size: 1.2em;
            opacity: 0.9;
        }}
        
        .param-table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 15px;
        }}
        
        .param-table th,
        .param-table td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid rgba(255,255,255,0.1);
        }}
        
        .param-table th {{
            background: rgba(255,255,255,0.1);
            font-weight: bold;
        }}
        
        .param-table tr:hover {{
            background: rgba(255,255,255,0.05);
        }}
        
        .badge {{
            display: inline-block;
            padding: 5px 15px;
            border-radius: 20px;
            font-size: 0.9em;
            font-weight: bold;
        }}
        
        .badge.success {{
            background: #4CAF50;
        }}
        
        .badge.warning {{
            background: #FFC107;
            color: #000;
        }}
        
        .badge.error {{
            background: #f44336;
        }}
        
        .badge.info {{
            background: #2196F3;
        }}
        
        .trades-section {{
            margin-top: 30px;
        }}
        
        .trade-item {{
            background: rgba(255,255,255,0.05);
            padding: 15px;
            border-radius: 10px;
            margin: 10px 0;
            border-left: 4px solid #4CAF50;
        }}
        
        .trade-item.exit {{
            border-left-color: #FFC107;
        }}
        
        .trade-item.short {{
            border-left-color: #f44336;
        }}
        
        .progress-bar {{
            width: 100%;
            height: 30px;
            background: rgba(255,255,255,0.1);
            border-radius: 15px;
            overflow: hidden;
            margin: 10px 0;
        }}
        
        .progress-fill {{
            height: 100%;
            background: linear-gradient(90deg, #4CAF50, #8BC34A);
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: bold;
            transition: width 0.3s ease;
        }}
        
        .summary-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }}
        
        .summary-card {{
            background: rgba(255,255,255,0.1);
            padding: 20px;
            border-radius: 10px;
            text-align: center;
        }}
        
        .summary-card .icon {{
            font-size: 2em;
            margin-bottom: 10px;
        }}
        
        .summary-card .value {{
            font-size: 2em;
            font-weight: bold;
            margin: 10px 0;
        }}
        
        .summary-card .label {{
            font-size: 1em;
            opacity: 0.8;
        }}
    </style>
</head>
<body>
    <div class="container">
        <!-- Header -->
        <div class="header">
            <div class="emoji">🐋</div>
            <h1>Whale Strategy Optimization Results</h1>
            <div class="timestamp">⏰ {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}</div>
        </div>
        
        <!-- Ana Metrikler -->
        <div class="summary-grid">
            <div class="summary-card">
                <div class="icon">💰</div>
                <div class="value" style="color: {'#4CAF50' if stats['total_return'] > 0 else '#f44336'}">${stats['total_return']:,.2f}</div>
                <div class="value" style="color: {color_ret}">${stats['total_return']:,.2f}</div>
                <div class="label">Toplam Kar/Zarar</div>
            </div>
            
            <div class="summary-card">
                <div class="icon">📈</div>
                <div class="value" style="color: {'#4CAF50' if stats['return_pct'] > 0 else '#f44336'}">%{stats['return_pct']:.2f}</div>
                <div class="value" style="color: {color_ret_pct}">%{stats['return_pct']:.2f}</div>
                <div class="label">Getiri Oranı</div>
            </div>
            
            <div class="summary-card">
                <div class="icon">🎯</div>
                <div class="value" style="color: {'#4CAF50' if stats['win_rate'] > 50 else '#f44336'}">%{stats['win_rate']:.2f}</div>
                <div class="value" style="color: {color_win}">%{stats['win_rate']:.2f}</div>
                <div class="label">Kazanma Oranı</div>
            </div>
            
            <div class="summary-card">
                <div class="icon">📉</div>
                <div class="value" style="color: {'#4CAF50' if stats['max_drawdown'] < 20 else '#f44336'}">%{stats['max_drawdown']:.2f}</div>
                <div class="value" style="color: {color_dd}">%{stats['max_drawdown']:.2f}</div>
                <div class="label">Max Drawdown</div>
            </div>
            
            <div class="summary-card">
                <div class="icon">🔢</div>
                <div class="value">{stats['num_trades']}</div>
                <div class="label">Toplam İşlem</div>
            </div>
            
            <div class="summary-card">
                <div class="icon">✅</div>
                <div class="value" style="color: #4CAF50">{stats['wins']}</div>
                <div class="label">Kazanan</div>
            </div>
            
            <div class="summary-card">
                <div class="icon">❌</div>
                <div class="value" style="color: #f44336">{stats['losses']}</div>
                <div class="label">Kaybeden</div>
            </div>
            
            <div class="summary-card">
                <div class="icon">⚖️</div>
                <div class="value">{win_loss_ratio}</div>
                <div class="label">Win/Loss Ratio</div>
            </div>
        </div>
        
        <!-- Win Rate Progress Bar -->
        <div class="card">
            <h2>📊 Kazanma Oranı Göstergesi</h2>
            <div class="progress-bar">
                <div class="progress-fill" style="width: {stats['win_rate']}%">
                    %{stats['win_rate']:.1f}
                </div>
            </div>
        </div>
        
        <!-- Detaylı Parametreler -->
        <div class="card">
            <h2>⚙️ En İyi Çıkış Parametreleri</h2>
            <table class="param-table">
                <thead>
                    <tr>
                        <th>Parametre</th>
                        <th>Değer</th>
                        <th>Durum</th>
                    </tr>
                </thead>
                <tbody>"""
    
    # Parametreleri kategorilere ayır ve güzel göster
    param_categories = {
        'Pozisyon Büyütme': ['position_multiplier'],
        'Take Profit': ['enable_take_profit', 'take_profit_perc'],
        'Trailing Stop': ['enable_trailing_stop', 'trailing_perc', 'enable_min_position_for_ts', 'min_position_for_ts'],
        'Maliyet Çıkışı': ['enable_maliyet_exit', 'min_entry_for_maliyet', 'maliyet_return_perc'],
        'Mini Loss': ['enable_mini_stop', 'min_entry_for_mini_loss', 'maliyet_stop_perc', 'mini_loss_cooldown_bars'],
        'DCA': ['enable_dca', 'max_dca', 'dca_drop_perc', 'dca_mum_delay'],
        'Whale Stop': ['enable_drop_speed', 'drop_speed_perc', 'drop_speed_bars', 'enable_panic_drop', 'panic_drop_perc', 'panic_drop_bars', 'panic_cooldown_bars']
    }

    
    param_labels = {
        'enable_take_profit': 'Take Profit Aktif',
        'take_profit_perc': 'TP Yüzdesi',
        'enable_trailing_stop': 'Trailing Stop Aktif',
        'trailing_perc': 'Trailing Yüzdesi',
        'enable_min_position_for_ts': 'Min Pozisyon Kontrolü',
        'min_position_for_ts': 'Min Pozisyon Sayısı',
        'position_multiplier': 'Pozisyon Çarpanı',
        'enable_dca': 'DCA Aktif',
        'max_dca': 'Maksimum DCA Sayısı',
        'dca_drop_perc': 'DCA Düşüş Yüzdesi',
        'dca_mum_delay': 'DCA Mum Gecikmesi',
        'enable_maliyet_exit': 'Maliyet Çıkışı Aktif',
        'min_entry_for_maliyet': 'Maliyet İçin Min Entry',
        'maliyet_return_perc': 'Maliyet Return %',
        'enable_mini_stop': 'Mini Loss Aktif',
        'min_entry_for_mini_loss': 'Mini Loss İçin Min Entry',
        'maliyet_stop_perc': 'Mini Loss Stop %',
        'mini_loss_cooldown_bars': 'Mini Loss Cooldown',
        'enable_drop_speed': 'Drop Speed Aktif',
        'drop_speed_perc': 'Drop Speed %',
        'drop_speed_bars': 'Drop Speed Barlar',
        'enable_panic_drop': 'Panic Drop Aktif',
        'panic_drop_perc': 'Panic Drop %',
        'panic_drop_bars': 'Panic Drop Barlar',
        'panic_cooldown_bars': 'Panic Cooldown'
    }
    
    for category, params in param_categories.items():
        html_content += f"""
                    <tr>
                        <td colspan="3" style="background: rgba(33,150,243,0.2); font-weight: bold; font-size: 1.1em;">
                            {category}
                        </td>
                    </tr>"""
        
        for param in params:
            if param in best_params:
                value = best_params[param]
                label = param_labels.get(param, param)
                
                # Değer formatı
                if isinstance(value, bool):
                    display_value = '✅ Aktif' if value else '❌ Pasif'
                    badge_class = 'success' if value else 'error'
                elif isinstance(value, float):
                    display_value = f'{value:.2f}'
                    badge_class = 'info'
                else:
                    display_value = str(value)
                    badge_class = 'info'
                
                html_content += f"""
                    <tr>
                        <td>{label}</td>
                        <td><strong>{display_value}</strong></td>
                        <td><span class="badge {badge_class}">Optimize Edildi</span></td>
                    </tr>"""
    
    html_content += """
                </tbody>
            </table>
        </div>
        
        <!-- İşlem Detayları -->
        <div class="card trades-section">
            <h2>📋 İşlem Geçmişi (Son 20 İşlem)</h2>"""
    
    # Son 20 işlemi göster
    recent_trades = trades[-20:] if len(trades) > 20 else trades
    
    for idx, trade in enumerate(reversed(recent_trades), 1):
        trade_class = 'trade-item'
        if trade['type'] == 'EXIT':
            trade_class += ' exit'
        elif trade.get('direction') == 'SHORT':
            trade_class += ' short'
        
        html_content += f"""
            <div class="{trade_class}">
                <strong>#{len(trades) - idx + 1}</strong> - 
                {trade['type']} {trade.get('direction', '')} @ 
                <strong>${trade['price']:.4f}</strong> | 
                Size: {trade['size']:.4f} | 
                {trade.get('comment', 'N/A')}
            </div>"""
    
    html_content += """
        </div>
        
        <!-- Footer -->
        <div class="card" style="text-align: center; margin-top: 30px;">
            <h3>🎯 Optimizasyon Başarıyla Tamamlandı!</h3>
            <p style="margin-top: 15px; opacity: 0.8;">
                Bu sonuçlar Optuna optimizasyon algoritması kullanılarak elde edilmiştir.
            </p>
        </div>
    </div>
</body>
</html>"""
    
    # HTML dosyasını kaydet
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"✅ Kapsamlı dashboard kaydedildi: {filename}")
    
    # Otomatik aç
    webbrowser.open('file://' + os.path.abspath(filename))
    
    return filename