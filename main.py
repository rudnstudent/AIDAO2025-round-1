import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import List, Tuple
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.linear_model import Ridge
from tqdm import tqdm
import math
import warnings
import pickle
import json
import random

np.random.seed(42)
random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

warnings.filterwarnings("ignore")

# --- Шаг 1: Конфиг (настраиваемые параметры в одном месте для удобства) ---

CONFIG = {
    "file_path": "frames_errors.csv",
    "target_feature": "E_mu_Z", # Целевая переменная
    "all_columns": [
        "block_id", "frame_idx", "E_mu_Z", "E_mu_phys_est", "E_mu_X", "E_nu1_X",
        "E_nu2_X", "E_nu1_Z", "E_nu2_Z", "N_mu_X", "M_mu_XX", "M_mu_XZ", "M_mu_X",
        "N_mu_Z", "M_mu_ZZ", "M_mu_Z", "N_nu1_X", "M_nu1_XX", "M_nu1_XZ", "M_nu1_X",
        "N_nu1_Z", "M_nu1_ZZ", "M_nu1_Z", "N_nu2_X", "M_nu2_XX", "M_nu2_XZ", "M_nu2_X",
        "N_nu2_Z", "M_nu2_ZZ", "M_nu2_Z", "nTot", "bayesImVoltage", "opticalPower",
        "polarizerVoltages[0]", "polarizerVoltages[1]", "polarizerVoltages[2]",
        "polarizerVoltages[3]", "temp_1", "biasVoltage_1", "temp_2", "biasVoltage_2",
        "synErr", "N_EC_rounds", "maintenance_flag", "estimator_name", "f_EC",
        "E_mu_Z_est", "R", "s", "p"
    ],
    # Основные признаки, отобранные ранее (по матрице корреляции)
    "feature_columns": [
        'E_mu_Z', 'E_mu_Z_est', 'E_nu1_Z', 'M_mu_XZ', 'M_nu1_XZ',
        'M_mu_Z', 'polarizerVoltages_2', 'polarizerVoltages_3',
        'M_nu1_XX', 'M_nu2_XX', 'M_mu_XX', 'temp_1', 'temp_2',
        'biasVoltage_1', 'biasVoltage_2'
    ],
    "id_columns": ['block_id', 'frame_idx'],

    # Границы для предсказания (как просят в ответе)
    "predict_start": {"block_id": 1489460492, "frame_idx": 99},
    "predict_end": {"block_id": 1840064900, "frame_idx": 101},

    # Параметры для обучения
    "validation_size": 0.15,
    "patience": 12,     # Сколько эпох ждать улучшения перед остановкой (счетчик терпения)

    # Параметры Transformer
    "transformer_params": {
    "seq_len": 160,  # Длина истории для модели (look-back window)
    "d_model": 64,  # Размерность эмбеддинга внутри трансформера
    "nhead": 4,  # Количество "голов" в Multi-Head Attention
    "num_encoder_layers": 2,  # Количество слоев трансформера
    "dropout": 0.1,
    "epochs": 150,  # Количество эпох обучения
    "batch_size": 32,
    "lr": 0.0005,  # Скорость обучения
    },

    # Параметры LightGBM
    "lgbm_params": {
        'objective': 'regression_l1',
        'metric': 'mae',
        'n_estimators': 5000, # Максимальное количество деревьев. Реальное число будет меньше из-за early stopping
        'learning_rate': 0.01,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 1,
        'lambda_l1': 0.1,
        'lambda_l2': 0.1,
        'num_leaves': 31,
        'verbose': -1,
        'n_jobs': -1,
        'seed': 42,
        'boosting_type': 'gbdt',
        'device': 'gpu',  # Используем GPU!!!
    },

    # Параметры CatBoost
    "catboost_params": {
        'iterations': 5000,
        'learning_rate': 0.02,
        'depth': 6,
        'l2_leaf_reg': 3,
        'loss_function': 'MAE',
        'eval_metric': 'MAE',
        'random_seed': 42,
        'verbose': 0,
        'task_type': 'GPU', # Используем GPU!
    },

    # Коэффициенты для финального расчета
    "f_ec": 1.15,
    "alpha_ema": 0.33, # Коэффициент сглаживания для экспоненциального скользящего среднего (EMA)

    # Централизованное хранение путей для сохранения артефактов модели
    "save_paths": {
        "transformer": "transformer_model.pth",
        "lightgbm": "lgbm_model.pkl",
        "catboost": "catboost_model.pkl",
        "meta_model": "meta_model.pkl",
        "preprocessor": "preprocessor.pkl",
        "config": "config.json"
    },
}

# Используем GPU, если доступно
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Используемое устройство: {device}")


# --- Шаг 2: Класс предобработки данных ---

class DataPreprocessor:
    def __init__(self, feature_cols: List[str], id_cols: List[str]):
        self.feature_cols = feature_cols
        self.id_cols = id_cols
        self.scaler = StandardScaler()  # StandardScaler приводит данные к нулевому среднему и единичной дисперсии
        self.is_fitted = False  # Флаг, показывающий, был ли scaler уже обучен.

    def _load_and_clean_data(self, file_path: str, all_columns: List[str]) -> pd.DataFrame:
        # Загружаем данные без заголовков и присваиваем им имена из конфига.
        df = pd.read_csv(file_path, header=None, names=all_columns)

        # Создаем словарь для переименования 'old_name[0]' в 'old_name_0' (т.к. LightGBM ругается на квадратные скобки)
        clean_columns = {col: col.replace('[', '_').replace(']', '') for col in df.columns}
        df = df.rename(columns=clean_columns)

        # Выбираем только необходимые для модели колонки.
        df_selected = df[self.id_cols + self.feature_cols].copy()
        # Приводим все признаки к числовому формату, заменяя некорректные значения на NaN.
        for col in self.feature_cols:
            df_selected[col] = pd.to_numeric(df_selected[col], errors='coerce')
        # Заполняем пропуски: сначала предыдущим известным значением (ffill), а оставшиеся в начале - следующим (bfill)
        df_selected[self.feature_cols] = df_selected[self.feature_cols].ffill().bfill()
        return df_selected

    def split_data(self, df: pd.DataFrame, start_marker: dict, end_marker: dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
        # Находим индекс начала и конца тестового периода по заданным block_id и frame_idx.
        start_idx = df.index[
            (df['block_id'] == start_marker['block_id']) & (df['frame_idx'] == start_marker['frame_idx'])].tolist()[0]
        end_idx = \
        df.index[(df['block_id'] == end_marker['block_id']) & (df['frame_idx'] == end_marker['frame_idx'])].tolist()[0]

        # Разделяем данные на те, что были до тестового периода (для обучения), и сам тестовый период (для предсказания)
        train_df = df.iloc[:start_idx]
        predict_df = df.iloc[start_idx: end_idx + 1]
        # Проверка, чтобы убедиться, что мы выбрали правильный срез данных для предсказания
        assert len(predict_df) == 2000
        return train_df, predict_df

    # Метод для обучения скейлера и трансформации данных. Вызывается только на обучающих данных
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        scaled_features = self.scaler.fit_transform(df[self.feature_cols])
        scaled_df = df[self.id_cols].copy()
        scaled_df[self.feature_cols] = scaled_features
        self.is_fitted = True
        return scaled_df

    # Метод для трансформации данных с использованием уже обученного скейлера
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        scaled_features = self.scaler.transform(df[self.feature_cols])
        scaled_df = df[self.id_cols].copy()
        scaled_df[self.feature_cols] = scaled_features
        return scaled_df


# --- Шаг 3: Реализация модели Transformer ---

class PositionalEncoding(nn.Module):
    # Стандартная реализация Positional Encoding
    # Этот модуль добавляет к эмбеддингам информацию о позиции каждого элемента, используя синусы и косинусы
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Добавляем позиционную информацию к входному тензору.
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class TimeSeriesTransformer(nn.Module):
    def __init__(self, num_features: int, d_model: int, nhead: int, num_encoder_layers: int, dropout: float):
        super().__init__()
        self.model_type = 'Transformer'
        # 1. Входной слой: преобразует вектор признаков в эмбеддинг нужной размерности (d_model)
        self.input_embedding = nn.Linear(num_features, d_model)
        # 2. Позиционное кодирование
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        # 3. Основная часть - энкодер Трансформера
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, d_model * 4, dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_encoder_layers)
        # 4. Выходной слой: преобразует обработанный эмбеддинг в одно число - предсказание
        self.output_layer = nn.Linear(d_model, 1)

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        src = self.input_embedding(src)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        # Для предсказания следующей точки нам нужен выход трансформера только для последнего элемента последовательности
        # Срез `output[:, -1, :]` как раз и берет этот последний временной шаг
        output = self.output_layer(output[:, -1, :])
        return output


class TimeSeriesDataset(Dataset):
    # Dataset для PyTorch для нарезки временных рядов на окна
    def __init__(self, data: pd.DataFrame, feature_cols: List[str], target_col: str, seq_len: int):
        self.seq_len = seq_len
        self.features = torch.tensor(data[feature_cols].values, dtype=torch.float32)
        self.target = torch.tensor(data[target_col].values, dtype=torch.float32)

    def __len__(self):
        # Общее количество возможных последовательностей, которые можно "нарезать"
        return len(self.features) - self.seq_len - 1

    def __getitem__(self, idx):
        # Вход: последовательность длиной seq_len
        x = self.features[idx: idx + self.seq_len]

        # Цель (y): значение целевой переменной на СЛЕДУЮЩЕМ временном шаге после последовательности
        y = self.target[idx + self.seq_len]

        return x, y


# --- Шаг 4: Feature Engineering для LightGBM ---

def create_lgbm_features(df, target_col, feature_cols, id_cols):
    """
    Создает РАСШИРЕННЫЙ набор признаков для LightGBM/CatBoost, включая:
    - Лаги
    - Скользящие средние
    - Статистические показатели (std, median, min, max)
    - Локальный тренд
    """
    print("Создание РАСШИРЕННЫХ признаков для LightGBM/CatBoost...")
    df_features = df[id_cols + feature_cols].copy()

    # --- 1. Лаги (значения из недавнего прошлого) ---
    # Помогают модели увидеть, какими были значения целевой переменной N шагов назад
    lags = [1, 2, 3, 5, 10, 20, 40]
    for lag in lags:
        df_features[f'{target_col}_lag_{lag}'] = df_features[target_col].shift(lag)

    # --- 2. Скользящие окна (тренды и статистики) ---
    windows = [10, 30, 80, 160]  # Короткие, средние и длинные окна
    # Признаки для расчета статистик
    features_to_stat = [target_col, 'E_mu_Z_est', 'temp_1', 'polarizerVoltages_3']

    for feature in features_to_stat:
        for window in windows:
            # Сдвигаем на 1, чтобы использовать только прошлые данные
            shifted_series = df_features[feature].shift(1)
            rolling_window = shifted_series.rolling(window)

            # Базовые статистики: среднее, стандартное отклонение, медиана, мин/макс
            df_features[f'{feature}_roll_mean_{window}'] = rolling_window.mean()
            df_features[f'{feature}_roll_std_{window}'] = rolling_window.std()
            df_features[f'{feature}_roll_median_{window}'] = rolling_window.median()
            df_features[f'{feature}_roll_min_{window}'] = rolling_window.min()
            df_features[f'{feature}_roll_max_{window}'] = rolling_window.max()

            # --- 3. Локальный тренд ---
            # Вычисляем наклон линии регрессии внутри окна
            if feature == target_col and window in [30, 80]:
                df_features[f'{target_col}_roll_trend_{window}'] = rolling_window.apply(
                    lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0,
                    raw=True
                )

    print("Признаки созданы. Удаляем строки с NaN...")
    # Заполняем возможные оставшиеся NaN нулями (может случиться после .std() на константах)
    df_features = df_features.fillna(0)
    # Удаляем строки, где лаги и окна не смогли посчитаться (в начале датасета)
    # Для этого найдем максимальный лаг/окно
    max_lookback = max(lags + windows)
    return df_features.iloc[max_lookback:].copy()


# --- Шаг 5: Функции для финального расчета ---
# ... (Копируем функции из baseline) ...

def calculate_ema(prev_ema, current_value, alpha):
    """Экспоненциальное скользящее среднее для сглаживания предсказаний"""
    if prev_ema is None: return current_value
    return alpha * current_value + (1 - alpha) * prev_ema


def h(x):
    """Энтропийная функция, используемая в расчетах"""
    if x > 0: return -x * np.log2(x) - (1 - x) * np.log2(1 - x)
    return 0.0


def select_code_rate(e_mu, f_ec, rates, frame_len=32000, sp_count=4800):
    """Выбор оптимальных параметров R, s, p на основе предсказанного значения"""
    r_candidate = 1 - h(e_mu) * f_ec
    R_res, s_n, p_n = 0.50, sp_count, 0
    for R in rates:
        p_n_cand = int(math.ceil((1 - R) * frame_len - (1 - r_candidate) * (frame_len - sp_count)))
        s_n_cand = int(sp_count - p_n_cand)
        if p_n_cand >= 0 and s_n_cand >= 0:
            return round(R, 2), s_n_cand, p_n_cand
    return round(R_res, 2), s_n, p_n


# --- Основной скрипт ---

if __name__ == "__main__":
    # --- 1. ПОДГОТОВКА ДАННЫХ ---
    feature_cols = CONFIG['feature_columns']
    id_cols = CONFIG['id_columns']
    preprocessor = DataPreprocessor(feature_cols, id_cols)
    # Загружаем и чистим данные
    cleaned_df = preprocessor._load_and_clean_data(CONFIG['file_path'], CONFIG['all_columns'])
    # Разделяем на полный трейн и данные для предсказания
    full_train_raw_df, predict_raw_df = preprocessor.split_data(cleaned_df, CONFIG['predict_start'],
                                                                CONFIG['predict_end'])

    # Отделяем от полного трейна валидационную выборку (последние 15% данных)
    val_size = int(len(full_train_raw_df) * CONFIG['validation_size'])
    train_raw_df = full_train_raw_df.iloc[:-val_size]
    validation_raw_df = full_train_raw_df.iloc[-val_size:]
    print(
        f"\nДанные разделены: Train={len(train_raw_df)}, Validation={len(validation_raw_df)}, Predict={len(predict_raw_df)}")

    # Обучаем скейлер на обучающей выборке и трансформируем ее.
    scaled_train_df = preprocessor.fit_transform(train_raw_df)
    # Трансформируем валидационную выборку, используя УЖЕ обученный скейлер
    scaled_validation_df = preprocessor.transform(validation_raw_df)
    # Создаем один большой масштабированный датафрейм для удобного доступа к истории при предсказании
    full_df_scaled = pd.concat([preprocessor.transform(full_train_raw_df), preprocessor.transform(predict_raw_df)],
                               ignore_index=True)

    # --- 2. ОБУЧЕНИЕ БАЗОВЫХ МОДЕЛЕЙ (УРОВЕНЬ 0) ---
    # Мы используем ансамбль из трех разных моделей: Transformer, LightGBM, CatBoost

    print("\n--- Начало обучения Transformer ---")
    tp = CONFIG['transformer_params']
    model = TimeSeriesTransformer(num_features=len(feature_cols), **{k: v for k, v in tp.items() if
                                                                     k in ['d_model', 'nhead', 'num_encoder_layers',
                                                                           'dropout']}).to(device)
    # Создаем датасеты и загрузчики данных для PyTorch
    train_dataset = TimeSeriesDataset(scaled_train_df, feature_cols, 'E_mu_Z', tp['seq_len'])
    train_loader = DataLoader(train_dataset, batch_size=tp['batch_size'], shuffle=True)
    validation_dataset = TimeSeriesDataset(scaled_validation_df, feature_cols, 'E_mu_Z', tp['seq_len'])
    validation_loader = DataLoader(validation_dataset, batch_size=tp['batch_size'], shuffle=False)

    criterion = nn.L1Loss() # Функция потерь (MAE)
    optimizer = torch.optim.Adam(model.parameters(), lr=tp['lr']) # Оптимизатор

    best_val_loss = float('inf')
    epochs_no_improve = 0

    # Цикл обучения
    for epoch in range(tp['epochs']):
        model.train()
        total_train_loss = 0
        for x_batch, y_batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{tp['epochs']} [Train]"):
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()  # Обнуляем градиенты
            output = model(x_batch)
            loss = criterion(output.squeeze(), y_batch) # Считаем ошибку
            loss.backward() # Обратное распространение ошибки
            optimizer.step() # Обновляем веса модели
            total_train_loss += loss.item()
        avg_train_loss = total_train_loss / len(train_loader)

        # Валидация после каждой эпохи
        model.eval()
        total_val_loss = 0
        with torch.no_grad(): # Отключаем расчет градиентов для ускорения
            for x_val, y_val in tqdm(validation_loader, desc=f"Epoch {epoch + 1}/{tp['epochs']} [Val]"):
                x_val, y_val = x_val.to(device), y_val.to(device)
                output_val = model(x_val)
                val_loss = criterion(output_val.squeeze(), y_val)
                total_val_loss += val_loss.item()
        avg_val_loss = total_val_loss / len(validation_loader)
        print(f"Epoch {epoch + 1}: Train Loss={avg_train_loss:.6f}, Val Loss={avg_val_loss:.6f}")

        # Логика Early Stopping и сохранения лучшей модели
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), CONFIG['save_paths']['transformer'])
            print(f"  -> Лучшая модель найдена! Сохранение {CONFIG['save_paths']['transformer']}")
        else:
            epochs_no_improve += 1
            print(f"  -> Счетчик терпения {epochs_no_improve}/{CONFIG['patience']}")
        if epochs_no_improve >= CONFIG['patience']:
            print("Ранняя остановка!")
            break

    # Загружаем веса лучшей модели для дальнейшего использования
    model.load_state_dict(torch.load(CONFIG['save_paths']['transformer']))
    model.eval()

    print("\n--- Начало обучения LightGBM и CatBoost ---")
    # Создаем расширенный набор признаков для бустинговых моделей
    lgbm_full_df = create_lgbm_features(cleaned_df, CONFIG['target_feature'], feature_cols, id_cols)
    lgbm_features = [c for c in lgbm_full_df.columns if c not in id_cols + [CONFIG['target_feature']]]

    # Находим соответствующие индексы в новом датафрейме с фичами
    train_indices = train_raw_df.index.intersection(lgbm_full_df.index)
    val_indices = validation_raw_df.index.intersection(lgbm_full_df.index)

    X_train, y_train = lgbm_full_df.loc[train_indices, lgbm_features], lgbm_full_df.loc[
        train_indices, CONFIG['target_feature']]
    X_val, y_val = lgbm_full_df.loc[val_indices, lgbm_features], lgbm_full_df.loc[val_indices, CONFIG['target_feature']]

    # Обучаем LightGBM с использованием early stopping
    lgbm = lgb.LGBMRegressor(**CONFIG['lgbm_params'])
    lgbm.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(100, verbose=False)])

    # Обучаем CatBoost с использованием early stopping
    cat = CatBoostRegressor(**CONFIG['catboost_params'])
    cat.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=100)

    # --- 3. ОБУЧЕНИЕ МЕТА-МОДЕЛИ (УРОВЕНЬ 1) ---
    # Стекинг. Мы берем предсказания базовых моделей как новые признаки
    # и обучаем на них простую "мета-модель" (здесь - Ridge регрессия), которая учится
    # оптимально комбинировать их предсказания.
    print("\n--- Обучение мета-модели на OOF-предсказаниях ---")

    # Генерируем предсказания базовых моделей на валидационной выборке.
    transformer_preds_val = []
    val_start_idx = len(train_raw_df)
    with torch.no_grad():
        for i in tqdm(range(len(validation_raw_df)), desc="Transformer OOF Preds"):
            history_start = val_start_idx + i - tp['seq_len']
            history_end = val_start_idx + i
            history = preprocessor.transform(full_train_raw_df.iloc[history_start:history_end])[feature_cols].values
            history_tensor = torch.tensor(history, dtype=torch.float32).unsqueeze(0).to(device)
            pred_scaled = model(history_tensor).cpu().item()
            transformer_preds_val.append(pred_scaled)

    # Возвращаем предсказания к исходному масштабу.
    dummy_array = np.zeros((len(transformer_preds_val), len(feature_cols)))
    dummy_array[:, feature_cols.index(CONFIG['target_feature'])] = transformer_preds_val
    transformer_preds_unscaled_val = preprocessor.scaler.inverse_transform(dummy_array)[:,
                                     feature_cols.index(CONFIG['target_feature'])]

    lgbm_preds_val = lgbm.predict(X_val)
    cat_preds_val = cat.predict(X_val)

    # Собираем предсказания в одну матрицу признаков для мета-модели
    transformer_preds_series = pd.Series(transformer_preds_unscaled_val, index=validation_raw_df.index).loc[
        y_val.index].values
    meta_features_train = np.column_stack([transformer_preds_series, lgbm_preds_val, cat_preds_val])

    # Обучаем мета-модель
    meta_model = Ridge(random_state=42)
    meta_model.fit(meta_features_train, y_val.values)
    print("Мета-модель обучена. Коэффициенты:", meta_model.coef_) # Коэффициенты показывают "вес" каждой базовой модели

    # --- 4. ГЕНЕРАЦИЯ ФИНАЛЬНЫХ ПРЕДСКАЗАНИЙ ---
    # Повторяем тот же пайплайн, что и для валидации, но уже на тестовых данных
    print("\n--- Генерация финальных предсказаний ---")
    predict_indices = predict_raw_df.index.intersection(lgbm_full_df.index)
    X_predict = lgbm_full_df.loc[predict_indices, lgbm_features]

    # Предсказания трансформера
    transformer_preds_test = []
    predict_start_idx = len(full_train_raw_df)
    with torch.no_grad():
        for i in tqdm(range(len(predict_raw_df)), desc="Transformer Test Preds"):
            history_start = predict_start_idx + i - tp['seq_len']
            history_end = predict_start_idx + i
            history = full_df_scaled.iloc[history_start:history_end][feature_cols].values
            history_tensor = torch.tensor(history, dtype=torch.float32).unsqueeze(0).to(device)
            pred_scaled = model(history_tensor).cpu().item()
            transformer_preds_test.append(pred_scaled)
    dummy_array = np.zeros((len(transformer_preds_test), len(feature_cols)))
    dummy_array[:, feature_cols.index(CONFIG['target_feature'])] = transformer_preds_test
    transformer_preds_unscaled_test = preprocessor.scaler.inverse_transform(dummy_array)[:,
                                      feature_cols.index(CONFIG['target_feature'])]

    # Предсказания LightGBM и CatBoost
    lgbm_preds_test = lgbm.predict(X_predict)
    cat_preds_test = cat.predict(X_predict)

    # Собираем предсказания базовых моделей в матрицу признаков
    transformer_preds_series_test = pd.Series(transformer_preds_unscaled_test, index=predict_raw_df.index).loc[
        X_predict.index].values
    meta_features_test = np.column_stack([transformer_preds_series_test, lgbm_preds_test, cat_preds_test])

    # Получаем финальные предсказания от мета-модели
    final_preds = meta_model.predict(meta_features_test)

    # Из-за создания лаговых фич мы не можем сделать предсказания для первых N точек тестовой выборки.
    # Заполняем пропуски первым доступным предсказанием
    num_missing = 2000 - len(final_preds)
    padded_preds = np.pad(final_preds, (num_missing, 0), 'edge')

    # --- 5. СОЗДАНИЕ SUBMISSION И СОХРАНЕНИЕ АРТЕФАКТОВ ---
    print("\n--- Создание submission.csv ---")
    rows = []
    prev_ema = None
    R_range = [round(0.50 + 0.05 * x, 2) for x in range(9)]
    # Применяем постобработку к каждому предсказанию.
    for pred in padded_preds:
        # Сглаживаем предсказание с помощью EMA
        ema_value = calculate_ema(prev_ema, float(pred), CONFIG['alpha_ema'])
        prev_ema = ema_value
        # Выбираем оптимальные R, s, p
        R, s_n, p_n = select_code_rate(ema_value, CONFIG['f_ec'], R_range)
        rows.append([f"{pred:.16f}", R, s_n, p_n])

    pd.DataFrame(rows).to_csv("../submission.csv", header=False, index=False)
    print("Файл 'submission.csv' успешно создан.")

    # Сохранение всех компонентов решения для предоставления жюри, если понадобится
    print("\n--- Сохранение артефактов модели ---")
    with open(CONFIG['save_paths']['lightgbm'], "wb") as f:
        pickle.dump(lgbm, f)
    with open(CONFIG['save_paths']['catboost'], "wb") as f:
        pickle.dump(cat, f)
    with open(CONFIG['save_paths']['meta_model'], "wb") as f:
        pickle.dump(meta_model, f)
    with open(CONFIG['save_paths']['preprocessor'], "wb") as f:
        pickle.dump(preprocessor, f)
    # Сохраняем конфиг, чтобы точно знать, с какими параметрами были получены модели
    with open(CONFIG['save_paths']['config'], 'w') as f:
        json.dump(CONFIG, f, indent=4)
    print("Все артефакты (модели, препроцессор, конфиг) успешно сохранены.")
