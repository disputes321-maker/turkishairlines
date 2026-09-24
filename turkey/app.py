import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "best_model_pipeline.pkl"
DATA_PATH = BASE_DIR / "real_estate_data.csv"

st.set_page_config(
    page_title="Прогноз стоимости недвижимости",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------
# Загрузка
# -----------------------------
@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_dataset():
    if not DATA_PATH.exists():
        return None
    return pd.read_csv(DATA_PATH, low_memory=False)

model = load_model()
df = load_dataset()

# -----------------------------
# Параметры именно этой модели
# -----------------------------
MODEL_FEATURES = [
    "type", "sub_type", "listing_type", "building_age", "total_floor_count",
    "floor_no", "room_count", "size", "heating_type", "city", "district",
    "neighborhood", "rooms_numeric", "is_new_building", "floor_ratio",
    "room_grouped", "city_grouped"
]
NUMERIC_FEATURES = [
    "listing_type", "building_age", "total_floor_count", "floor_no", "size",
    "rooms_numeric", "is_new_building", "floor_ratio"
]
CATEGORICAL_FEATURES = [
    "type", "sub_type", "room_count", "heating_type", "city", "district",
    "neighborhood", "room_grouped", "city_grouped"
]

# В датасете значения русские а сохранённая модель обучалась
# на исходных турецких категориях. Поэтому перед prediction выполняем
# обратное преобразование категорий.
SUBTYPE_MAP = {
    "Квартира": "Daire",
    "Вилла": "Villa",
    "Отдельный дом": "Müstakil Ev",
    "Резиденция": "Rezidans",
    "Дача": "Yazlık",
    "Целое здание": "Komple Bina",
    "Сборный дом": "Prefabrik Ev",
    "Фермерский дом": "Çiftlik Evi",
    "Особняк / Усадьба / Дом у воды": "Köşk / Konak / Yalı",
    "Квартира у воды": "Yalı Dairesi",
    "Кооператив": "Kooperatif",
    "Лофт": "Loft",
}

HEATING_MAP = {
    "Фанкойл": "Fancoil",
    "Нет": "Yok",
    "Электрическое": "Kombi (Elektrikli)",
    "Солярное отопление": "Güneş Enerjisi",
    "Центральное (газ)": "Kalorifer (Doğalgaz)",
    "Калорифер (уголь)": "Kalorifer (Kömür)",
}

CITY_MAP = {
    "Адана": "Adana", "Анкара": "Ankara", "Анталья": "Antalya",
    "Айдын": "Aydın", "Балыкесир": "Balıkesir", "Бурса": "Bursa",
    "Эскишехир": "Eskişehir", "Кайсери": "Kayseri", "Коджаэли": "Kocaeli",
    "Мерсин": "Mersin", "Мугла": "Muğla", "Сакарья": "Sakarya",
    "Самсун": "Samsun", "Стамбул": "İstanbul", "Измир": "İzmir",
}

# В сохранённом pipeline listing_type находится среди числовых признаков.
# Порядок соответствует кодированию категорий: Аренда=0, Аренда на день=1, Продажа=2.
LISTING_MAP = {"Продажа": 0, "Аренда": 2}

AGE_MAP = {
    "0": 0, "1": 1, "2": 2, "3": 3, "4": 4, "5": 5,
    "6-10 лет": 8, "11-15 лет": 13, "16-20 лет": 18,
    "21-25 лет": 23, "26-30 лет": 28, "31-35 лет": 33,
    "36-40 лет": 38, "40 и более": 40,
}

ROOM_GROUPS = {
    "+", "1+0", "1+1", "2+1", "2+2", "3+1", "3+2",
    "4+1", "4+2", "5+1", "5+2", "6+1", "6+2", "7+1", "7+2"
}

CITY_GROUPS = {
    "Adana", "Ankara", "Antalya", "Aydın", "Balıkesir", "Bursa",
    "Eskişehir", "Kayseri", "Kocaeli", "Mersin", "Muğla", "Sakarya",
    "Samsun", "İstanbul", "İzmir"
}

FLOOR_SPECIAL = {
    "Высокий первый этаж": 1,
    "Первый этаж": 1,
    "Цокольный этаж -1": -1,
    "Цокольный этаж -2": -2,
    "Последний этаж": 99,
    "Мансарда": 99,
    "Садовый этаж": 0,
    "Отдельный": 0,
}


def parse_number(value, default=np.nan):
    if pd.isna(value):
        return default
    m = re.search(r"-?\d+(?:[.,]\d+)?", str(value))
    if not m:
        return default
    return float(m.group(0).replace(",", "."))


def parse_range_midpoint(value):
    text = str(value)
    nums = re.findall(r"\d+", text)
    if len(nums) >= 2:
        return (float(nums[0]) + float(nums[1])) / 2
    if len(nums) == 1:
        return float(nums[0])
    return np.nan


def parse_rooms(value):
    nums = re.findall(r"\d+", str(value))
    return sum(int(x) for x in nums) if nums else np.nan


def convert_age(value):
    if value in AGE_MAP:
        return AGE_MAP[value]
    return parse_range_midpoint(value)


def convert_floor_count(value):
    if str(value) == "20 и более":
        return 20.0
    return parse_range_midpoint(value)


def convert_floor(value):
    if value in FLOOR_SPECIAL:
        return float(FLOOR_SPECIAL[value])
    return parse_number(value)


def make_input(
    city, district, neighborhood, listing_type, sub_type, room_count,
    heating_type, size, building_age, total_floor_count, floor_no
):
    age = convert_age(building_age)
    total_floors = convert_floor_count(total_floor_count)
    floor = convert_floor(floor_no)
    rooms_numeric = parse_rooms(room_count)

    city_model = CITY_MAP.get(city, city)
    sub_type_model = SUBTYPE_MAP.get(sub_type, sub_type)
    heating_model = HEATING_MAP.get(heating_type, heating_type)

    row = {
        "type": "Konut",  # в исходном обучении модели
        "sub_type": sub_type_model,
        "listing_type": LISTING_MAP[listing_type],
        "building_age": age,
        "total_floor_count": total_floors,
        "floor_no": floor,
        "room_count": room_count,
        "size": float(size),
        "heating_type": heating_model,
        "city": city_model,
        "district": district,
        "neighborhood": neighborhood,
        "rooms_numeric": rooms_numeric,
        "is_new_building": 1.0 if pd.notna(age) and age <= 5 else 0.0,
        "floor_ratio": floor / total_floors if pd.notna(floor) and pd.notna(total_floors) and total_floors > 0 else np.nan,
        "room_grouped": room_count if room_count in ROOM_GROUPS else "Другое",
        "city_grouped": city_model if city_model in CITY_GROUPS else "Другое",
    }
    return pd.DataFrame([row], columns=MODEL_FEATURES)


def predict_price(input_df):
    """Модель возвращает log(price), поэтому возвращаем цену через expm1."""
    pred = float(model.predict(input_df)[0])
    # Для этой модели целевая переменная обучалась в log1p(price).
    return max(float(np.expm1(pred)), 0.0)


# -----------------------------
# Данные для интерфейса
# -----------------------------
if df is not None:
    cities = sorted(df["city"].dropna().astype(str).unique().tolist())
    districts = sorted(df["district"].dropna().astype(str).unique().tolist())
    neighborhoods = sorted(df["neighborhood"].dropna().astype(str).unique().tolist())
    subtypes = sorted(df["sub_type"].dropna().astype(str).unique().tolist())
    rooms = sorted(df["room_count"].dropna().astype(str).unique().tolist())
    heating = sorted(df["heating_type"].dropna().astype(str).unique().tolist())
    listing_types = sorted(df["listing_type"].dropna().astype(str).unique().tolist())
    ages = sorted(df["building_age"].dropna().astype(str).unique().tolist(), key=lambda x: convert_age(x) if pd.notna(convert_age(x)) else 999)
    floors = sorted(df["total_floor_count"].dropna().astype(str).unique().tolist(), key=lambda x: convert_floor_count(x) if pd.notna(convert_floor_count(x)) else 999)
    floor_nos = sorted(df["floor_no"].dropna().astype(str).unique().tolist(), key=lambda x: convert_floor(x) if pd.notna(convert_floor(x)) else 999)
else:
    cities = ["Стамбул"]
    districts = ["Kartal"]
    neighborhoods = ["Kordonboyu"]
    subtypes = list(SUBTYPE_MAP)
    rooms = ["1+1", "2+1", "3+1", "4+1"]
    heating = list(HEATING_MAP)
    listing_types = list(LISTING_MAP)
    ages = list(AGE_MAP)
    floors = ["1", "2", "3", "4", "5", "20 и более"]
    floor_nos = ["1", "2", "3", "4", "5", "Высокий первый этаж", "Последний этаж"]

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.title("🏠 Навигация")
page = st.sidebar.radio("Раздел", ["📊 Прогноз", "📈 Дашборд", "ℹ️ Справка"])
st.sidebar.markdown("---")

if model is not None:
    st.sidebar.success(" Random Forest загружен")
else:
    st.sidebar.error(" best_model_pipeline.pkl не найден")

if df is not None:
    st.sidebar.success(f" Датасет: {len(df):,} строк")
else:
    st.sidebar.warning("CSV не найден рядом с app.py")

# -----------------------------
# Прогноз
# -----------------------------
if page == "📊 Прогноз":
    st.title("🏠 Прогноз стоимости недвижимости")
    st.caption("Датасет real_estate_data.csv • модель best_model_pipeline.pkl")

    if model is None:
        st.error("Положите best_model_pipeline.pkl в ту же папку, что и app.py")
        st.stop()

    with st.form("prediction_form"):
        st.subheader("Локация")
        c1, c2, c3 = st.columns(3)
        city = c1.selectbox("Город", cities, index=cities.index("Стамбул") if "Стамбул" in cities else 0)
        district = c2.selectbox("Район", districts, index=districts.index("Kartal") if "Kartal" in districts else 0)
        neighborhood = c3.selectbox("Микрорайон", neighborhoods, index=neighborhoods.index("Kordonboyu") if "Kordonboyu" in neighborhoods else 0)

        st.subheader(" Тип объекта")
        c4, c5, c6 = st.columns(3)
        listing_type = c4.selectbox("Тип объявления", listing_types, index=listing_types.index("Продажа") if "Продажа" in listing_types else 0)
        sub_type = c5.selectbox("Тип недвижимости", subtypes, index=subtypes.index("Квартира") if "Квартира" in subtypes else 0)
        heating_type = c6.selectbox("Отопление", heating, index=heating.index("Kombi (Doğalgaz)") if "Kombi (Doğalgaz)" in heating else 0)

        st.subheader(" Параметры")
        c7, c8 = st.columns(2)
        size = c7.number_input("Площадь, м²", min_value=1.0, max_value=5000.0, value=100.0, step=1.0)
        room_count = c8.selectbox("Количество комнат", rooms, index=rooms.index("2+1") if "2+1" in rooms else 0)

        st.subheader(" Здание")
        c9, c10, c11 = st.columns(3)
        building_age = c9.selectbox("Возраст здания", ages)
        total_floor_count = c10.selectbox("Этажность здания", floors)
        floor_no = c11.selectbox("Этаж объекта", floor_nos, index=floor_nos.index("3") if "3" in floor_nos else 0)

        submitted = st.form_submit_button(" Рассчитать стоимость", use_container_width=True, type="primary")

    if submitted:
        floor_value = convert_floor(floor_no)
        total_value = convert_floor_count(total_floor_count)
        if pd.notna(floor_value) and pd.notna(total_value) and total_value > 0 and floor_value > total_value:
            st.warning("Этаж объекта не может быть выше этажности здания.")
        else:
            try:
                x = make_input(
                    city, district, neighborhood, listing_type, sub_type, room_count,
                    heating_type, size, building_age, total_floor_count, floor_no
                )
                price = predict_price(x)

                st.markdown("---")
                a, b, c = st.columns([1, 2, 1])
                with b:
                    st.success("###  Результат")
                    st.metric("Оценочная стоимость", f"{price:,.0f} TRY")
                    st.caption("Прогноз рассчитан сохранённой моделью Random Forest.")

                with st.expander(" Данные, переданные модели"):
                    st.dataframe(x.T.rename(columns={0: "Значение"}), use_container_width=True)
            except Exception as e:
                st.error("Не удалось выполнить прогноз.")
                st.code(str(e))
                st.info("Если появляется ошибка _fill_dtype, установите ту же версию scikit-learn, на которой сохранялась модель (1.7.2).")

# -----------------------------
# Dashboard
# -----------------------------
elif page == "📈 Дашборд":
    st.title("📈 Дашборд датасета")
    if df is None:
        st.error("Файл real_estate_data.csv не найден.")
        st.stop()

    d = df.dropna(subset=["price", "size"]).copy()
    d = d[(d["price"] > 0) & (d["size"] > 0)]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Объектов", f"{len(d):,}")
    c2.metric("Признаков", len(d.columns))
    c3.metric("Городов", d["city"].nunique())
    c4.metric("Медианная цена", f"{d['price'].median():,.0f} TRY")

    st.markdown("---")
    chart = st.selectbox("График", ["Площадь × цена", "Цены по городам", "Средняя цена по городам"])

    if chart == "Площадь × цена":
        sample = d.sample(min(4000, len(d)), random_state=42)
        fig = px.scatter(sample, x="size", y="price", color="listing_type", log_x=True, log_y=True,
                         title="Зависимость цены от площади")
        st.plotly_chart(fig, use_container_width=True)
    elif chart == "Цены по городам":
        top = d["city"].value_counts().head(10).index
        fig = px.box(d[d["city"].isin(top)], x="city", y="price", log_y=True,
                     title="Распределение цен в 10 самых представленных городах")
        st.plotly_chart(fig, use_container_width=True)
    else:
        top = d.groupby("city", as_index=False)["price"].mean().nlargest(15, "price")
        fig = px.bar(top, x="city", y="price", title="Средняя цена по городам")
        st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# Справка
# -----------------------------
else:
    st.title("ℹ️ Справка")
    st.markdown("""
### Датасет
`real_estate_data.csv` содержит 403 487 объявлений недвижимости.

### Модель
Используется сохранённый `best_model_pipeline.pkl` — Pipeline с `RandomForestRegressor`.

Модель принимает 17 признаков:
- категориальные: `type`, `sub_type`, `room_count`, `heating_type`, `city`, `district`, `neighborhood`, `room_grouped`, `city_grouped`;
- числовые: `listing_type`, `building_age`, `total_floor_count`, `floor_no`, `size`, `rooms_numeric`, `is_new_building`, `floor_ratio`.

### Важный момент
CSV и модель используют разные представления части категориальных значений: CSV содержит русские названия, а сохранённый pipeline обучался на исходных турецких категориях. Приложение выполняет преобразование перед передачей данных в модель.

`listing_type` также переводится в числовой код, потому что в сохранённом pipeline он находится среди числовых признаков.

### Файлы проекта
В одной папке должны находиться:
- `app.py`
- `best_model_pipeline.pkl`
- `real_estate_data.csv`
""")
