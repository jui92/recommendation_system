import os
from pathlib import Path
import pickle
import numpy as np
import pandas as pd
import streamlit as st

# -----------------------------
# 0) TensorFlow 가드 (Cloud에서 원인 표시)
# -----------------------------
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
except Exception as e:
    st.error(
        "TensorFlow를 불러오지 못했습니다.\n"
        "requirements.txt와 runtime.txt를 확인하세요.\n\n"
        f"[원인] {e}"
    )
    st.stop()

st.set_page_config(page_title="MovieLens AutoInt Recommender", layout="wide")

# -----------------------------
# 1) 경로/필수 파일 체크
# -----------------------------
DATA_DIR = Path("data/ml-1m")
ART_DIR  = Path("artifacts")
MODEL_W  = Path("model/autoInt_model.weights.h5")

required_files = [
    DATA_DIR / "users.dat",
    DATA_DIR / "movies.dat",
    DATA_DIR / "ratings.dat",
    ART_DIR / "field_dims.npy",
    ART_DIR / "label_encoders.pkl",
    MODEL_W,
]

missing = [str(p) for p in required_files if not p.exists()]
if missing:
    st.error(
        "필수 파일이 없습니다. 아래 파일을 레포지토리에 포함하세요.\n\n"
        + "\n".join(missing)
    )
    st.stop()

# -----------------------------
# 2) 경량 로딩 (캐시)
# -----------------------------
@st.cache_data(show_spinner=False)
def load_small_tables():
    users = pd.read_csv(
        DATA_DIR / "users.dat", sep="::", engine="python",
        names=["user_id","gender","age","occupation","zip"]
    )
    movies = pd.read_csv(
        DATA_DIR / "movies.dat", sep="::", engine="python",
        names=["movie_id","title","genres"]
    )
    ratings = pd.read_csv(
        DATA_DIR / "ratings.dat", sep="::", engine="python",
        names=["user_id","movie_id","rating","timestamp"]
    )
    ratings["label"] = (ratings["rating"] >= 4).astype(int)
    ratings["ts"] = pd.to_datetime(ratings["timestamp"], unit="s")
    ratings["rating_year"]  = ratings["ts"].dt.year
    ratings["rating_month"] = ratings["ts"].dt.month
    movies["main_genre"] = movies["genres"].str.split("|").str[0]
    return users, movies, ratings

@st.cache_resource(show_spinner=False)
def load_artifacts_and_model():
    # label encoders / field dims
    field_dims = np.load(ART_DIR / "field_dims.npy")
    with open(ART_DIR / "label_encoders.pkl", "rb") as f:
        enc_obj = pickle.load(f)
    cat_cols        = enc_obj["cat_cols"]
    label_encoders  = enc_obj["label_encoders"]

    # ===== AutoInt 모델 구조 (학습과 동일해야 함) =====
    num_fields  = len(cat_cols)
    embed_dim   = 32
    num_heads   = 4
    attn_layers = 2
    dropout_rate = 0.2
    mlp_units   = [128, 64]

    inp = keras.Input(shape=(num_fields,), dtype="int32")
    embeds = []
    for i, dim in enumerate(field_dims):
        vi = layers.Lambda(lambda x: tf.gather(x, indices=i, axis=1))(inp)
        vi = layers.Reshape((1,))(vi)
        ei = layers.Embedding(input_dim=int(dim), output_dim=embed_dim)(vi)
        embeds.append(ei)
    E = layers.Concatenate(axis=1)(embeds)

    x = E
    for _ in range(attn_layers):
        attn_out = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim, dropout=dropout_rate)(x, x)
        x = layers.Add()([x, attn_out])
        x = layers.LayerNormalization()(x)

    x = layers.GlobalAveragePooling1D()(x)
    for u in mlp_units:
        x = layers.Dense(u, activation="relu")(x)
        x = layers.Dropout(dropout_rate)(x)
    out = layers.Dense(1, activation="sigmoid")(x)

    model = keras.Model(inputs=inp, outputs=out)
    model.compile(optimizer="adam", loss="binary_crossentropy")
    # 더미 호출로 build
    _ = model.predict(np.zeros((1, num_fields), dtype=np.int32), verbose=0)
    # 가중치 로드 (.weights.h5 필수)
    model.load_weights(str(MODEL_W))
    return cat_cols, label_encoders, field_dims, model

users, movies, ratings = load_small_tables()
cat_cols, label_encoders, field_dims, model = load_artifacts_and_model()

# -----------------------------
# 3) 유틸
# -----------------------------
def map_single(col, val):
    m = label_encoders[col]
    return m.get(str(val), 0)

def recommend_for_user(original_user_id: int, topn: int = 10):
    """유저가 보지 않은 모든 영화에 대해 점수 예측 → TopN 반환"""
    # 유저 side features
    urow = users[users["user_id"]==original_user_id]
    if len(urow)==0:
        g, a, o, z = "M", 25, 0, "00000"
    else:
        g, a, o, z = urow.iloc[0][["gender","age","occupation","zip"]]

    # 이미 본 영화 제외
    seen = set(ratings.loc[ratings["user_id"]==original_user_id, "movie_id"].tolist())
    cand = movies[~movies["movie_id"].isin(seen)].copy()
    if cand.empty:
        return pd.DataFrame(columns=["movie_id","title","genres","score"])

    cand["main_genre"] = cand["genres"].str.split("|").str[0]

    # 라벨 인코딩 (없는 값은 0으로)
    mg_idx = cand["main_genre"].astype(str).map(label_encoders["main_genre"]).fillna(0).astype(int).values
    m_idx  = cand["movie_id"].astype(str).map(label_encoders["movie_id"]).fillna(0).astype(int).values

    # 고정 값
    g_idx = map_single("gender", g)
    a_idx = map_single("age", a)
    o_idx = map_single("occupation", o)
    z_idx = map_single("zip", z)

    # 예시로 현재 연/월(또는 ratings의 중앙값)을 사용
    year  = int(ratings["rating_year"].median())
    month = int(ratings["rating_month"].median())

    # 입력 행렬 (cat_cols 순서와 동일해야 함)
    # 기본 cat_cols: ["user_id","movie_id","gender","age","occupation","zip","main_genre"]
    n = len(cand)
    U = np.full((n,), map_single("user_id", original_user_id), dtype=np.int32)
    G = np.full((n,), g_idx, dtype=np.int32)
    A = np.full((n,), a_idx, dtype=np.int32)
    O = np.full((n,), o_idx, dtype=np.int32)
    Z = np.full((n,), z_idx, dtype=np.int32)

    # 모델 학습 시 cat_cols가 위 7개와 동일했다고 가정
    X = np.stack([U, m_idx, G, A, O, Z, mg_idx], axis=1)

    scores = model.predict(X, batch_size=65536, verbose=0).ravel()
    cand = cand.assign(score=scores)
    top = cand.sort_values("score", ascending=False).head(topn)
    return top[["movie_id","title","genres","score"]]

def get_user_profile(uid: int, k: int = 10):
    hist = (
        ratings[ratings["user_id"]==uid]
        .sort_values("ts", ascending=False)
        .head(k)
        .merge(movies[["movie_id","title","genres"]], on="movie_id", how="left")
    )
    return hist[["user_id","movie_id","rating","ts","title","genres"]]

# -----------------------------
# 4) UI
# -----------------------------
st.title("🎬 MovieLens AutoInt 추천 결과")

col_a, col_b, col_c = st.columns([2,2,1])

with col_a:
    st.subheader("사용자 선택")
    uid = st.selectbox(
        "User ID",
        options=sorted(users["user_id"].unique().tolist()),
        index=0
    )
with col_b:
    topn = st.slider("추천 개수", 5, 50, 10, 1)
with col_c:
    st.write("")
    st.write("")

st.divider()

st.markdown("#### 사용자 최근 시청 이력")
st.dataframe(get_user_profile(uid, k=10), use_container_width=True, height=260)

if st.button("🔎 추천 결과 보기", type="primary"):
    with st.spinner("추천 계산 중…"):
        recs = recommend_for_user(int(uid), topn=topn)
    st.markdown("#### 추천 결과")
    st.dataframe(recs.reset_index(drop=True), use_container_width=True, height=400)
else:
    st.info("상단에서 사용자/추천 개수를 설정하고 버튼을 눌러 주세요.")
