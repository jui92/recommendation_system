import os
import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib

from autoint import AutoIntModel, predict_model

@st.cache_resource
def load_data():
    """
    - data/ 아래에 저장된 전처리 csv와 인코더, field_dims, 모델 가중치를 로드
    - 경로 구조
      project_root/
        ├─ data/
        │   ├─ field_dims.npy
        │   ├─ label_encoders.pkl
        │   └─ ml-1m/
        │       ├─ ratings_prepro.csv
        │       ├─ movies_prepro.csv
        │       └─ users_prepro.csv
        └─ model/
            └─ autoInt_model.weights.h5   # ← 주의: 확장자 규칙!
    """
    project_path   = os.path.abspath(os.getcwd())
    data_dir_nm    = 'data'
    movielens_dir  = 'ml-1m'
    model_dir_nm   = 'model'
    data_path      = f"{project_path}/{data_dir_nm}"
    model_path     = f"{project_path}/{model_dir_nm}"

    field_dims     = np.load(f'{data_path}/field_dims.npy')
    label_encoders = joblib.load(f'{data_path}/label_encoders.pkl')

    ratings_df = pd.read_csv(f'{data_path}/{movielens_dir}/ratings_prepro.csv')
    movies_df  = pd.read_csv(f'{data_path}/{movielens_dir}/movies_prepro.csv')
    users_df   = pd.read_csv(f'{data_path}/{movielens_dir}/users_prepro.csv')

    # 모델 빌드 & 가중치 로드
    dropout   = 0.4
    embed_dim = 16
    model = AutoIntModel(
        field_dims, embed_dim,
        att_layer_num=3, att_head_num=2, att_res=True,
        l2_reg_dnn=0, l2_reg_embedding=1e-5,
        dnn_use_bn=False, dnn_dropout=dropout, init_std=0.0001
    )
    # 더미 한 배치로 build
    _ = model([[0 for _ in range(len(field_dims))]])
    model.load_weights(f'{model_path}/autoInt_model.weights.h5')  # ← 파일명 수정

    return users_df, movies_df, ratings_df, model, label_encoders


def get_user_seen_movies(ratings_df: pd.DataFrame) -> pd.DataFrame:
    """사용자가 과거에 본 영화 리스트"""
    return ratings_df.groupby('user_id')['movie_id'].apply(list).reset_index()


def get_user_non_seen_dict(movies_df: pd.DataFrame, users_df: pd.DataFrame, user_seen_movies: pd.DataFrame) -> dict:
    """사용자가 보지 않은 영화 리스트 사전 {user_id: [movie_id, ...]}"""
    unique_movies = movies_df['movie_id'].unique()
    unique_users  = users_df['user_id'].unique()
    user_non_seen = {}
    seen_map = dict(zip(user_seen_movies['user_id'], user_seen_movies['movie_id']))
    for uid in unique_users:
        seen_list = seen_map.get(uid, [])
        user_non_seen[uid] = list(set(unique_movies) - set(seen_list))
    return user_non_seen


def get_user_info(users_df: pd.DataFrame, user_id: int) -> pd.DataFrame:
    return users_df[users_df['user_id'] == user_id]


def get_user_past_interactions(ratings_df: pd.DataFrame, movies_df: pd.DataFrame, user_id: int) -> pd.DataFrame:
    """4점 이상만"""
    return (
        ratings_df[(ratings_df['user_id'] == user_id) & (ratings_df['rating'] >= 4)]
        .merge(movies_df, on='movie_id', how='left')
    )


def _safe_transform(le, series: pd.Series):
    """라벨 인코더에 없는 값이 들어오면 예외가 나므로, 미지 값은 가장 빈도 낮은 클래스로 매핑"""
    try:
        return le.transform(series)
    except ValueError:
        known = set(le.classes_)
        s = series.astype(str).map(lambda x: x if x in known else list(known)[-1])
        return le.transform(s)


def get_recom(user: int, user_non_seen_dict: dict, users_df: pd.DataFrame,
              movies_df: pd.DataFrame, r_year: int, r_month: int,
              model, label_encoders: dict, topk: int = 20) -> pd.DataFrame:
    """
    1) 사용자가 보지 않은 영화들 생성
    2) 모델 입력 포맷으로 조립
    3) 저장된 label encoders로 transform
    4) 예측 → 상위 K 반환
    """
    non_seen = user_non_seen_dict.get(user, [])
    if len(non_seen) == 0:
        return pd.DataFrame(columns=list(movies_df.columns) + ['score'])

    # side features
    user_info = users_df[users_df['user_id'] == user].copy()
    if user_info.empty:
        return pd.DataFrame(columns=list(movies_df.columns) + ['score'])

    r_decade = f"{r_year - (r_year % 10)}s"

    cand = pd.DataFrame({'movie_id': non_seen}).merge(movies_df, on='movie_id', how='left')
    user_block = pd.DataFrame({'user_id': [user]*len(cand)})
    user_block = user_block.merge(user_info, on='user_id', how='left')
    user_block['rating_year']   = r_year
    user_block['rating_month']  = r_month
    user_block['rating_decade'] = r_decade

    merge_data = pd.concat([cand, user_block], axis=1)

    # 모델 입력 컬럼 순서(학습때와 동일해야 함)
    model_cols = [
        'user_id', 'movie_id', 'movie_decade', 'movie_year',
        'rating_year', 'rating_month', 'rating_decade',
        'genre1', 'genre2', 'genre3',
        'gender', 'age', 'occupation', 'zip'
    ]
    merge_data = merge_data[model_cols].fillna('no')

    # transform (※ 반드시 transform)
    for col, le in label_encoders.items():
        if col in merge_data.columns:
            merge_data[col] = _safe_transform(le, merge_data[col].astype(str))

    # 예측
    recom_top = predict_model(model, merge_data)  # [(movie_idx, score), ...] 가정
    # 추천 중 영화 id 인덱스만 추출
    m_idx  = [r[0] for r in recom_top[:topk]]
    score  = [r[1] for r in recom_top[:topk]]
    # 원본 id로 역변환
    origin_m_id = label_encoders['movie_id'].inverse_transform(m_idx)

    out = movies_df[movies_df['movie_id'].isin(origin_m_id)].copy()
    out = out.set_index('movie_id').loc[origin_m_id].reset_index()
    out['score'] = score
    return out


# ===== App =====
users_df, movies_df, ratings_df, model, label_encoders = load_data()
user_seen_movies  = get_user_seen_movies(ratings_df)
user_non_seen_dict = get_user_non_seen_dict(movies_df, users_df, user_seen_movies)

st.title("영화 추천 결과 살펴보기")

st.subheader("사용자 정보를 넣어주세요.")
user_id = st.number_input("사용자 ID 입력", min_value=int(users_df['user_id'].min()),
                          max_value=int(users_df['user_id'].max()),
                          value=int(users_df['user_id'].min()))
r_year  = st.number_input("추천 타겟 연도 입력",  min_value=int(ratings_df['rating_year'].min()),
                          max_value=int(ratings_df['rating_year'].max()),
                          value=int(ratings_df['rating_year'].min()))
r_month = st.number_input("추천 타겟 월 입력",    min_value=int(ratings_df['rating_month'].min()),
                          max_value=int(ratings_df['rating_month'].max()),
                          value=int(ratings_df['rating_month'].min()))
topk    = st.slider("추천 개수", 5, 50, 20, 1)

if st.button("추천 결과 보기"):
    st.write("### 사용자 기본 정보")
    st.dataframe(get_user_info(users_df, user_id))

    st.write("### 사용자가 과거에 봤던 이력(평점 4점 이상)")
    st.dataframe(get_user_past_interactions(ratings_df, movies_df, user_id))

    st.write("### 추천 결과")
    recs = get_recom(user_id, user_non_seen_dict, users_df, movies_df, r_year, r_month, model, label_encoders, topk=topk)
    st.dataframe(recs)
