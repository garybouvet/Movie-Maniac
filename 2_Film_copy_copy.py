import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Configuration de la page
st.set_page_config(page_title="Titre de votre application", layout="wide")

def load_custom_styles():
    st.markdown(
        """
        <style>
        .centered-text { display: flex; justify-content: center; align-items: center; height: 200px; }
        div[data-baseweb="select"] > div { border: 2px solid #F0B900 !important; }
        button { border: 2px solid #F0B900 !important; background-color: transparent !important; color: #F0B900 !important; }
        button:hover, button:active { background-color: #F0B900 !important; color: #ffffff !important; }
        .stButton>button { background-color: transparent; color: white; border: 1px solid #F0B900; }
        .stButton>button:hover, .stButton>button:active { color: #F0B900; border-color: #F0B900; background-color: #F0B900; color: white; }
        </style>
        """,
        unsafe_allow_html=True
    )

load_custom_styles()

def display_header():
    st.markdown(
        """
        <div class="centered-text">
            <h1>FILMS</h1>
        </div>
        """,
        unsafe_allow_html=True
    )

display_header()

@st.cache_data(show_spinner=False)
def load_and_preprocess_data():
    df = pd.read_csv("./Data_base/merged_data.csv")
    df['production_companies_name_y'] = df['production_companies_name_y'].str.split(',')
    df = df.explode('production_companies_name_y')
    df['production_companies_name_y'] = df['production_companies_name_y'].str.strip().str.replace('"', '').str.replace("'", '')
    df.drop_duplicates(subset="tconst", keep="first", inplace=True)
    return df

df = load_and_preprocess_data()

@st.cache_data(show_spinner=False)
def preprocess_actor_data(df):
    df['actor_set'] = df['Acteurs_Film'].apply(lambda x: set(x.split(',')) if isinstance(x, str) else set())
    df['actor_set_str'] = df['actor_set'].apply(lambda x: '|'.join(x))
    return df

df = preprocess_actor_data(df)

@st.cache_data(show_spinner=False)
def calculate_similarity(df):
    cv = CountVectorizer()
    count_matrix = cv.fit_transform(df['combined_features'])
    return cosine_similarity(count_matrix)

cosine_sim = calculate_similarity(df)

def setup_sidebar_filters():
    genres = df['genres'].str.get_dummies(sep=',').columns.tolist()
    genres_filter = st.sidebar.multiselect("Genre :", options=genres)

    actors = df['Acteurs_Film'].str.get_dummies(sep=',').columns.tolist()
    selected_actors = st.sidebar.multiselect("Acteurs :", options=actors, key="unique_key_for_actor_selection")

    year_range = st.sidebar.slider("Filtrer par année", min_value=int(df['startYear'].min()), max_value=int(df['startYear'].max()), value=(int(df['startYear'].min()), int(df['startYear'].max())))
    rating_range = st.sidebar.slider("Notes :", min_value=float(df['averageRating'].min()), max_value=float(df['averageRating'].max()), step=0.5, value=(0.0, 10.0), key="rating_slider")
    
    studio_counts = df['production_companies_name_y'].value_counts()
    top_studios = studio_counts.nlargest(10).index.tolist()
    other_studios = studio_counts.index[~studio_counts.index.isin(top_studios)].sort_values().tolist()
    studio_options = ['★ POPULAIRE ★'] + top_studios + ['-' * 54] + other_studios
    selected_studios = st.sidebar.multiselect("Studio :", options=studio_options, key="unique_key_studios")
    
    search_query = st.text_input("", placeholder="Rechercher :", key="search_input")
    
    return genres_filter, selected_actors, year_range, rating_range, selected_studios, search_query

genres_filter, selected_actors, year_range, rating_range, selected_studios, search_query = setup_sidebar_filters()

def apply_filters(df, genres_filter, year_range, rating_range, selected_studios, selected_actors, search_query):
    if genres_filter:
        df = df[df['genres'].str.contains('|'.join(genres_filter))]
    if year_range:
        df = df[(df['startYear'] >= year_range[0]) & (df['startYear'] <= year_range[1])]
    if rating_range:
        df = df[(df['averageRating'] >= rating_range[0]) & (df['averageRating'] <= rating_range[1])]
    if selected_studios:
        df = df[df['production_companies_name_y'].isin(selected_studios)]
    if selected_actors:
        df = df[df['Acteurs_Film'].apply(lambda x: any(actor in x.split(',') for actor in selected_actors))]
    if search_query:
        df = df[df['title'].str.contains(search_query, case=False, regex=False)]
    return df

df_filtered = apply_filters(df, genres_filter, year_range, rating_range, selected_studios, selected_actors, search_query)

def display_movies(df_filtered):
    cols = st.columns(4)
    for i, movie_row in enumerate(df_filtered.itertuples()):
        with cols[i % 4]:
            st.image(f"https://image.tmdb.org/t/p/w500{movie_row.poster_path_y}", width=150)
            st.write(movie_row.title)
        if (i + 1) % 4 == 0:
            cols = st.columns(4)

display_movies(df_filtered)

def show_movie_details(tconst):
    movie = df[df['tconst'] == tconst].iloc[0]
    col1, col2 = st.columns([1, 3])
    with col1:
        st.image(f"https://image.tmdb.org/t/p/w900{movie['poster_path_y']}", use_column_width=True)
    with col2:
        st.write(f"**Titre :** {movie['title']}")
        st.write(f"**Année :** {int(movie['startYear'])}")
        st.write(f"**Durée :** {movie['runtimeMinutes']} minutes")
        st.write(f"**Genres :** {movie['genres']}")
        st.write(f"**Note :** {movie['averageRating']}")

def title_from_index(index):
    return df[df.index == index]["title"].values[0]

def index_from_title(title):
    return df[df.title == title]["index"].values[0]

def find_similar_movies(movie_title, num_movies=5):
    movie_index = index_from_title(movie_title)
    similar_movies = list(enumerate(cosine_sim[movie_index]))
    sorted_similar_movies = sorted(similar_movies, key=lambda x: x[1], reverse=True)[1:num_movies + 1]
    return [title_from_index(element[0]) for element in sorted_similar_movies]

st.markdown(
    """
    <style>
    .stTextInput input {
        border: 1px solid #F0B900 !important;
        border-radius: 10px !important;
        padding: 5px !important;
        outline: none !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Gestion de l'état de la page et de la pagination
num_movies_per_page = 52

if 'page_state' not in st.session_state:
    st.session_state['page_state'] = "gallery"
    st.session_state['movies_shown'] = num_movies_per_page

# Afficher plus de films
if len(df_filtered) > st.session_state['movies_shown']:
    if st.button("Afficher plus", key='unique_key_afficher_plus'):
        st.session_state['movies_shown'] += num_movies_per_page
        st.experimental_rerun()
