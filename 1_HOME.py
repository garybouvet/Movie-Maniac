import streamlit as st
import pandas as pd
from st_click_detector import click_detector
import ast
from PIL import Image

# Set the page
st.set_page_config(page_title="DISCOVERY", layout="wide")

# Define session state variables for navigation
if 'random_movies' not in st.session_state:
    st.session_state['random_movies'] = None
if 'selected_movie_details' not in st.session_state:
    st.session_state['selected_movie_details'] = None
if 'current_view' not in st.session_state:
    st.session_state['current_view'] = 'main_page'


logo_path = "./images/small_movie_maniac.png"  

# Load the logo image
logo_image = Image.open(logo_path)

# Create 3 columns with the middle column being wider to center the logo
col1, col2, col3 = st.columns([1, 6, 1])

with col2:
    st.image(logo_image, use_column_width=True)
    
    # Utilisez du HTML pour centrer le texte au-dessus de l'image et réduire la marge
    #st.markdown("<div style='text-align: center; margin-top: -40px;'><h1>Shuffle Discovery</h1></div>", unsafe_allow_html=True)

st.markdown("""
    <style>
    /* Sélecteur pour l'image spécifique */
    #root > div:nth-child(1) > div.withScreencast > div > div > div > section.main.st-emotion-cache-uf99v8.ea3mdgi3 > div.block-container.st-emotion-cache-z5fcl4.ea3mdgi2 > div > div > div > div.st-emotion-cache-ocqkz7.e1f1d6gn5 > div.st-emotion-cache-ml2xh6.e1f1d6gn3 > div > div > div > div:nth-child(1) > div > div > div > img {
        margin-top: -145px; /* Ajustez cette valeur selon vos besoins */
        margin-bottom: -110px;
    }
    
    /* Ajoutez du CSS similaire pour d'autres éléments si nécessaire */
    </style>
""", unsafe_allow_html=True)

#----------------------------------------------------------------------------------------#
## Styling: CSS personnalisé                                                             
#st.markdown(                                                                            
#    """                                                                                 
#    <style>                                                                             
#    .filter-title {                                                                     
#        font-size: 24px;                                                                
#        border: 2px solid #F0B900;                                                      
#        border-radius: 10px;                                                            
#        padding: 5px;                                                                   
#        text-align: center;                                                             
#    }                                                                                   
#    .centered-text {                                                                    
#        display: flex;                                                                  
#        justify-content: center;                                                        
#        align-items: center;                                                            
#        height: 200px;                                                                  
#    }
#    div[data-baseweb="select"] > div {
#        border: 2px solid #F0B900 !important;
#    }
#    button {
#        border: 2px solid #F0B900 !important;
#        background-color: transparent !important;
#        color: #F0B900 !important;
#    }
#    button:hover {
#        background-color: #F0B900 !important;
#        color: #ffffff !important;
#    }
#    button:active {
#        background-color: #F0B900 !important;
#        color: #ffffff !important;
#    }
#    .stButton>button {
#        background-color: transparent;
#        color: white; 
#        border: 1px solid #F0B900; 
#    }
#    .stButton>button:hover {
#        color: #F0B900; 
#        border-color: #F0B900; 
#    }
#    .stButton>button:active {
#        background-color: #F0B900; 
#        color: white; 
#    }
#    </style>
#    """,
#    unsafe_allow_html=True
#)

## Titre du filtre avec marge en dessous
#st.sidebar.markdown(
#    """
#    <style>
#    .filter-title {
#        font-size: 24px;
#        border: 2px solid #F0B900;
#        border-radius: 10px;
#        padding: 5px;
#        text-align: center;
#        margin-bottom: 20px;  /* Ajoutez cette ligne pour la marge en dessous */
#    }
#    </style>
#    <div class="filter-title">
#        FILTRE
#    </div>
#    """,
#    unsafe_allow_html=True
#)
#----------------------------------------------------------------------------------------#
    
def extract_names(column):
    names = []
    for item in column:
        try:
            dict_item = eval(item)
            if isinstance(dict_item, list):
                names.append([d['name'] for d in dict_item if 'name' in d])
            else:
                names.append([])
        except:
            names.append([])
    return names


@st.cache_data
def load_data():
    df = pd.read_csv("./Data/all_movies.csv")
    df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
    df['year'] = df['release_date'].dt.year
    return df

df = load_data()


default_genres_filter = []
default_year_range = (df['release_date'].min().year, df['release_date'].max().year)
default_rating_range = (0.0, 10.0)
default_studios_range = []
default_search_query = ""  # Aucune recherche par défaut


def extract_studios(df):
    studios_list = []
    for item in df['production_companies'].dropna():
        studios = ast.literal_eval(item)  # Convertit la chaîne en liste de dictionnaires
        for studio in studios:
            if 'name' in studio:
                studios_list.append(studio['name'])
    return sorted(set(studios_list))


def extract_top_studios(df, count):
    studio_dict = {}
    for item in df['production_companies'].dropna():
        studios = ast.literal_eval(item)
        for studio in studios:
            if 'name' in studio:
                studio_name = studio['name']
                studio_dict[studio_name] = studio_dict.get(studio_name, 0) + 1

    top_studios = sorted(studio_dict, key=studio_dict.get, reverse=True)[:count]

    if 'Pixar' not in top_studios:
        top_studios.append('Pixar')
    if 'Marvel Studios' not in top_studios:
        top_studios.append('Marvel Studios')

    return top_studios


def setup_sidebar_filters():
    genres = df['genres'].str.get_dummies(sep=',').columns.tolist()
    genres_filter = st.sidebar.multiselect("Genre :", options=genres)
    studios = extract_studios(df)
    studio_counts = df['production_companies'].value_counts()
    top_studios = extract_top_studios(df, 20)
    other_studios = [studio for studio in studios if studio not in top_studios]
    studio_options = ['★ POPULAIRE ★'] + top_studios + ['-' * 54] + other_studios
    selected_studios = st.sidebar.multiselect("Studio :", options=studio_options)
    year_range = st.sidebar.slider("Filtrer par année", min_value=default_year_range[0], max_value=default_year_range[1], value=default_year_range, key='year_range')
    rating_range = st.sidebar.slider("Notes :", min_value=float(df['averageRating'].min()), max_value=float(df['averageRating'].max()), step=0.5, value=(0.0, 10.0), key="rating_slider")

    if '★ POPULAIRE ★' in selected_studios:
        selected_studios = top_studios
    return genres_filter, year_range, rating_range, selected_studios

#Extract actors infos
def extract_actor_info(actor_data):
    unknow_actor_picture = 'https://us.123rf.com/450wm/diddleman/diddleman1205/diddleman120500025/13784515-pas-de-main-image-profil-de-l-utilisateur-dessin%C3%A9e.jpg'
    actors_info = []
    for actor in actor_data:
        actor_name = actor.get('name', 'Inconnu')
        actor_profile_path = actor.get('profile_path', unknow_actor_picture)
        actor_image_url = f'https://image.tmdb.org/t/p/w500{actor_profile_path}' if actor.get('profile_path') else unknow_actor_picture
        actor_role = actor.get('known_for_department', 'Inconnu')
        actors_info.append((actor_name, actor_image_url, actor_role))
    return actors_info


def filter_data(df, genres_filter, year_range, rating_range, selected_studios, search_query):
    if genres_filter:
        df = df[df['genres'].apply(lambda x: all(genre in x for genre in genres_filter))]

    if selected_studios:
        df = df[df['production_companies'].apply(lambda x: any(studio['name'] in selected_studios for studio in ast.literal_eval(x)))]

    if year_range:
        df = df[(df['year'] >= year_range[0]) & (df['year'] <= year_range[1])]

    if rating_range:
        df = df[(df['averageRating'] >= rating_range[0]) & (df['averageRating'] <= rating_range[1])]

    return df

genres_filter, year_range, rating_range, selected_studios = setup_sidebar_filters()


def poster_url(title_id, df):
    try:
        poster_path = df[df['titleId'] == title_id]['poster_path'].values[0]
        return 'https://image.tmdb.org/t/p/w500' + poster_path
    except Exception as e:
        st.error(f"Erreur pour titleId {title_id}: {str(e)}")
        return None
    

def format_votes(number):
    if number < 1000:
        return str(int(number))
    elif number >= 1000 and number < 1000000:
        formatted = str(round(number/1000, 1))
        if formatted.endswith('.0'):
            return formatted[:-2] + ' K'
        return formatted + ' K'
    elif number >= 1000000 and number < 1000000000:
        formatted = str(round(number/1000000, 1))
        if formatted.endswith('.0'):
            return formatted[:-2] + ' M'
        return formatted + ' M'
    else:
        formatted = str(round(number/1000000000, 1))
        if formatted.endswith('.0'):
            return formatted[:-2] + ' B'
        return formatted + ' B'


def format_duration(minutes):
    hours = minutes // 60
    minutes = minutes % 60
    if hours == 1:
        return f"{hours}h{minutes:02d}"
    elif hours > 1:
        return f"{hours}h{minutes:02d}"
    else:
        return f"{minutes} minute{'s' if minutes > 1 else ''}"
    
def escape_html(text):
    """ Remplace les caractères spéciaux par leurs entités HTML. """
    return text.replace("'", "&#39;").replace('"', "&quot;")

# Fonction pour afficher les films
def load_html_template(filename, **kwargs):
    with open(filename, 'r') as file:
        html_template = file.read()
    for key, value in kwargs.items():
        html_template = html_template.replace(f"{{{{ {key} }}}}", str(value))
    return html_template


if st.session_state['current_view'] == 'main_page':
    if st.session_state['random_movies'] is None:
        random_df_filtered = df.copy()  
        num_random_movies = min(8, len(random_df_filtered))
        st.session_state['random_movies'] = random_df_filtered.sample(n=num_random_movies) 
    # Utilisez st.columns pour créer un espace vide sur les côtés du bouton, centré ainsi le bouton
    col1, col2, col3 = st.columns([6, 3, 6])  # Ajustez les proportions si nécessaire
    
    with col2:  # Utilisez la colonne du milieu pour placer le bouton
        if st.button("➡︎ Generate movies", key="random_button"):
            random_df_filtered = df.copy()

            if genres_filter:
                random_df_filtered = random_df_filtered[random_df_filtered['genres'].str.contains('|'.join(genres_filter))]

            if selected_studios:
                random_df_filtered = random_df_filtered[random_df_filtered['production_companies'].apply(lambda x: any(studio['name'] in selected_studios for studio in ast.literal_eval(x)))]

            random_df_filtered['release_date'] = pd.to_datetime(random_df_filtered['release_date'], errors='coerce')

            random_df_filtered['release_year'] = random_df_filtered['release_date'].dt.year

            random_df_filtered = random_df_filtered[
                (random_df_filtered['averageRating'] >= rating_range[0]) & 
                (random_df_filtered['averageRating'] <= rating_range[1]) &
                (random_df_filtered['release_year'] >= year_range[0]) &
                (random_df_filtered['release_year'] <= year_range[1])
            ]

            if len(random_df_filtered) == 0:
                st.warning("Aucun film ne correspond aux filtres sélectionnés.")
            else:
                num_random_movies = min(8, len(random_df_filtered))
                st.session_state['random_movies'] = random_df_filtered.sample(n=num_random_movies)

    if st.session_state['random_movies'] is not None:
        random_movies = st.session_state['random_movies']
        cols = st.columns(4)
        for i, movie_row in enumerate(random_movies.itertuples()):
            movie_data = {
                'movie_id': f"movie-{movie_row.titleId}",
                'poster_url': f"https://image.tmdb.org/t/p/w500{movie_row.poster_path}",
                'average_rating': movie_row.averageRating,
                'votes_formatted': format_votes(movie_row.numVotes),
                'movie_title': escape_html(movie_row.primaryTitle),
                'genres': movie_row.genres.replace(',', ', '),
                'movie_year': movie_row.release_date.year if pd.notnull(movie_row.release_date) else 'Inconnue'
            }
            html_content = load_html_template('./html_script/movie_display.html', **movie_data)

            with cols[i % 4]:
                clicked = click_detector(html_content)
                if clicked != "":
                    st.session_state['selected_movie_details'] = clicked
                    st.session_state['current_view'] = 'details_page'
                    st.rerun()
                    
# ------------------------------------------ DETAILS MOVIE ------------------------------------------#
                    
def show_movie_details(unique_id):
    title_id = unique_id.replace('movie-', '')

    if title_id in df['titleId'].values:
        movie_data = df[df['titleId'] == title_id].iloc[0]
        movie_title = movie_data['primaryTitle']

        trailer_url = movie_data['trailer_url']
        if pd.isna(trailer_url):
            trailer_url = None

        num_votes_formatted = format_votes(int(movie_data['numVotes']))
        formatted_genres = movie_data['genres'].replace(',', ', ')
        num_votes_formatted = format_votes(int(movie_data['numVotes']))
        runtime_minutes = int(movie_data['runtimeMinutes'])
        duration_text = format_duration(runtime_minutes)
        selected_movie_description = movie_data['overview']
        poster_img_url = poster_url(title_id, df)

        # Manage missing values with "/"
        writer = ', '.join(ast.literal_eval(movie_data['writers'])) if pd.notna(movie_data['writers']) else "/"
        writer = "/" if not writer else writer  
        director = movie_data['director'] if pd.notna(movie_data['director']) else "/"
        studios = ', '.join([studio['name'] for studio in ast.literal_eval(movie_data['production_companies'])]) if pd.notna(movie_data['production_companies']) else "/"
        sortie = movie_data['release_date'].year if pd.notna(movie_data['release_date']) else "/"
        duree = duration_text if pd.notna(movie_data['runtimeMinutes']) else "/"

        youtube_embed_html = ""
        if trailer_url:
            youtube_embed_url = f"https://www.youtube.com/embed/{trailer_url.split('watch?v=')[-1]}"
            youtube_embed_html = f"""<iframe width="430" height="215" src="{youtube_embed_url}" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>"""
        
        # Display detailed information about the movie
        st.markdown("### Your choice :")
        st.markdown(f"""
            <div style="border: 2px solid #F0B900; border-radius: 10px; padding: 20px;">
                <div style="display: flex;">
                    <div style="flex-shrink: 0;">
                    <img src="{poster_img_url}" style="width: 400px; height: auto; border-radius: 10px; margin-right: 20px;">
                    </div>
                    <div style="flex-grow: 1; display: flex; flex-direction: column; justify-content: space-between;">
                        <div>
                            <h3 style="margin: 0; font-size: 24px;">{movie_title} ({sortie})</h3>
                            <p style="margin: 0; font-size: 18px;"><span style='color: gold; font-size: 22px;'>★</span> {movie_data['averageRating']} | <span style='color: #E0E1DB;'>Votes: {num_votes_formatted}</span></p>
                            <p style="margin: 0; font-size: 18px;"><b>Genres :</b> {formatted_genres}</p>
                            <p style="margin: 0; font-size: 18px;"><b>Runtime :</b> {duree}</p>
                            <p style="margin: 0; font-size: 18px;"><b>Director :</b> {director}</p>
                            <p style="margin: 0; font-size: 18px;"><b>Writer :</b> {writer}</p>
                            <p style="margin: 0; font-size: 18px;"><b>Studios :</b> {studios}</p>
                        </div>
                        <div>
                            <p style="margin-top: 10px; margin-bottom: 5px; font-size: 18px; font-weight: bold;">Trailer:</p>
                            <div style="border: 2px solid #E0E1DB; border-radius: 10px; overflow: hidden; width: 430px; height: 215px; margin: auto;">
                                {youtube_embed_html if trailer_url else "<p style='margin: 0; font-size: 18px;'>Not available</p>"}
                            </div>
                        </div>
                    </div>
                </div>
                <div style="margin-top: 10px;">
                    <p style="margin: 0; font-size: 18px; font-weight: bold;">Description :</p>
                    <p style="margin: 0; font-size: 16px; white-space: pre-line;">{selected_movie_description}</p>
                </div>
            </div>
        """, unsafe_allow_html=True)

        st.markdown("""
            <div style="margin-top: 20px;">
                <h3 style="margin-bottom: 10px;">Actors :</h3>
            </div>
        """, unsafe_allow_html=True)

        actors_info = extract_actor_info(ast.literal_eval(movie_data['cast']))

        actor_groups = [actors_info[i:i+5] for i in range(0, len(actors_info), 5)]
        for group in actor_groups:
            columns = st.columns(len(group))  
            for index, actor in enumerate(group):
                with columns[index]:  
                    st.markdown(f"""
                        <div style="display: flex; flex-direction: column; align-items: center; margin-bottom: 5px;">
                            <div style="width: 100px; height: 150px; overflow: hidden; border-radius: 10px; text-align: center;">
                                <img src="{actor[1]}" style="width: 100%; height: 100%; object-fit: cover; border-radius: 10px;">
                            </div>
                            <p style="margin: 5px 0 15px;">{actor[0]}</p>
                        </div>
                    """, unsafe_allow_html=True)

    else:
        st.error("Détails du film non disponibles.")

#Move the detail container on the bug Js automatic scroll up
st.markdown("""
    <style>
    /* Selector for the title */
    #root > div:nth-child(1) > div.withScreencast > div > div > div > section.main.st-emotion-cache-uf99v8.ea3mdgi3 > div.block-container.st-emotion-cache-z5fcl4.ea3mdgi2 > div > div > div > div.st-emotion-cache-0.e1f1d6gn0 > div > div > div:nth-child(2) {
        margin-top: -50px; /* Adjust this value */
    }
    
    /* Selector for the detailed movie */
    #root > div:nth-child(1) > div.withScreencast > div > div > div > section.main.st-emotion-cache-uf99v8.ea3mdgi3 > div.block-container.st-emotion-cache-z5fcl4.ea3mdgi2 > div > div > div > div.st-emotion-cache-0.e1f1d6gn0 > div > div > div:nth-child(3) {
        margin-top: -190px; /* Adjust this value */
    }

    /* Add similar CSS for other elements as needed */
    </style>
""", unsafe_allow_html=True)

#Make the background detail container black to hide the bug Js automatic scroll up
st.markdown("""
    <style>
    /* Specific selector for the movie details container */
    #root > div:nth-child(1) > div.withScreencast > div > div > div > section.main.st-emotion-cache-uf99v8.ea3mdgi3 > div.block-container.st-emotion-cache-z5fcl4.ea3mdgi2 > div > div > div > div.st-emotion-cache-0.e1f1d6gn0 > div > div > div:nth-child(3) > div > div > div {
        background-color: #0e1117;
    }
    </style>
""", unsafe_allow_html=True)


# ----------------------- MOVIE DETAILS PAGE LOGIC -----------------------
if st.session_state['current_view'] == 'details_page':
    # To go back to the discovery page
    if st.button('⬅ Back to discovery'):
        st.session_state['current_view'] = 'main_page'
        st.rerun()

    # Use a container for the detailed movie information
    with st.container():
        # JavaScript to scroll to the top when on the detailed movie page
        js = '''
        <script>
            var body = window.parent.document.querySelector(".main");
            console.log(body);
            body.scrollTop = 0;
        </script>
        '''
        st.components.v1.html(js)

        # Function to show movie details
        show_movie_details(st.session_state['selected_movie_details'])

#----------------------------------- END -------------------------------------#
