import streamlit as st
import pandas as pd
from st_click_detector import click_detector
import ast
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

# Page configuration
st.set_page_config(page_title="Movie", layout="wide")

# Session state definition for pages navigation
if 'current_view' not in st.session_state:
    st.session_state['current_view'] = 'main_page'
if 'selected_movie_details' not in st.session_state:
    st.session_state['selected_movie_details'] = None
if 'search_results' not in st.session_state:
    st.session_state['search_results'] = False
if 'search_query' not in st.session_state:
    st.session_state['search_query'] = ""

# Création of 3 columns to force center the title
col1, col2, col3 = st.columns([1, 6, 1])

with col2:
    st.markdown("<div style='text-align: center; margin-top: -40px;'><h1>MOVIE</h1></div>", unsafe_allow_html=True)

# Loading the movie dataframe
def load_data():
    df = pd.read_csv("./Data/europe_and_us_after_90s_movies.csv")
    df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
    df['year'] = df['release_date'].dt.year
    return df

df = load_data()

#---------------------------------------#
#      Calcul for the recommandation    #
#---------------------------------------#

# 1. Set the model
tfidf = TfidfVectorizer(stop_words='english', ngram_range=(1, 2), max_features=100)
tfidf_matrix = tfidf.fit_transform(df['combined_features'])

# 2. Cosine similarity calcul
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# 3. Training the KNN model within Cosine Similarity
knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=5)
knn.fit(tfidf_matrix)

# 4. Features normalization
scaler = MinMaxScaler()
df['normalized_averageRating'] = scaler.fit_transform(df[['averageRating']])
df['normalized_popularity'] = scaler.fit_transform(df[['popularity']])


# Function to get recommandation within the trained models
def get_recommendations_knn(title, num_recommendations=200):
    idx = df.index[df['originalTitle'] == title].tolist()[0]
    target_year = df.loc[idx, 'year']
    
    # Récupération des voisins les plus proches et des scores de similarité cosinus
    distances, indices = knn.kneighbors(tfidf_matrix[idx], n_neighbors=num_recommendations + 1)
    movie_indices = indices.flatten()[1:]
    similarity_scores = distances.flatten()[1:]  # Scores de similarité cosinus
    
    recommended_movies = df.iloc[movie_indices].copy()
    recommended_movies['similarity_score'] = 1 - similarity_scores  # Convertir les distances en scores de similarité
    
    # Calculer la différence d'année
    recommended_movies['year_difference'] = abs(recommended_movies['year'] - target_year)
    
    # Calculer le score composite
    weight_similarity = 0.5
    weight_rating = 0.5
    weight_popularity = 0.5
    weight_year = 0.8         # Poids pour la proximité de l'année
    
    # Normaliser la différence d'année
    max_year_difference = recommended_movies['year_difference'].max()
    recommended_movies['year_score'] = 1 - (recommended_movies['year_difference'] / max_year_difference)
    
    # Calculer le score composite en intégrant le score de similarité cosinus
    recommended_movies['composite_score'] = (weight_similarity * recommended_movies['similarity_score'] +
                                            weight_rating * recommended_movies['averageRating'] +
                                            weight_popularity * recommended_movies['popularity'] +
                                            weight_year * recommended_movies['year_score'])
    
    return recommended_movies.sort_values('composite_score', ascending=False).reset_index(drop=True)



# Function to escape special characters in movies title
def escape_html(text):
    """ Remplace les caractères spéciaux par leurs entités HTML. """
    return text.replace("'", "&#39;").replace('"', "&quot;")

# Set filters default values
default_genres_filter = []
default_year_range = (df['release_date'].min().year, df['release_date'].max().year)
default_rating_range = (0.0, 10.0)
default_studios_range = []
default_search_query = ""  # Aucune recherche par défaut

# Extracting studios
def extract_studios(df):
    studios_list = []
    for item in df['production_companies'].dropna():
        studios = ast.literal_eval(item)  # Convertit la chaîne en liste de dictionnaires
        for studio in studios:
            if 'name' in studio:
                studios_list.append(studio['name'])
    return sorted(set(studios_list))

# Extracting Top studios
def extract_top_studios(df, count):
    studio_dict = {}
    for item in df['production_companies'].dropna():
        studios = ast.literal_eval(item)
        for studio in studios:
            if 'name' in studio:
                studio_name = studio['name']
                studio_dict[studio_name] = studio_dict.get(studio_name, 0) + 1

    # Sort and get most popular 'counted studios'
    top_studios = sorted(studio_dict, key=studio_dict.get, reverse=True)[:count]

    # Check if Pixar & Marvel studios are in the list
    if 'Pixar' not in top_studios:
        top_studios.append('Pixar')
    if 'Marvel Studios' not in top_studios:
        top_studios.append('Marvel Studios')

    return top_studios

# Set the sidebar filter
def setup_sidebar_filters():
    genres = df['genres'].str.get_dummies(sep=',').columns.tolist()
    genres_filter = st.sidebar.multiselect("Genre :", options=genres)
    studios = extract_studios(df)
    studio_counts = df['production_companies'].value_counts()
    top_studios = extract_top_studios(df, 20)
    other_studios = [studio for studio in studios if studio not in top_studios]
    studio_options = ['★ POPULAIRE ★'] + top_studios + ['-' * 54] + other_studios
    selected_studios = st.sidebar.multiselect("Studio :", options=studio_options)
    year_range = st.sidebar.slider(label="Filtrer par année",min_value=default_year_range[0],max_value=default_year_range[1],value=default_year_range,key='year_range',label_visibility="collapsed")
    rating_range = st.sidebar.slider(label="Notes :",min_value=float(df['averageRating'].min()),max_value=float(df['averageRating'].max()),step=0.5,value=(0.0, 10.0),key="rating_slider",label_visibility="collapsed")
    if '★ POPULAIRE ★' in selected_studios:
        selected_studios = top_studios
    #search_query = st.text_input("", placeholder="Rechercher un film ou un acteur :", key="search_input")
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

# To create the sidebar filter function
def filter_data(df, genres_filter, year_range, rating_range, selected_studios, search_query):
    # Genres filter
    if genres_filter:
        df = df[df['genres'].apply(lambda x: all(genre in x for genre in genres_filter))]

    # Studios filter
    if selected_studios:
        df = df[df['production_companies'].apply(lambda x: any(studio['name'] in selected_studios for studio in ast.literal_eval(x)))]

    # Years filter
    if year_range:
        df = df[(df['year'] >= year_range[0]) & (df['year'] <= year_range[1])]

    # Rating filter
    if rating_range:
        df = df[(df['averageRating'] >= rating_range[0]) & (df['averageRating'] <= rating_range[1])]

    # Search query within filters
    if search_query:
        keywords = search_query.lower().split()
        title_search = df['originalTitle'].str.lower().apply(lambda title: all(keyword in title for keyword in keywords))
        actor_search = df['cast'].apply(lambda x: any(search_query in actor['name'].lower() for actor in ast.literal_eval(x) if 'name' in actor))
        #actor_search = df['cast'].apply(lambda x: any(is_close_match(search_query, actor['name']) for actor in ast.literal_eval(x) if 'name' in actor)) #For fuzzyWuzzy distance of Levenshtein 80%
        df = df[title_search | actor_search]

    return df

# Calling the sidebar filter function
genres_filter, year_range, rating_range, selected_studios = setup_sidebar_filters()

# Set Search query
def handle_search(query):
    df = load_data()
    query = query.lower().strip()
    title_search = df['originalTitle'].str.lower().str.contains(query, case=False)
    actor_search = df['cast'].apply(lambda x: any(query in actor.lower().strip() for actor in x))
    results_df = df[title_search | actor_search]
    
    movie_names = results_df['originalTitle'].tolist()
    actor_names = [actor for actors in results_df['cast'].tolist() for actor in actors]
    
    return movie_names + actor_names

# Set the session state main page gallery
if 'page_state' not in st.session_state:
    st.session_state['page_state'] = "gallery"
    st.session_state['movies_shown'] = 32

#Get the image for each movies from TMDB
def poster_url(title_id, df):
    try:
        poster_path = df[df['titleId'] == title_id]['poster_path'].values[0]
        return 'https://image.tmdb.org/t/p/w500' + poster_path
    except Exception as e:
        st.error(f"Erreur pour titleId {title_id}: {str(e)}")
        return None
    
# For showing clearly the number of voters (Ex 10 K votes instead 10.000)
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

# Créez une fonction pour formater la durée en heures et minutes
# To format duration to hour and minute
def format_duration(minutes):
    hours = minutes // 60
    minutes = minutes % 60
    if hours == 1:
        return f"{hours}h{minutes:02d}"
    elif hours > 1:
        return f"{hours}h{minutes:02d}"
    else:
        return f"{minutes} minute{'s' if minutes > 1 else ''}"

# Fonction to show html template
def load_html_template(filename, **kwargs):
    with open(filename, 'r') as file:
        html_template = file.read()
    for key, value in kwargs.items():
        html_template = html_template.replace(f"{{{{ {key} }}}}", str(value))
    return html_template

# The way the movies are showed using template (in the half part)
def display_movies(df_to_display):
    cols = st.columns(4)
    for i, movie_row in enumerate(df_to_display.iloc[:st.session_state['movies_shown']].itertuples()):
        # Preparing data for the template
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

        # For allow clicking movies
        with cols[i % 4]:
            clicked = click_detector(html_content)
            if clicked != "":
                st.session_state['selected_movie_details'] = clicked
                st.session_state['current_view'] = 'details_page'
                st.rerun()
        # To set 4 rows of movies
        if (i + 1) % 4 == 0:
            cols = st.columns(4)

    # If bottom page, click 'Show more' button to add 16 movies
    if len(df_to_display) > st.session_state['movies_shown']:
        if st.button("Afficher plus", key='unique_key_afficher_plus'):
            st.session_state['movies_shown'] += 16
            st.rerun()

# To sort movies by default as they are popular 
def sort_popular_movies(df):
    return df.sort_values(by='popularity', ascending=False)
df_sorted_popular = sort_popular_movies(df)

# Messages for the use holding cases
def display_status_message(df, df_filtered, search_query, genres_filter, year_range, rating_range, selected_studios):
    filters_modified = (
        genres_filter != default_genres_filter or
        year_range != default_year_range or
        rating_range != default_rating_range or
        selected_studios != default_studios_range
    )
    search_performed = search_query != default_search_query

    if df_filtered.empty and search_performed:
        if filters_modified:
            st.warning(f"A movie or actor contains '{search_query}' but not according to your filters...")
        else:
            st.warning(f"Movie or actor '{search_query}' unknown or invalid...")
    elif filters_modified or search_performed:
        st.markdown(f"### Your results for : {search_query}")
    else:
        st.markdown("### Last popular movies :")

# Define a function to apply sorting based on the selected option
def apply_sorting(df, sort_option):
    if sort_option == "Title - A to Z":
        return df.sort_values(by='originalTitle', ascending=True)
    elif sort_option == "Title - Z to A":
        return df.sort_values(by='originalTitle', ascending=False)
    elif sort_option == "Year - ascending":
        return df.sort_values(by='release_date', ascending=True)
    elif sort_option == "Year - descending":
        return df.sort_values(by='release_date', ascending=False)
    elif sort_option == "Rating - ascending":
        return df.sort_values(by='averageRating', ascending=True)
    elif sort_option == "Rating - descending":
        return df.sort_values(by='averageRating', ascending=False)
    return df  # Return the DataFrame unsorted if no valid option is chosen

if st.session_state['current_view'] == 'main_page':
    col1, col2, col3 = st.columns([4, 2, 2])

    with col1:
        # Hold temporarily the search query
        current_query = st.session_state.get('search_query', '')

        # Search input field
        new_search_query = st.text_input(
            label="Search Input",  # Non-empty label for accessibility
            placeholder="Search for movie or actors",
            value=current_query,
            label_visibility="collapsed"  # Hide the label visually
        )

        # Check if the user has submitted a new query
        if new_search_query and new_search_query != current_query:
            st.session_state['search_query'] = new_search_query
            st.session_state['search_results'] = True
            # Réinitialiser le champ de saisie
            st.rerun()

    with col2:
        sort_options = ["Title - A to Z", "Title - Z to A", "Year - ascending", "Year - descending", "Rating - ascending", "Rating - descending"]
        sort_option = st.selectbox(
            label="Sort Options",
            options=sort_options,
            index=None,
            placeholder="Sort ⬆︎⬇︎ :",
            label_visibility="collapsed"
        )

    if st.session_state['search_results']:
        df_filtered = filter_data(df, genres_filter, year_range, rating_range, selected_studios, st.session_state['search_query'])
        # Réinitialiser l'affichage des résultats si l'utilisateur clique sur "Back to popular movies"
        if st.button("⬅ Back to popular movies"):
            st.session_state['search_results'] = False
            st.session_state['search_query'] = ""
            st.rerun()
    else:
        df_filtered = filter_data(sort_popular_movies(df), genres_filter, year_range, rating_range, selected_studios, "")

    df_filtered = apply_sorting(df_filtered, sort_option)
    display_status_message(df, df_filtered, st.session_state['search_query'], genres_filter, year_range, rating_range, selected_studios)
    display_movies(df_filtered)


# ----------------------- MOVIE DETAILS PAGE LOGIC -----------------------
    
top_placeholder = st.empty()

def show_movie_details(unique_id):
    # Extract the movie titleId 
    title_id = unique_id.replace('movie-', '')

    if title_id in df['titleId'].values:
        movie_data = df[df['titleId'] == title_id].iloc[0]

        # Use primaryTitle for the movie title
        movie_title = movie_data['primaryTitle']

        trailer_url = movie_data['trailer_url']
        if pd.isna(trailer_url):
            trailer_url = None

        num_votes_formatted = format_votes(int(movie_data['numVotes']))
        formatted_genres = movie_data['genres'].replace(',', ', ')

        #num_votes = int(movie_data['numVotes'])  # Convert to integer to remove decimal
        num_votes_formatted = format_votes(int(movie_data['numVotes']))

        runtime_minutes = int(movie_data['runtimeMinutes'])
        duration_text = format_duration(runtime_minutes)
        selected_movie_description = movie_data['overview']
        poster_img_url = poster_url(title_id, df)

        # Managing missing values with '/'
        writer = ', '.join(ast.literal_eval(movie_data['writers'])) if pd.notna(movie_data['writers']) else "/"
        writer = "/" if not writer else writer  
        director = movie_data['director'] if pd.notna(movie_data['director']) else "/"
                # Extraction et formatage des studios
        studios = ', '.join([studio['name'] for studio in ast.literal_eval(movie_data['production_companies'])]) if pd.notna(movie_data['production_companies']) else "/"
        sortie = movie_data['release_date'].year if pd.notna(movie_data['release_date']) else "/"
        duree = duration_text if pd.notna(movie_data['runtimeMinutes']) else "/"

        # Building iframe for the traler
        youtube_embed_html = ""
        if trailer_url:
            youtube_embed_url = f"https://www.youtube.com/embed/{trailer_url.split('watch?v=')[-1]}"
            youtube_embed_html = f"""<iframe width="430" height="215" src="{youtube_embed_url}" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>"""
        
        # Display detailed informations about the movie
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

        # Show actor informations
        st.markdown("""
            <div style="margin-top: 20px;">
                <h3 style="margin-bottom: 10px;">Actors :</h3>
            </div>
        """, unsafe_allow_html=True)

        actors_info = extract_actor_info(ast.literal_eval(movie_data['cast']))

        # Divide actor picture a group of 5
        actor_groups = [actors_info[i:i+5] for i in range(0, len(actors_info), 5)]
        for group in actor_groups:
            columns = st.columns(len(group))  # Cela crée un nombre de colonnes égal à la taille du groupe
            for index, actor in enumerate(group):
                with columns[index]:  # Utilisez 'columns[index]' pour chaque acteur
                    st.markdown(f"""
                        <div style="display: flex; flex-direction: column; align-items: center; margin-bottom: 5px;">
                            <div style="width: 100px; height: 150px; overflow: hidden; border-radius: 10px; text-align: center;">
                                <img src="{actor[1]}" style="width: 100%; height: 100%; object-fit: cover; border-radius: 10px;">
                            </div>
                            <p style="margin: 5px 0 15px;">{actor[0]}</p>
                        </div>
                    """, unsafe_allow_html=True)

        # Showing recommanded movies avec the actors part            
        st.markdown("### Top 50 Recommended Movies:")

        # Get the title from the titleId
        title_id = st.session_state['selected_movie_details'].replace('movie-', '')
        selected_movie_title = df[df['titleId'] == title_id]['originalTitle'].iloc[0]

        # Get recommandations
        recommended_movies = get_recommendations_knn(selected_movie_title, num_recommendations=200)
        cols = st.columns(4)

        # Showing first 52 movies
        if 'movies_shown' not in st.session_state:
            st.session_state['movies_shown'] = 52
        for i, movie_row in recommended_movies.iterrows():
            if i >= st.session_state['movies_shown']:  
                break

            original_movie_row = df.loc[movie_row.name]

            movie_data = {
                'movie_id': f"movie-{movie_row['titleId']}",
                'poster_url': f"https://image.tmdb.org/t/p/w500{movie_row['poster_path']}",
                'average_rating': movie_row['averageRating'],  
                'votes_formatted': format_votes(original_movie_row['numVotes']),
                'movie_title': escape_html(original_movie_row['primaryTitle']),
                'genres': original_movie_row['genres'],
                'movie_year': movie_row['release_date'].year if pd.notnull(original_movie_row['release_date']) else 'Inconnue',
            }
            html_content = load_html_template('./html_script/movie_display.html', **movie_data)
        

            with cols[i % 4]:
                clicked = click_detector(html_content)
                if clicked != "":
                    st.session_state['selected_movie_details'] = clicked
                    st.session_state['current_view'] = 'details_page'
                    st.rerun()

            if (i + 1) % 4 == 0:
                cols = st.columns(4)

        if len(recommended_movies) > st.session_state['movies_shown']:
            if st.button("Afficher plus", key='unique_key_afficher_plus_detailed_movie'):
                st.session_state['movies_shown'] += 16 
                st.rerun()            
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

# For the movie detailed page
if st.session_state['current_view'] == 'details_page':
    back_button_text = '⬅ Back to results' if st.session_state['search_results'] else '⬅ Back to popular movies'

    if st.button(back_button_text if back_button_text else "Default Label"):
        if st.session_state['search_results']:
            # Reload the previous page whiting the previous result
            st.session_state['current_view'] = 'main_page'
            st.rerun()
        else:
            # Back to the main page without query
            st.session_state['current_view'] = 'main_page'
            st.session_state['search_results'] = False
            st.session_state.pop('search_query', None)
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

#-------------------------- END ----------------------------------------#