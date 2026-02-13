import streamlit as st
import os
import hashlib
import chromadb
import google.generativeai as genai
from openai import OpenAI
import pandas as pd
import numpy as np
from collections import Counter
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

# ============================================================
# CONFIGURACIÓN GENERAL
# ============================================================
st.set_page_config(page_title="Chat Instagram - Análisis de Seguidos")

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
client_ai = OpenAI(
  base_url="https://openrouter.ai/api/v1",
  api_key=os.getenv("OPENAI_API_KEY"),
)

EMBEDDING_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
client = chromadb.Client()

# ============================================================
# SESSION STATE
# ============================================================
if "collection" not in st.session_state:
    st.session_state.collection = None

if "file_processed" not in st.session_state:
    st.session_state.file_processed = False

if "file_hash" not in st.session_state:
    st.session_state.file_hash = None

if "df" not in st.session_state:
    st.session_state.df = None

# ============================================================
# FUNCIONES
# ============================================================

def load_instagram_csv(csv_path):
    """Carga el CSV de Instagram y lo convierte en texto estructurado por perfil."""
    df = pd.read_csv(csv_path)

    # Clasificar bios por tema
    if 'biography' in df.columns:
        df['tema_bio'] = df['biography'].apply(classify_bio_theme)

    text_parts = []
    text_parts.append(f"Total de seguidos analizados: {len(df)}\n")

    # Resumen por patrón
    if 'analisis_patron' in df.columns:
        patron_counts = df['analisis_patron'].value_counts()
        text_parts.append("Resumen de patrones encontrados:")
        for patron, count in patron_counts.items():
            text_parts.append(f"  - {patron}: {count} usuarios")
        text_parts.append("")

    # Cada perfil como un bloque de texto
    for _, row in df.iterrows():
        perfil = f"Perfil: @{row.get('username', 'N/A')}"
        perfil += f" | Nombre: {row.get('full_name', 'N/A')}"
        perfil += f" | Bio: {row.get('biography', 'Sin bio')}"
        perfil += f" | Seguidores: {row.get('follower_count', 0)}"
        perfil += f" | Seguidos: {row.get('following_count', 0)}"
        perfil += f" | Privada: {row.get('is_private', 'N/A')}"
        perfil += f" | Verificada: {row.get('is_verified', 'N/A')}"
        if 'analisis_patron' in df.columns:
            perfil += f" | Patrón: {row.get('analisis_patron', 'N/A')}"
        perfil += f" | Tema: {row.get('tema_bio', 'N/A')}"
        text_parts.append(perfil)

    return "\n".join(text_parts), df


def chunk_text(text):
    """Divide texto en fragmentos con solapamiento."""
    chunk_size = 500
    overlap = 100
    chunks = []
    start = 0
    chunk_id = 0

    while start < len(text):
        chunk_content = text[start:start + chunk_size]
        chunks.append({
            "id": f"chunk_{chunk_id}",
            "content": chunk_content,
            "start_index": start,
            "size": len(chunk_content)
        })
        chunk_id += 1
        start += chunk_size - overlap

    return chunks


def create_chroma_collection(chunks):
    """Crea colección en ChromaDB a partir de los chunks."""
    try:
        client.delete_collection("instagram_rag")
    except:
        pass

    collection = client.create_collection(name="instagram_rag")
    texts = [c["content"] for c in chunks]
    embeddings = EMBEDDING_MODEL.encode(texts)

    collection.add(
        documents=texts,
        embeddings=embeddings.tolist(),
        ids=[c["id"] for c in chunks],
        metadatas=[
            {
                "chunk_index": i,
                "start_index": c["start_index"],
                "chunk_size": c["size"]
            }
            for i, c in enumerate(chunks)
        ]
    )
    return collection


def retrieve_context(collection, query, k=6):
    """Recupera los k chunks más similares a la pregunta."""
    query_embedding = EMBEDDING_MODEL.encode([query])
    results = collection.query(
        query_embeddings=query_embedding.tolist(),
        n_results=k
    )
    return results


def build_structured_data(df):
    """Construye datos estructurados del DataFrame relevantes a la pregunta."""
    parts = []

    # Estadísticas generales
    parts.append(f"Total perfiles: {len(df)}")
    parts.append(f"Promedio seguidores: {df['follower_count'].mean():.0f}")
    parts.append(f"Promedio seguidos: {df['following_count'].mean():.0f}")
    parts.append(f"Cuentas privadas: {int(df['is_private'].sum())}")
    parts.append(f"Cuentas verificadas: {int(df['is_verified'].sum())}")

    # Usuarios por patrón (listar todos los usernames por categoría)
    if 'analisis_patron' in df.columns:
        parts.append("\n--- USUARIOS POR PATRON ---")
        for patron in df['analisis_patron'].unique():
            grupo = df[df['analisis_patron'].str.contains(patron, na=False)]
            usernames = grupo.apply(
                lambda r: f"@{r['username']} (seguidores:{r['follower_count']}, seguidos:{r['following_count']})",
                axis=1
            ).tolist()
            parts.append(f"\n{patron} ({len(usernames)} usuarios):")
            for u in usernames:
                parts.append(f"  - {u}")

    # Usuarios por tema de bio
    if 'tema_bio' in df.columns:
        parts.append("\n--- USUARIOS POR TEMA DE BIO ---")
        for tema in df['tema_bio'].value_counts().index:
            grupo = df[df['tema_bio'].str.contains(tema, na=False)]
            usernames = grupo.apply(
                lambda r: f"@{r['username']} (bio: {str(r.get('biography', ''))[:80]})",
                axis=1
            ).tolist()
            parts.append(f"\n{tema} ({len(usernames)} usuarios):")
            for u in usernames:
                parts.append(f"  - {u}")

    # Top 20 por seguidores
    top = df.nlargest(20, 'follower_count')
    parts.append("\n--- TOP 20 MAS SEGUIDORES ---")
    for _, r in top.iterrows():
        parts.append(f"  @{r['username']}: {r['follower_count']} seguidores | Bio: {r.get('biography', '')}")

    # Benford
    for col in ['follower_count', 'following_count']:
        bdf = benford_analysis(df[col])
        if bdf is not None:
            max_diff = bdf['Diferencia'].max()
            parts.append(f"\nBenford ({col}): desviacion maxima = {max_diff:.1%}")

    return "\n".join(parts)


def ask_ia(provider, context, question, df):
    """Llama a la IA con contexto RAG + datos estructurados del DataFrame."""
    structured = build_structured_data(df)

    prompt = f"""
    Eres un asistente experto en análisis de redes sociales.
    Analizas los seguidos de una cuenta de Instagram.
    Tienes acceso a datos estructurados completos y contexto adicional.
    Responde con la información proporcionada. Si piden listar usuarios, usa los datos estructurados.

    DATOS ESTRUCTURADOS:
    {structured}

    CONTEXTO ADICIONAL (RAG):
    {context}

    Pregunta del usuario:
    {question}
    """

    try:
        if provider == "Gemini (Google)":
            model = genai.GenerativeModel("models/gemini-2.5-flash-lite")
            response = model.generate_content(prompt)
            return response.text

        elif provider == "ChatGPT (OpenAI)":
            response = client_ai.chat.completions.create(
                model="openai/gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}]
            )
            return response.choices[0].message.content
    except Exception as e:
        return f"Error con el proveedor {provider}: {str(e)}"


def benford_analysis(series):
    """
    Aplica la Ley de Benford a una serie numérica.
    Retorna un DataFrame con dígitos 1-9, frecuencia observada y esperada.
    """
    # Distribución esperada según Benford
    benford_expected = {d: np.log10(1 + 1/d) for d in range(1, 10)}

    # Extraer primer dígito de cada valor > 0
    first_digits = []
    for val in series:
        val = abs(int(val))
        if val > 0:
            first_digit = int(str(val)[0])
            first_digits.append(first_digit)

    if not first_digits:
        return None

    total = len(first_digits)
    digit_counts = pd.Series(first_digits).value_counts().sort_index()

    rows = []
    for d in range(1, 10):
        observed = digit_counts.get(d, 0) / total
        expected = benford_expected[d]
        rows.append({
            "Digito": d,
            "Observado": round(observed, 4),
            "Esperado (Benford)": round(expected, 4),
            "Diferencia": round(abs(observed - expected), 4),
            "Cantidad": digit_counts.get(d, 0)
        })

    result_df = pd.DataFrame(rows)
    return result_df


# Categorías temáticas por palabras clave en bio
THEME_KEYWORDS = {
    "Deportes/Fitness": ["futbol", "football", "soccer", "basket", "gym", "fitness", "crossfit",
                         "runner", "running", "athlete", "atleta", "entrenador", "coach", "deporte",
                         "sport", "tennis", "boxing", "boxeo", "yoga", "surf", "ciclismo", "cycling"],
    "Musica/Arte": ["musica", "music", "cantante", "singer", "rapper", "dj", "productor", "producer",
                    "artista", "artist", "arte", "pintor", "painter", "actor", "actriz", "actress",
                    "teatro", "band", "banda", "guitarra", "piano", "drummer"],
    "Belleza/Moda": ["makeup", "maquillaje", "beauty", "belleza", "fashion", "moda", "modelo", "model",
                     "skincare", "nails", "uñas", "hair", "cabello", "stylist", "estilista", "outfit"],
    "Tecnologia": ["developer", "desarrollador", "programador", "programmer", "software", "tech",
                   "codigo", "code", "data", "ai", "web", "devops", "hacker", "ciberseguridad",
                   "cybersecurity", "startup", "fintech", "blockchain", "crypto"],
    "Negocios/Emprendimiento": ["ceo", "founder", "fundador", "entrepreneur", "emprendedor", "empresa",
                                 "business", "marketing", "ventas", "sales", "manager", "gerente",
                                 "director", "consultor", "consultant", "coach empresarial", "negocio"],
    "Politica/Activismo": ["politica", "politics", "diputado", "senador", "congresista", "presidente",
                           "activista", "activist", "derechos", "rights", "justicia", "democracia",
                           "partido", "gobierno", "government", "feminista", "feminist"],
    "Educacion": ["profesor", "teacher", "educacion", "education", "universidad", "university",
                  "estudiante", "student", "docente", "academia", "investigador", "researcher",
                  "phd", "maestro", "escuela", "school"],
    "Fotografia/Video": ["fotografo", "photographer", "fotografia", "photography", "videographer",
                         "filmmaker", "cineasta", "video", "camera", "foto", "retrato", "portrait"],
    "Gastronomia": ["chef", "cocina", "cooking", "food", "comida", "restaurante", "restaurant",
                    "recetas", "recipes", "foodie", "gastronomia", "panadero", "baker", "barista"],
    "Viajes": ["travel", "viaje", "viajero", "traveler", "wanderlust", "mochilero", "backpacker",
               "turismo", "tourism", "aventura", "adventure", "explore", "nomad"],
    "Salud/Bienestar": ["doctor", "medico", "salud", "health", "psicologo", "psychologist",
                        "terapeuta", "therapist", "nutricion", "nutrition", "bienestar", "wellness",
                        "mental health", "enfermero", "nurse", "dentista"],
}


def get_top_words(df, top_n=20):
    """Extrae las palabras mas frecuentes de las bios, filtrando stopwords."""
    stopwords = {
        "de", "la", "el", "en", "y", "a", "los", "las", "del", "un", "una", "que",
        "es", "por", "con", "para", "al", "se", "lo", "no", "mi", "me", "te", "tu",
        "su", "nos", "le", "si", "ya", "o", "e", "i", "the", "and", "of", "to", "in",
        "is", "for", "on", "my", "it", "at", "we", "be", "do", "an", "or", "so", "if",
        "he", "she", "as", "up", "all", "but", "not", "you", "are", "was", "has", "had",
        "nan", "sin", "bio", "com", "www", "https", "http"
    }
    all_words = []
    for bio in df['biography'].dropna():
        words = str(bio).lower().split()
        for w in words:
            w = w.strip(".,;:!?¡¿()[]{}\"'@#$%^&*-_=+<>/\\|~`")
            if len(w) > 2 and w not in stopwords and not w.isdigit():
                all_words.append(w)
    return Counter(all_words).most_common(top_n)


def classify_bio_theme(bio):
    """Clasifica una bio en temas según palabras clave."""
    if not isinstance(bio, str) or not bio.strip():
        return "Sin bio"

    bio_lower = bio.lower()
    matched = []

    for theme, keywords in THEME_KEYWORDS.items():
        if any(kw in bio_lower for kw in keywords):
            matched.append(theme)

    return ", ".join(matched) if matched else "Otros"


# ============================================================
# INTERFAZ
# ============================================================

st.title("Chat Instagram - Analisis de Seguidores")

with st.sidebar:
    st.header("Configuracion")
    ai_provider = st.selectbox("Elegir IA", ["Gemini (Google)", "ChatGPT (OpenAI)"])
    st.info(f"Modelo: {'gemini-2.5-flash-lite' if 'Gemini' in ai_provider else 'gpt-4o-mini'}")

# Seleccionar archivo CSV
# uploaded_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "nayeli.nxx_seguidos_p2.csv")
uploaded_file = st.file_uploader("Selecciona un archivo CSV de seguidos", type=["csv"])

# Detectar cambio de archivo y resetear estado
if uploaded_file:
    current_hash = hashlib.sha256(uploaded_file.getvalue()).hexdigest()
    if st.session_state.file_hash != current_hash:
        st.session_state.file_hash = current_hash
        st.session_state.file_processed = False
        st.session_state.collection = None
        st.session_state.df = None

# Para leer el archivo directo sin file_uploader:
# uploaded_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "nayeli.nxx_seguidos_p2.csv")
# Y reemplazar el bloque del hash (líneas 332-339) por:
# if not st.session_state.file_processed:
#     current_hash = hashlib.sha256(open(uploaded_file, 'rb').read()).hexdigest()
#     if st.session_state.file_hash != current_hash:
#         st.session_state.file_hash = current_hash


# Procesar archivo
if uploaded_file and not st.session_state.file_processed:
    if st.button("Cargar datos de Instagram"):
        with st.spinner("Procesando datos de seguidos..."):
            raw_text, df = load_instagram_csv(uploaded_file)
            st.session_state.df = df
            chunks = chunk_text(raw_text)
            st.session_state.collection = create_chroma_collection(chunks)
            st.session_state.file_processed = True
            st.success(f"Listo! {len(df)} perfiles cargados en {len(chunks)} fragmentos.")

# Mostrar resumen
if st.session_state.file_processed and st.session_state.df is not None:
    df = st.session_state.df

    col1, col2, col3 = st.columns(3)
    col1.metric("Total seguidos", len(df))
    col2.metric("Privadas", int(df['is_private'].sum()))
    col3.metric("Verificadas", int(df['is_verified'].sum()))

    if 'analisis_patron' in df.columns:
        st.subheader("Distribucion de patrones")
        st.bar_chart(df['analisis_patron'].value_counts())

    # Ley de Benford
    st.subheader("Ley de Benford")
    st.caption("Compara la distribucion del primer digito de los datos vs la distribucion teorica de Benford. "
               "Desviaciones grandes pueden indicar datos anomalos o manipulados.")

    benford_col = st.selectbox("Columna a analizar:", ["follower_count", "following_count"])
    benford_df = benford_analysis(df[benford_col])

    if benford_df is not None:
        col_b1, col_b2 = st.columns(2)

        with col_b1:
            chart_data = benford_df.set_index("Digito")[["Observado", "Esperado (Benford)"]]
            st.bar_chart(chart_data)

        with col_b2:
            st.dataframe(benford_df, use_container_width=True, hide_index=True)

        max_diff = benford_df["Diferencia"].max()
        if max_diff < 0.05:
            st.success("Los datos siguen la Ley de Benford (desviacion maxima < 5%)")
        elif max_diff < 0.10:
            st.warning(f"Desviacion moderada detectada (max: {max_diff:.1%})")
        else:
            st.error(f"Desviacion significativa (max: {max_diff:.1%}) - posible anomalia en los datos")
    else:
        st.warning("No hay datos suficientes para el analisis de Benford.")

    # Clustering por tema de bio
    if 'tema_bio' in df.columns:
        st.subheader("Clustering por tema de bio")
        st.caption("Clasificacion automatica de perfiles segun palabras clave en su biografia.")

        theme_counts = df['tema_bio'].value_counts()
        st.bar_chart(theme_counts)

        # Selector para explorar un tema
        selected_theme = st.selectbox("Ver usuarios de un tema:", theme_counts.index.tolist())
        theme_users = df[df['tema_bio'].str.contains(selected_theme, na=False)]
        st.write(f"**{len(theme_users)} usuarios en '{selected_theme}':**")
        st.dataframe(
            theme_users[['username', 'full_name', 'biography', 'follower_count', 'tema_bio']],
            use_container_width=True,
            hide_index=True
        )

    # Palabras mas frecuentes en bios
    if 'biography' in df.columns:
        st.subheader("Palabras mas frecuentes en bios")
        st.caption("Top de palabras que mas aparecen en las biografias (sin stopwords).")
        top_words = get_top_words(df, top_n=25)
        if top_words:
            words_df = pd.DataFrame(top_words, columns=["Palabra", "Frecuencia"])
            st.bar_chart(words_df.set_index("Palabra"))
        else:
            st.warning("No hay suficientes bios para analizar.")

    st.divider()

    # Preguntas sugeridas
    st.subheader("Preguntas sugeridas")
    suggested = [
        "Cuantos influencers sigue?",
        "Que usuarios tienen mas seguidores?",
        "Hay cuentas sospechosas de ser bots?",
        "Que usuarios tienen bio relacionada con negocios?",
        "Cuantas cuentas privadas hay?",
    ]

    selected_q = st.selectbox("Elegir pregunta rapida:", ["Escribir mi pregunta..."] + suggested)

    if selected_q != "Escribir mi pregunta...":
        question = selected_q
    else:
        question = st.text_input("Que quieres saber sobre los seguidos?")

    if st.button("Preguntar") and question:
        with st.spinner(f"Buscando respuesta con {ai_provider}..."):
            results = retrieve_context(st.session_state.collection, question)
            context_text = "\n\n".join(results["documents"][0])
            answer = ask_ia(ai_provider, context_text, question, df)

        st.subheader("Respuesta")
        st.write(answer)

        with st.expander("Contexto usado (detallado)"):
            for i, (doc, meta) in enumerate(
                zip(results["documents"][0], results["metadatas"][0])
            ):
                st.markdown(f"""
**Chunk #{meta['chunk_index']}**
- Inicio en texto: `{meta['start_index']}`
- Tamano: `{meta['chunk_size']}` caracteres

```text
{doc}
""")

    # Tabla completa
    with st.expander("Ver tabla completa de seguidos"):
        st.dataframe(df, use_container_width=True)
