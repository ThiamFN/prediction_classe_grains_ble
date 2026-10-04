"""
Application Streamlit — Clustering des graines de blé (KMeans)

Lancement en local :  streamlit run app.py
"""

import numpy as np
import pandas as pd
import joblib as jb
import streamlit as st

# Configuration de la page
st.set_page_config(
    page_title="Clustering des graines de blé",
    page_icon="🌾",
    layout="centered",
)

DESCRIPTION = (
    "Ce modèle de clustering regroupe les graines de blé selon leurs "
    "caractéristiques physiques (aire, périmètre, compacité, dimensions du noyau...)."
)

COLONNES_FEATURES = [
    "area A", "perimeter", "compactness", "length of kernel",
    "width of kernel", "asymmetry coefficient", "length of kernel groove",
]


# Chargement des artefacts (mis en cache : chargés une seule fois)
@st.cache_resource
def load_artifacts():
    scaler = jb.load("scaler.joblib")           # normaliseur
    kmeans = jb.load("kmeans_model.joblib")      # modèle KMeans
    return scaler, kmeans


scaler, kmeans = load_artifacts()
n_clusters = kmeans.n_clusters


# Fonction de prédiction simple
def pred_func(valeurs: dict):
    entree = pd.DataFrame([valeurs], columns=COLONNES_FEATURES)
    x_new = scaler.transform(entree)
    cluster = kmeans.predict(x_new)[0]
    # Distance au centroïde du cluster attribué (indicateur de confiance)
    distance = np.linalg.norm(x_new[0] - kmeans.cluster_centers_[cluster])
    return int(cluster), float(distance)


# Fonction de prédiction multiple à partir d'un CSV
def pred_func_csv(file):
    df = pd.read_csv(file)
    entree = df[COLONNES_FEATURES]
    x_new = scaler.transform(entree)
    clusters = kmeans.predict(x_new)
    df["cluster"] = clusters
    return df


# Interface
st.title("🌾 Clustering des graines de blé")

onglet1, onglet2 = st.tabs(["Prédiction simple", "Prédiction multiple"])

# ----------------------------- Onglet 1 -------------------------------
with onglet1:
    st.subheader("Assigner un cluster à une graine à partir de ses mesures")
    st.write(DESCRIPTION)

    with st.form("formulaire_simple"):
        col1, col2 = st.columns(2)

        with col1:
            area = st.number_input("Aire (area A)", min_value=0.0, value=14.9, step=0.1, format="%.2f")
            perimeter = st.number_input("Périmètre", min_value=0.0, value=14.6, step=0.1, format="%.2f")
            compactness = st.number_input("Compacité", min_value=0.0, max_value=1.0, value=0.87, step=0.001, format="%.4f")
            length_kernel = st.number_input("Longueur du noyau", min_value=0.0, value=5.64, step=0.01, format="%.3f")

        with col2:
            width_kernel = st.number_input("Largeur du noyau", min_value=0.0, value=3.27, step=0.01, format="%.3f")
            asymmetry = st.number_input("Coefficient d'asymétrie", min_value=0.0, value=3.70, step=0.01, format="%.3f")
            groove = st.number_input("Longueur du sillon du noyau", min_value=0.0, value=5.42, step=0.01, format="%.3f")

        soumettre = st.form_submit_button("Prédire le cluster", type="primary")

    if soumettre:
        try:
            valeurs = {
                "area A": area,
                "perimeter": perimeter,
                "compactness": compactness,
                "length of kernel": length_kernel,
                "width of kernel": width_kernel,
                "asymmetry coefficient": asymmetry,
                "length of kernel groove": groove,
            }
            cluster, distance = pred_func(valeurs)
            st.success(f"**Cluster attribué : {cluster}** (sur {n_clusters} clusters, 0 à {n_clusters - 1})")
            st.caption(f"Distance au centroïde du cluster : {distance:.3f} (plus c'est bas, plus la graine est typique de ce cluster)")
        except Exception as e:
            st.error(f"Erreur lors de la prédiction : {e}")

# ----------------------------- Onglet 2 -------------------------------
with onglet2:
    st.subheader("Assigner un cluster à plusieurs graines à partir d'un fichier CSV")
    st.write(DESCRIPTION)
    st.caption(
        "Le fichier CSV doit contenir, avec ces noms de colonnes exacts : "
        + ", ".join(COLONNES_FEATURES)
    )

    fichier = st.file_uploader("Importer un fichier CSV", type=["csv"])

    if fichier is not None:
        try:
            with st.spinner("Calcul des clusters en cours…"):
                df_resultat = pred_func_csv(fichier)

            st.success(f"{len(df_resultat)} graine(s) classée(s).")
            st.dataframe(df_resultat, use_container_width=True)

            st.download_button(
                label="⬇️ Télécharger le fichier CSV",
                data=df_resultat.to_csv(index=False).encode("utf-8"),
                file_name="clusters.csv",
                mime="text/csv",
                type="primary",
            )
        except Exception as e:
            st.error(f"Erreur lors du traitement du fichier : {e}")
