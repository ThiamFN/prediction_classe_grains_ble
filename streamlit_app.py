"""Application Streamlit : segmentation des graines de blé.

Lancer en local :  streamlit run streamlit_app.py
"""
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from scipy.spatial.distance import cdist

CHEMIN_MODELE = "modele_clustering.joblib"


# --- Modèle et prédiction ----------------------------------------------------
@st.cache_resource
def charger_modele(chemin=CHEMIN_MODELE):
    return joblib.load(chemin)


def predire(art, X):
    """Renvoie (groupes, distances, X standardisé) pour un DataFrame X.

    KMeans / CAH : groupe du prototype (centre) le plus proche.
    DBSCAN       : groupe du point coeur le plus proche s'il est à moins de
                   `eps`, sinon -1 (anomalie).
    """
    Xs = art["scaler"].transform(X[art["colonnes"]])
    D = cdist(Xs, art["prototypes"])
    i = D.argmin(axis=1)
    distances = D[np.arange(len(D)), i]
    groupes = np.asarray(art["labels_prototypes"])[i].astype(int)
    if art["eps"] is not None:
        groupes = np.where(distances > art["eps"], -1, groupes)
    return groupes, distances, Xs


def nom_groupe(art, g):
    return art["noms_groupes"].get(int(g), "Anomalie")


def colonnes_hors_plage(art, ligne):
    return [c for c in art["colonnes"]
            if ligne[c] < art["minimum"][c] or ligne[c] > art["maximum"][c]]


# --- Affichage (compatible avec les anciennes et les récentes versions de Streamlit) ---
def afficher_graphique(fig):
    try:
        st.plotly_chart(fig, width="stretch")
    except TypeError:                      # anciennes versions
        st.plotly_chart(fig, use_container_width=True)


def afficher_tableau(tableau):
    try:
        st.dataframe(tableau, width="stretch")
    except TypeError:                      # anciennes versions
        st.dataframe(tableau, use_container_width=True)


# --- Onglet 1 : une graine -----------------------------------------------------
def onglet_une_graine(art):
    colonnes = art["colonnes"]
    st.subheader("Saisir les mesures d'une graine")

    with st.form("saisie"):
        cols = st.columns(2)
        valeurs = {}
        for i, c in enumerate(colonnes):
            etendue = art["maximum"][c] - art["minimum"][c]
            with cols[i % 2]:
                valeurs[c] = st.number_input(
                    c, value=float(art["mediane"][c]),
                    step=float(etendue / 100) or 0.01, format="%.4f")
        envoye = st.form_submit_button("Trouver le groupe", type="primary")

    if not envoye:
        return

    X = pd.DataFrame([valeurs])
    groupes, distances, Xs = predire(art, X)
    g, d = int(groupes[0]), float(distances[0])
    nom = nom_groupe(art, g)

    if g == -1:
        st.warning(f"Graine atypique : elle ne ressemble à aucun groupe "
                   f"(distance {d:.2f} > seuil {art['eps']:.2f}).")
    else:
        st.success(f"{nom}  |  distance au prototype : {d:.2f}")

    hors = colonnes_hors_plage(art, X.iloc[0])
    if hors:
        st.info("Valeur(s) hors de la plage observée à l'entraînement : "
                + ", ".join(hors) + ". La prédiction est moins fiable.")

    # Position de la nouvelle graine sur le plan de l'ACP
    fig = px.scatter(art["train_pca"], x="PC 1", y="PC 2", color="Groupe",
                     opacity=0.6, title="Position de la graine sur le plan de l'ACP")
    p = art["pca"].transform(Xs)
    fig.add_scatter(x=[p[0, 0]], y=[p[0, 1]], mode="markers", name="Nouvelle graine",
                    marker=dict(size=18, symbol="star", color="black"))
    afficher_graphique(fig)


# --- Onglet 2 : un fichier CSV ---------------------------------------------------
def classer_fichier(art, donnees):
    """Ajoute les colonnes Groupe et Distance à un DataFrame. Lève ValueError si le fichier est invalide."""
    colonnes = art["colonnes"]
    manquantes = [c for c in colonnes if c not in donnees.columns]
    if manquantes:
        raise ValueError("Colonnes manquantes : " + ", ".join(manquantes))

    X = donnees[colonnes].apply(pd.to_numeric, errors="coerce")
    valides = X.notna().all(axis=1)

    resultat = donnees.copy()
    resultat["Groupe"] = "Données invalides"
    resultat["Distance"] = np.nan
    if valides.any():
        groupes, distances, _ = predire(art, X[valides])
        resultat.loc[valides, "Groupe"] = [nom_groupe(art, g) for g in groupes]
        resultat.loc[valides, "Distance"] = distances.round(3)
    return resultat


def onglet_fichier(art):
    st.subheader("Classer plusieurs graines à la fois")
    st.write("Le fichier CSV doit contenir ces colonnes : " + ", ".join(art["colonnes"]))

    fichier = st.file_uploader("Fichier CSV", type="csv")
    if fichier is None:
        return

    try:
        donnees = pd.read_csv(fichier, sep=None, engine="python")  # détecte , ou ;
        resultat = classer_fichier(art, donnees)
    except ValueError as e:
        st.error(str(e))
        return
    except Exception:
        st.error("Impossible de lire ce fichier. Vérifiez qu'il s'agit d'un CSV valide.")
        return

    invalides = int((resultat["Groupe"] == "Données invalides").sum())
    if invalides:
        st.warning(f"{invalides} ligne(s) contiennent des valeurs manquantes ou non numériques.")

    afficher_tableau(resultat)

    effectifs = resultat["Groupe"].value_counts().rename_axis("Groupe").reset_index(name="Effectif")
    afficher_graphique(px.bar(effectifs, x="Groupe", y="Effectif", color="Groupe", text_auto=True,
                              title="Répartition des graines par groupe"))

    st.download_button("Télécharger les résultats (CSV)",
                       resultat.to_csv(index=False).encode("utf-8"),
                       file_name="graines_classees.csv", mime="text/csv")


# --- Onglet 3 : profils des groupes ---------------------------------------------
def onglet_profils(art):
    noms = art["noms_groupes"]
    st.subheader("Profil moyen de chaque groupe")
    afficher_tableau(art["profil"].rename(index=noms))

    st.subheader("Écart à la moyenne générale")
    st.caption("Positif : au-dessus de la moyenne. Négatif : en dessous.")
    fig = px.imshow(art["profil_std"].rename(index=noms).round(2), text_auto=True,
                    color_continuous_scale="RdBu_r", color_continuous_midpoint=0, aspect="auto",
                    labels={"x": "Variable", "y": "Groupe", "color": "Écart"})
    afficher_graphique(fig)


# --- Application ---------------------------------------------------------------------
def main():
    st.set_page_config(page_title="Graines de blé", page_icon="🌾", layout="wide")
    st.title("🌾 Segmentation des graines de blé")

    try:
        art = charger_modele()
    except FileNotFoundError:
        st.error(f"Fichier introuvable : {CHEMIN_MODELE}. "
                 "Placez-le dans le même dossier que l'application.")
        st.stop()

    n_groupes = len(art["noms_groupes"])
    st.caption(f"Modèle : {art['type_modele']}  |  {n_groupes} groupes")

    t1, t2, t3 = st.tabs(["Une graine", "Un fichier CSV", "Profils des groupes"])
    with t1:
        onglet_une_graine(art)
    with t2:
        onglet_fichier(art)
    with t3:
        onglet_profils(art)


if __name__ == "__main__":
    main()
