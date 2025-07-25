# config.py
from vertexai.language_models import TextEmbeddingModel
from vertexai.generative_models import GenerativeModel, GenerationConfig, HarmCategory, HarmBlockThreshold, Candidate
from pandas.api.types import is_numeric_dtype
import faiss 

# GCP & BigQuery
PROJECT_ID = "#############################"
DATASET_ID = "########################"
LOCATION = "europe-west1"
TABLE_PREFIX_FILTER = "######################"

# Vertex AI
EMBEDDING_MODEL_NAME = "text-multilingual-embedding-002"
MODEL_ID = "gemini-1.5-pro-002"

BQ_DATASET_REF = f"{PROJECT_ID}.{DATASET_ID}"
# Pipeline RAG
MAX_SAMPLE_VALUES = 10
EMBEDDING_BATCH_SIZE = 5
RELEVANCE_THRESHOLD = 0.2
MAX_RESULTS_FOR_CONTEXT = 20
MAX_CONVERSATION_HISTORY = 5

# Inner Product for cosine similarity
FAISS_INDEX_TYPE = faiss.IndexFlatIP

# Generation configs

SQL_GENERATION_CONFIG = GenerationConfig(temperature=0.0, top_p=0.95, max_output_tokens=1024)
ANSWER_GENERATION_CONFIG = GenerationConfig(temperature=0.3, top_p=0.95, max_output_tokens=1024)
SAFETY_SETTINGS = {
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    # … autres catégories …
}

# Safety Settings
SAFETY_SETTINGS = {
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
}

# UI
APP_TITLE = "Renault Geo-Mobility Explorer"
LOGO_PATH = "logo.png"
BACKGROUND_IMAGE_PATH = "logo 2.jpg"

FAQS =[
    "Quel est le nom et la population de la commune de Paris (code 75056)?",
    "Quelle est la densité de population (hab/km²) de la commune de Lyon (code 69381)",
    "Parmi les communes situées dans la zone à faibles émissions (ZFE) de Paris, quelles sont les cinq plus peuplées",
    "Quel est le nombre total de communes par département dans la région Île-de-France (code 11) ?",
    "Liste des départements et de leurs régions correspondantes (nom de la région) ?",
    "Quel est le code INSEE de la commune de Bordeaux ?",
    "À quelle région appartient la commune de Nice ?",
    "Quels arrondissements composent la ville de Paris ? ",
    "Donne-moi la liste des communes du département de la Loire (42)",
    "Quel est le code de bassin de vie pour la commune de Lille ?",
    "Quelles communes appartiennent à la ZFE Lyon et depuis quelle date ?",
    "La commune de Saint-Denis a-t-elle connu une fusion, un passage ou une scission récente ?",
    "Quelle est la répartition du parc VP en France par tranche d’âge de véhicules ? ",
    "Donne-moi la composition du parc de véhicules utilitaires légers à Toulouse en 2020 (par carburant).",
    "Quelle est la proportion de véhicules particuliers de moins de 5 ans en France ?",
    "Quelle est la répartition par tranche d’âge du parc VP dans la commune de Marseille ?",
    "Quelle part du parc VP national est diesel vs essence en 2022 ?",
    "Quel est pour Toulouse en 2020 le pourcentage de VP fonctionnant à l’électricité ?",
    "Comment a évolué la part des VP de plus de 15 ans entre 2011 et 2019 ? ",
    "Quelle est la répartition du parc de véhicules particuliers par secteur économique en 2019 ?",
    "Quelle est la distribution par tranche d’âge des véhicules utilitaires légers en France en 2022 ?",
    "Quel pourcentage du parc VUL à Lyon en 2021 fonctionne au gaz ?",
    "Combien de VP en France détenaient une vignette Crit’Air 1 vs Crit’Air 2 en 2019 ?",
    "Compare la part des véhicules particuliers et des utilitaires légers fonctionnant au GPL à l’échelle nationale en 2022"]
