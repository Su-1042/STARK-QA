# STARK-QA Project Configuration

# API Configuration
MISTRAL_MODEL = "mistral-medium"
MISTRAL_TEMPERATURE = 0.1
MISTRAL_MAX_TOKENS = 200

# API Retry Configuration (for rate limiting)
API_RETRY_MAX_ATTEMPTS = 3
API_RETRY_INITIAL_DELAY = 1.0  # seconds
API_RETRY_EXPONENTIAL_BASE = 2  # for exponential backoff

# RAG Configuration
RAG_CHUNK_SIZE = 512
RAG_CHUNK_OVERLAP = 50
RAG_TOP_K = 5
RAG_HYBRID_ALPHA = 0.7  # Weight for dense retrieval

# Knowledge Graph Configuration
KG_MAX_ENTITIES = 10
KG_MAX_HOPS = 2
KG_ENTITY_THRESHOLD = 0.3  # Minimum similarity for entity relevance

# Evaluation Configuration
EVAL_METRICS = [
    'rouge1', 'rouge2', 'rougeL', 
    'bleu', 'bert_score', 
    'flesch_reading_ease', 'gunning_fog'
]

# Model Configuration
SENTENCE_TRANSFORMER_MODEL = 'all-MiniLM-L6-v2'
DENSE_EMBEDDING_MODEL = 'all-mpnet-base-v2'
SPACY_MODEL = 'en_core_web_sm'

# File Paths
DEFAULT_CACHE_DIR = "cache/"
DEFAULT_OUTPUT_DIR = "results/"
DEFAULT_DATA_DIR = "data/"

# Logging Configuration
LOG_LEVEL = "INFO"
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'