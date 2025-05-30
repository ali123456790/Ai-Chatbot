# Fast Configuration for Quick Development
# Use this for faster startup times

class FastConfig:
    """Ultra-fast configuration for development and quick testing."""
    
    # Skip topic modeling entirely for speed
    ENABLE_TOPIC_MODELING = False
    ENABLE_HIERARCHICAL_TOPICS = False  
    ENABLE_DYNAMIC_TOPICS = False
    
    # Use lighter spaCy model for speed
    SPACY_MODEL_NAME = 'en_core_web_sm'  # Much faster than 'en_core_web_trf'
    
    # Reduced embedding batch sizes for faster processing
    BATCH_SIZE_EMBEDDING = 32  # Larger batches = faster
    OPTIMAL_BATCH_SIZE_EMBEDDING = 32
    
    # Fewer workers to avoid overhead
    MAX_WORKERS = 4
    
    # Skip QA model for ultra-fast startup (can enable later)
    ENABLE_QA_MODEL = True  # Set to False to skip QA loading
    
    # Use smaller checkpoint intervals
    EMBEDDING_CHECKPOINT_INTERVAL = 2000
    
    # Quick topic modeling if enabled
    SAMPLE_SIZE_FOR_TOPIC_TRAINING = 100  # Very small sample
    NUM_TOPICS = 10  # Fewer topics
    
    # Fast text processing
    MAX_TOKENS_FOR_GENSIM = 200  # Fewer tokens
    MIN_TOKENS_PER_DOCUMENT = 3
    
    # Skip heavy preprocessing
    ENABLE_NGRAMS = False  # Skip bigrams/trigrams for speed

# Usage: python chat.py --config fast --create-index --xmldir xmlfiles --sample-size 1000 