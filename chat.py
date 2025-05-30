import os
# Prevent HuggingFace tokenizers warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import xml.etree.ElementTree as ET
import re
import glob
import argparse
import pickle
import logging
from typing import Optional, List, Dict, Any, Tuple, Any
from pathlib import Path
import time
import torch # Import torch to check for MPS
import hashlib
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
from functools import partial
import sys
import traceback
import gzip

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('chat_analyzer.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# AI Integration: Import spaCy with better error handling
try:
    import spacy
    SPACY_AVAILABLE = True
    logger.info("spaCy library loaded successfully")
except ImportError:
    logger.warning("spaCy library not found. Please install it: pip install spacy")
    logger.warning("You also need to download a model: python -m spacy download en_core_web_sm")
    spacy = None
    SPACY_AVAILABLE = False

# AI Integration: Import sentence-transformers with better error handling
try:
    from sentence_transformers import SentenceTransformer, util
    import numpy as np
    SENTENCE_TRANSFORMERS_AVAILABLE = True
    logger.info("sentence-transformers library loaded successfully")
except ImportError:
    logger.warning("sentence-transformers library not found. Please install it: pip install sentence-transformers")
    SentenceTransformer = None
    util = None
    np = None
    SENTENCE_TRANSFORMERS_AVAILABLE = False

# Gensim for topic modelling with better error handling
try:
    from gensim import corpora, models
    GENSIM_AVAILABLE = True
    logger.info("Gensim library loaded successfully")
except ImportError:
    logger.warning("gensim not installed → topic modelling disabled.")
    GENSIM_AVAILABLE = False

# Phase 2: Enhanced dependencies
try:
    from transformers import pipeline, AutoTokenizer, AutoModelForQuestionAnswering
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    HF_TRANSFORMERS_AVAILABLE = True
    logger.info("Transformers library loaded successfully for QA")
except ImportError:
    HF_TRANSFORMERS_AVAILABLE = False
    logger.warning("Transformers library not available - QA features disabled")

try:
    from gensim.models.ldaseqmodel import LdaSeqModel
    from gensim.models.coherencemodel import CoherenceModel
    from gensim.models.hdpmodel import HdpModel
    ADVANCED_GENSIM_AVAILABLE = True
    logger.info("Advanced Gensim models loaded successfully")
except ImportError:
    ADVANCED_GENSIM_AVAILABLE = False
    logger.warning("Advanced Gensim models not available - hierarchical/dynamic topic modeling disabled")

# Configuration class for better organization
class Config:
    """Configuration settings for the chat analyzer."""
    # Enhanced topic modeling settings for historical documents with n-grams
    NUM_TOPICS = 25  # Reduced from 35 for better performance with 2000 documents

    # Default list of known uploaded files (for testing)
    DEFAULT_UPLOADED_FILES = [
        "42251.xml", "42252.xml", "42253.xml", "42254.xml",
        "42255.xml", "42256.xml", "42257.xml", "42258.xml", "42259.xml"
    ]

    # Model for sentence embeddings
    SENTENCE_MODEL_NAME = 'sentence-transformers/all-mpnet-base-v2'  # More powerful general-purpose model

    # File to save/load the processed index
    SAVED_INDEX_FILE = "letter_index.pkl"

    # XML parsing constants
    TEI_NAMESPACE = {'tei': 'http://www.tei-c.org/ns/1.0'}

    # Enhanced performance settings for historical document analysis with n-grams
    BATCH_SIZE_EMBEDDING = 4  # Reduced from 8 for better CPU performance with 2000 files
    MAX_TOKENS_FOR_GENSIM = 800  # Reduced from 1000 for faster processing
    PROGRESS_REPORT_INTERVAL = 50  # Increased from 10 for less frequent logging

    # Topic modeling quality settings for n-gram enhancement
    MIN_TOPIC_DOCUMENTS = 10  # Minimum documents needed for topic modeling
    MIN_TOKENS_PER_DOCUMENT = 5  # Minimum meaningful tokens per document
    MIN_CORPUS_TERMS = 3  # Minimum unique terms per document in corpus

    # N-gram settings
    BIGRAM_MIN_COUNT = 5  # Minimum frequency for bigrams
    BIGRAM_THRESHOLD = 10.0  # Threshold for bigram formation
    TRIGRAM_MIN_COUNT = 3  # Minimum frequency for trigrams
    TRIGRAM_THRESHOLD = 8.0  # Threshold for trigram formation

    # Phase 2: Advanced Topic Modeling Settings
    # Hierarchical Topic Modeling
    HIERARCHICAL_MIN_TOPICS = 10  # Minimum topics for hierarchical modeling
    HIERARCHICAL_MAX_TOPICS = 20  # Maximum topics at each level
    HDP_ALPHA = 1.0  # HDP concentration parameter
    HDP_GAMMA = 1.0  # HDP concentration parameter for topic-level distribution

    # Dynamic Topic Modeling
    DYNAMIC_TIME_SLICES = None  # Will be calculated based on available years
    DYNAMIC_MIN_DOCS_PER_SLICE = 50  # Minimum documents per time slice
    DTM_PASSES = 10  # Passes for dynamic topic model
    DTM_VAR_CONVERGE = 0.01  # Convergence threshold for DTM

    # Question Answering Settings
    QA_MODEL_NAME = 'deepset/roberta-large-squad2'  # More powerful QA model with SQuAD 2.0 support
    QA_MAX_ANSWER_LENGTH = 512  # Maximum length of extracted answers
    QA_MIN_SCORE_THRESHOLD = 0.0  # Removed threshold - accept all answers
    QA_TOP_K_CONTEXTS = 10  # Increased from 5 to search even more contexts

    # Hybrid Search Settings
    TFIDF_MAX_FEATURES = 5000  # Maximum features for TF-IDF vectorizer
    TFIDF_MIN_DF = 2  # Minimum document frequency for TF-IDF
    TFIDF_MAX_DF = 0.8  # Maximum document frequency for TF-IDF
    HYBRID_SEMANTIC_WEIGHT = 0.6  # Weight for semantic search in hybrid
    HYBRID_KEYWORD_WEIGHT = 0.4  # Weight for keyword search in hybrid
    HYBRID_TOP_N = 10  # Number of results to consider from each search type

    # Phase 1: Performance Optimization Settings
    MAX_WORKERS = min(8, mp.cpu_count())  # Parallel processing workers (max 8 to avoid memory issues)
    EMBEDDING_CHECKPOINT_INTERVAL = 1000  # Save embeddings every N documents
    CHECKPOINT_DIR = "checkpoints"  # Directory for saving incremental progress
    
    # Optimal batching for M4 Mac
    OPTIMAL_BATCH_SIZE_EMBEDDING = 16  # Optimized for M4 performance
    OPTIMAL_BATCH_SIZE_LDA = 2000  # Larger chunks for LDA training
    
    # File metadata tracking for incremental updates
    METADATA_FILE = "index_metadata.pkl"
    
    # Topic model sampling for development
    SAMPLE_SIZE_FOR_TOPIC_TRAINING = 2000  # Use subset for faster topic model iteration
    ENABLE_TOPIC_SAMPLING = False  # Set to True for development mode

# Create global config instance
config = Config()

# Legacy constants for backward compatibility
NUM_TOPICS = config.NUM_TOPICS
DEFAULT_UPLOADED_FILES = config.DEFAULT_UPLOADED_FILES
SENTENCE_MODEL_NAME = config.SENTENCE_MODEL_NAME
SAVED_INDEX_FILE = config.SAVED_INDEX_FILE

# Global LDA model for topic modeling (improved handling)
global_lda_model: Optional[Any] = None
global_lda_dictionary: Optional[Any] = None

# Phase 2: Global models for advanced topic modeling
global_hdp_model: Optional[Any] = None
global_hdp_dictionary: Optional[Any] = None
global_dtm_model: Optional[Any] = None
global_dtm_dictionary: Optional[Any] = None
global_time_slices: Optional[List[int]] = None

# Phase 2: Global models for QA and hybrid search
global_qa_pipeline: Optional[Any] = None

# Enhanced XML Parsing Functions
def get_element_text_content(element: ET.Element) -> str:
    """
    Recursively gets all text within an element and its children,
    replacing line breaks (<lb/>) with spaces for better flow.

    Args:
        element: XML element to extract text from

    Returns:
        Cleaned text content
    """
    if element is None:
        return ""

    text = ""
    if element.text:
        text += element.text.strip()

    for child in element:
        if child.tag.endswith('lb'):  # Handle TEI line breaks
            text += " "  # Add a space for line breaks
        text += get_element_text_content(child)
        if child.tail:
            text += " " + child.tail.strip()  # Add space before tail text

    return re.sub(r'\s+', ' ', text).strip()  # Consolidate multiple spaces

def safe_find_text(root: ET.Element, xpath: str, namespaces: Dict[str, str], default: str = "N/A") -> str:
    """
    Safely find and extract text from XML element.

    Args:
        root: Root XML element
        xpath: XPath expression
        namespaces: XML namespaces
        default: Default value if element not found

    Returns:
        Extracted text or default value
    """
    try:
        element = root.find(xpath, namespaces)
        if element is not None and element.text:
            return element.text.strip()
        return default
    except Exception as e:
        logger.warning(f"Error finding element with xpath '{xpath}': {e}")
        return default

def extract_people_from_xml(root: ET.Element, namespaces: Dict[str, str]) -> Tuple[str, str, List[str]]:
    """
    Extract sender, recipient, and all people from XML.

    Args:
        root: Root XML element
        namespaces: XML namespaces

    Returns:
        Tuple of (sender, recipient, list_of_all_people)
    """
    people_in_doc = [] 
    sender = "Unknown"
    recipient = "Unknown" 

    try:
        # Try to get creator from metadata first
        creator_element = root.find('.//tei:xenoData/tei:iiifMetadata/tei:element[@label="Creator"]/tei:value', namespaces)
        if creator_element is not None and creator_element.text:
            sender = creator_element.text.strip()
            if sender not in people_in_doc: 
                people_in_doc.append(sender)

        # Extract from listPerson
        for person_element in root.findall('.//tei:listPerson/tei:person', namespaces):
            pers_name_element = person_element.find('.//tei:persName', namespaces)
            if pers_name_element is not None and pers_name_element.text:
                person_name = pers_name_element.text.strip()
                if person_name not in people_in_doc:
                     people_in_doc.append(person_name)

                # Simple heuristic for recipient (looking for common names)
                if any(name in person_name.lower() for name in ['pettus', 'stone', 'governor']):
                    recipient = person_name
                elif sender == "Unknown" and not any(name in person_name.lower() for name in ['pettus', 'stone']):
                    sender = person_name

    except Exception as e:
        logger.warning(f"Error extracting people from XML: {e}")

    return sender, recipient, list(set(people_in_doc))

def extract_places_from_xml(root: ET.Element, namespaces: Dict[str, str]) -> List[str]:
    """
    Extract places from XML.

    Args:
        root: Root XML element
        namespaces: XML namespaces

    Returns:
        List of unique places
    """
    places_in_doc = []

    try:
        # Extract from listPlace
        for place_element in root.findall('.//tei:listPlace/tei:place', namespaces):
            place_name_element = place_element.find('.//tei:placeName', namespaces)
            if place_name_element is not None and place_name_element.text:
                 place_text = place_name_element.text.strip()
                 if place_text not in places_in_doc:
                     places_in_doc.append(place_text)

        # Extract from metadata
        geo_loc_element = root.find('.//tei:xenoData/tei:iiifMetadata/tei:element[@label="Geographic location"]/tei:value', namespaces)
        if geo_loc_element is not None and geo_loc_element.text:
            meta_place = geo_loc_element.text.strip()
            if meta_place not in places_in_doc: 
                places_in_doc.append(meta_place)

    except Exception as e:
        logger.warning(f"Error extracting places from XML: {e}")

    return list(set(places_in_doc))

def extract_date_info(root: ET.Element, namespaces: Dict[str, str], title: str) -> Tuple[str, Optional[int]]:
    """
    Extract date information from XML.

    Args:
        root: Root XML element
        namespaces: XML namespaces
        title: Document title (fallback for date extraction)

    Returns:
        Tuple of (main_date_string, year_integer)
    """
    main_date = "N/A"
    year = None

    try:
        # Try metadata first
        date_element = root.find('.//tei:xenoData/tei:iiifMetadata/tei:element[@label="Date"]/tei:value', namespaces)
        if date_element is not None and date_element.text:
            main_date = date_element.text.strip()
        else:
            # Fallback to title extraction
            date_match = re.search(r'(\w+\s+\d{1,2},\s+\d{4}|\d{4}-\d{2}-\d{2})', title)
            if date_match:
                main_date = date_match.group(0)

        # Extract year
        year_match = re.search(r'\b(\d{4})\b', main_date)
        if year_match:
            year = int(year_match.group(1))

    except Exception as e:
        logger.warning(f"Error extracting date info: {e}")

    return main_date, year

def parse_letter_xml(xml_file_path: str) -> Optional[Dict[str, Any]]:
    """
    Enhanced XML parsing with better error handling and logging.

    Args:
        xml_file_path: Path to the XML file

    Returns:
        Dictionary containing extracted information, or None if parsing fails
    """
    try:
        if not os.path.exists(xml_file_path):
            logger.error(f"File not found: {xml_file_path}")
            return None

        if not os.access(xml_file_path, os.R_OK):
            logger.error(f"File not readable: {xml_file_path}")
            return None

        tree = ET.parse(xml_file_path)
        root = tree.getroot()

        extracted_data = {
            'file_name': os.path.basename(xml_file_path),
            'file_path': xml_file_path,
            'parsing_timestamp': time.time()
        }

        namespaces = config.TEI_NAMESPACE

        # Extract title
        title_element = root.find('.//tei:titleStmt/tei:title/tei:title[@type="main"]', namespaces)
        extracted_data['title'] = title_element.text.strip() if title_element is not None and title_element.text else "N/A"

        # Extract date information
        main_date, year = extract_date_info(root, namespaces, extracted_data['title'])
        extracted_data['main_date'] = main_date
        extracted_data['year'] = year

        # Extract people information
        sender, recipient, people = extract_people_from_xml(root, namespaces)
        extracted_data['sender'] = sender
        extracted_data['recipient'] = recipient
        extracted_data['people'] = people

        # Extract places
        places = extract_places_from_xml(root, namespaces)
        extracted_data['places'] = places
        
        # Geocode places if any exist
        if places:
            try:
                geocoded_places = geocode_locations(places)
                extracted_data['geocoded_places'] = geocoded_places
                logger.debug(f"Geocoded {len([g for g in geocoded_places if g.get('geocoded', False)])}/{len(places)} places for {extracted_data['file_name']}")
            except Exception as e:
                logger.warning(f"Geocoding failed for {extracted_data['file_name']}: {e}")
                extracted_data['geocoded_places'] = []
        else:
            extracted_data['geocoded_places'] = []

        # Extract description
        extracted_data['description'] = safe_find_text(
            root,
            './/tei:xenoData/tei:iiifMetadata/tei:element[@label="Description"]/tei:value',
            namespaces
        )

        # Extract full text
        body_element = root.find('.//tei:text/tei:body', namespaces)
        full_text_paragraphs = []
        if body_element is not None:
            for p_element in body_element.findall('.//tei:p', namespaces):
                 paragraph_text = get_element_text_content(p_element)
                 if paragraph_text: 
                     full_text_paragraphs.append(paragraph_text)

        extracted_data['full_text'] = "\n\n".join(full_text_paragraphs) if full_text_paragraphs else "N/A"
        
        # Initialize embedding and topic fields
        extracted_data['embedding'] = None
        extracted_data['topic_id'] = None
        extracted_data['topic_score'] = None
        extracted_data['topic_terms'] = []

        logger.debug(f"Successfully parsed {xml_file_path}")
        return extracted_data

    except ET.ParseError as e:
        logger.error(f"XML parsing error in '{xml_file_path}': {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error processing '{xml_file_path}': {e}")
        return None

def preprocess_for_gensim(text_or_doc: Any, nlp_for_str_processing=None, phrases_model=None) -> List[str]:
    """
    Enhanced text preprocessing for Gensim with n-gram support for historical content.
    Detects meaningful multi-word expressions and produces more mature topics.
    Accepts raw text string or a pre-processed spaCy Doc object.

    Args:
        text_or_doc: Input text string or spaCy Doc object to preprocess
        nlp_for_str_processing: spaCy NLP model (only used if text_or_doc is a string)
        phrases_model: Trained gensim Phrases model for detecting n-grams (optional)

    Returns:
        List of cleaned, meaningful tokens including n-grams
    """
    if not text_or_doc:
        return []
    if isinstance(text_or_doc, str) and text_or_doc.strip() == "N/A":
        return []

    # Custom stop words for historical letters - removing generic administrative terms
    historical_stop_words = {
        'sir', 'dear', 'yours', 'very', 'much', 'well', 'good', 'great', 'little', 'large',
        'time', 'day', 'week', 'month', 'year', 'morning', 'evening', 'night',
        'state', 'county', 'city', 'town', 'place', 'home', 'house', 'office',
        'letter', 'write', 'written', 'writing', 'received', 'send', 'sent',
        'excellency', 'honor', 'honourable', 'respectfully', 'obedient', 'servant',
        'sincerely', 'truly', 'faithfully', 'humble', 'most', 'very', 'quite',
        'please', 'would', 'could', 'should', 'will', 'shall', 'may', 'might',
        'said', 'says', 'tell', 'told', 'ask', 'asked', 'give', 'given', 'take', 'taken',
        'come', 'came', 'go', 'went', 'see', 'seen', 'know', 'knew', 'think', 'thought',
        'way', 'ways', 'thing', 'things', 'matter', 'matters', 'case', 'cases',
        'hope', 'wish', 'want', 'need', 'like', 'find', 'found', 'look', 'looking',
        'make', 'made', 'doing', 'done', 'work', 'working', 'worked',
        'mississippi', 'miss', 'jackson', 'vicksburg',  # Remove state-specific common terms
        'governor', 'general', 'colonel', 'captain', 'major',  # Generic titles
        'mr', 'mrs', 'dr', 'prof', 'hon'  # Generic honorifics
    }

    initial_tokens = []
    try:
        doc_to_process = None
        if SPACY_AVAILABLE and spacy.tokens and isinstance(text_or_doc, spacy.tokens.Doc):
            doc_to_process = text_or_doc
        elif SPACY_AVAILABLE and nlp_for_str_processing and isinstance(text_or_doc, str):
            doc_to_process = nlp_for_str_processing(text_or_doc)
        
        if doc_to_process:  # Enhanced spaCy processing
            for token in doc_to_process:
                if (token.is_stop or token.is_punct or token.like_num or
                        token.is_space or len(token.lemma_) < 3):
                    continue
                if token.pos_ in ['NOUN', 'PROPN', 'ADJ', 'VERB']:
                    lemma = token.lemma_.lower()
                    if lemma in historical_stop_words:
                        continue
                    if lemma.isalpha():
                        if token.pos_ in ['PROPN', 'NOUN']:
                            initial_tokens.append(lemma)
                        elif token.pos_ in ['ADJ', 'VERB'] and len(lemma) > 4:
                            initial_tokens.append(lemma)
        elif isinstance(text_or_doc, str): # Fallback: text_or_doc is a string, and no/unusable nlp
            text_for_fallback = re.sub(r'[^\w\s]', ' ', text_or_doc.lower())
            raw_tokens = re.findall(r'\b[a-z]{4,}\b', text_for_fallback)
            initial_tokens = [t for t in raw_tokens if t not in historical_stop_words]
        else:
            logger.warning(f"Unexpected input type to preprocess_for_gensim: {type(text_or_doc)}. Returning empty list.")
            return []

        # Apply n-gram detection if phrases model is available
        if phrases_model and initial_tokens:
            if callable(phrases_model):
                processed_tokens = phrases_model(initial_tokens)
            else:
                processed_tokens = phrases_model[initial_tokens]
        else:
            processed_tokens = initial_tokens

        # Remove duplicates while preserving order
        seen = set()
        unique_tokens = []
        for token in processed_tokens:
            if token not in seen:
                seen.add(token)
                unique_tokens.append(token)

        return unique_tokens[:config.MAX_TOKENS_FOR_GENSIM]

    except Exception as e:
        logger.warning(f"Error in enhanced text preprocessing with n-grams: {e} for input type {type(text_or_doc)}")
        # Simple fallback
        try:
            if isinstance(text_or_doc, str):
                fallback_tokens = re.findall(r"\b[a-zA-Z]{4,}\b", text_or_doc.lower())
                return [t for t in fallback_tokens if t not in historical_stop_words][:config.MAX_TOKENS_FOR_GENSIM]
        except Exception:
            pass # Fall through to return empty list
        return []

def generate_embeddings_batch(texts: List[str], sentence_model, 
                             checkpoint_path: Optional[str] = None,
                             start_idx: int = 0) -> List[Any]:
    """
    Generate embeddings for a list of texts with enhanced performance and checkpointing.

    Args:
        texts: List of text strings to embed
        sentence_model: Loaded sentence transformer model
        checkpoint_path: Path to save/load checkpoint data
        start_idx: Starting index for resuming from checkpoint

    Returns:
        List of embedding vectors
    """
    if not SENTENCE_TRANSFORMERS_AVAILABLE or not texts:
        logger.warning("Sentence transformers not available or no texts provided")
        return []

    total_texts = len(texts)
    logger.info(f"Generating embeddings for {total_texts} texts starting from index {start_idx}")
    
    # Load existing embeddings if checkpoint exists
    embeddings = []
    if checkpoint_path and os.path.exists(checkpoint_path):
        try:
            with open(checkpoint_path, 'rb') as f:
                checkpoint_data = pickle.load(f)
                embeddings = checkpoint_data.get('embeddings', [])
                start_idx = len(embeddings)
                logger.info(f"Loaded checkpoint with {len(embeddings)} existing embeddings")
        except Exception as e:
            logger.error(f"Error loading checkpoint: {e}")
            embeddings = []
            start_idx = 0
    
    # Use optimized batch size
    batch_size = config.OPTIMAL_BATCH_SIZE_EMBEDDING
    
    # Process remaining texts in batches
    for i in range(start_idx, total_texts, batch_size):
        batch_texts = texts[i:i + batch_size]
        actual_batch_size = len(batch_texts)
        
        try:
            logger.info(f"Processing embedding batch {i//batch_size + 1}: texts {i+1}-{i+actual_batch_size} of {total_texts}")
            
            # Generate embeddings for this batch
                batch_embeddings = sentence_model.encode(
                batch_texts,
                show_progress_bar=False,  # We handle progress ourselves
                batch_size=actual_batch_size,
                convert_to_tensor=False,
                normalize_embeddings=True  # Normalize for better similarity calculation
            )
            
            # Convert to list if numpy array
            if hasattr(batch_embeddings, 'tolist'):
                batch_embeddings = batch_embeddings.tolist()

                embeddings.extend(batch_embeddings)

            # Save checkpoint periodically
            if checkpoint_path and (i + batch_size) % config.EMBEDDING_CHECKPOINT_INTERVAL == 0:
                try:
                    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
                    checkpoint_data = {
                        'embeddings': embeddings,
                        'processed_count': len(embeddings),
                        'total_count': total_texts,
                        'timestamp': datetime.now().isoformat()
                    }
                    with open(checkpoint_path, 'wb') as f:
                        pickle.dump(checkpoint_data, f)
                    logger.info(f"Checkpoint saved: {len(embeddings)}/{total_texts} embeddings")
            except Exception as e:
                    logger.error(f"Error saving checkpoint: {e}")
            
        except Exception as e:
            logger.error(f"Error processing batch {i//batch_size + 1}: {e}")
            # Fill with zero embeddings for failed batch
            embedding_dim = getattr(sentence_model, 'get_sentence_embedding_dimension', lambda: 768)()
            for _ in range(actual_batch_size):
                embeddings.append([0.0] * embedding_dim)
    
    # Save final checkpoint
    if checkpoint_path:
        try:
            os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
            checkpoint_data = {
                'embeddings': embeddings,
                'processed_count': len(embeddings),
                'total_count': total_texts,
                'timestamp': datetime.now().isoformat(),
                'completed': True
            }
            with open(checkpoint_path, 'wb') as f:
                pickle.dump(checkpoint_data, f)
            logger.info(f"Final embedding checkpoint saved: {len(embeddings)} embeddings")
    except Exception as e:
            logger.error(f"Error saving final checkpoint: {e}")
    
    logger.info(f"Embedding generation completed: {len(embeddings)} embeddings generated")
    return embeddings

def load_embeddings_checkpoint(checkpoint_path: str) -> Tuple[List[Any], int]:
    """
    Load embeddings from checkpoint file.

    Returns:
        Tuple of (embeddings_list, processed_count)
    """
    try:
        with open(checkpoint_path, 'rb') as f:
            checkpoint_data = pickle.load(f)
            embeddings = checkpoint_data.get('embeddings', [])
            processed_count = checkpoint_data.get('processed_count', len(embeddings))
            logger.info(f"Loaded embeddings checkpoint: {processed_count} embeddings")
            return embeddings, processed_count
    except Exception as e:
        logger.error(f"Error loading embeddings checkpoint: {e}")
        return [], 0

def train_lda_model(working_index: List[Dict[str, Any]], spacy_nlp_model, num_topics: int = 25) -> Tuple[Any, Any]:
    """
    Modern fast topic modeling using embedding clustering.
    No more slow preprocessing - uses embeddings we already computed!
    """
    logger.info(f"Starting modern topic modeling on {len(working_index)} documents...")
    
    # Use modern embedding-based approach if available
    if MODERN_TOPICS_AVAILABLE:
        try:
            model, results = replace_slow_topic_modeling(working_index, config)
            logger.info(f"Fast topic modeling completed - found {model.num_topics} topics")
            return model, None  # No corpus needed for modern approach
            except Exception as e:
            logger.error(f"Modern topic modeling failed: {e}")
            logger.info("Falling back to simple keyword-based topics...")
    
    # Fallback: Simple keyword extraction if modern approach fails
    logger.info("Using simple keyword-based topic assignment...")
    
    # Quick keyword-based topic assignment
    topic_keywords = {
        0: ['war', 'battle', 'fight', 'army', 'soldier'],
        1: ['family', 'home', 'wife', 'child', 'love'],
        2: ['supply', 'food', 'money', 'need', 'send'],
        3: ['government', 'president', 'politics', 'union'],
        4: ['health', 'sick', 'doctor', 'medicine', 'hospital'],
        5: ['travel', 'journey', 'road', 'march', 'camp'],
        6: ['news', 'report', 'hear', 'tell', 'information'],
        7: ['business', 'work', 'trade', 'merchant', 'commerce'],
        8: ['religion', 'god', 'church', 'pray', 'faith'],
        9: ['weather', 'rain', 'cold', 'hot', 'season']
    }
    
    # Assign topics based on keyword matching
    for doc in working_index:
        text = doc.get('full_text', '').lower()
        best_topic = 0
        best_score = 0
        
        for topic_id, keywords in topic_keywords.items():
            score = sum(1 for kw in keywords if kw in text)
            if score > best_score:
                best_score = score
                best_topic = topic_id
        
        doc['topic_id'] = best_topic
        doc['lda_tokens'] = []  # Empty for compatibility
    
    # Create simple model object
    class SimpleTopicModel:
        def __init__(self):
            self.num_topics = len(topic_keywords)
            self.topics = topic_keywords
            
        def print_topics(self, num_topics=10):
            for topic_id, keywords in list(self.topics.items())[:num_topics]:
                print(f"Topic {topic_id}: {', '.join(keywords[:5])}")
    
    logger.info(f"Simple topic assignment completed with {len(topic_keywords)} topics")
    return SimpleTopicModel(), None

def assign_topics_to_documents(index: List[Dict[str, Any]], lda_model, lda_dictionary) -> None:
    """
    Enhanced topic assignment with detailed analysis for historical documents.

    Args:
        index: List of document dictionaries
        lda_model: Trained LDA model
        lda_dictionary: LDA dictionary
    """
    if not lda_model or not lda_dictionary:
        logger.warning("No LDA model available for topic assignment")
        return

    try:
        logger.info("Assigning enhanced topics to documents...")
        assigned_count = 0

        for item in index:
            try:
                tokens = item.get("lda_tokens", [])
                if not tokens:
                    continue

                # Create bag-of-words representation
                bow = lda_dictionary.doc2bow(tokens)
                if not bow:  # Skip if no valid tokens
                    continue

                # Get topic distribution with enhanced analysis
                topic_probs = lda_model.get_document_topics(
                    bow,
                    minimum_probability=0.05  # Only consider topics with at least 5% probability
                )

                if topic_probs:
                    # Sort topics by probability (highest first)
                    topic_probs_sorted = sorted(topic_probs, key=lambda x: x[1], reverse=True)

                    # Get the primary topic (highest probability)
                    primary_topic, primary_score = topic_probs_sorted[0]

                    item["topic_id"] = int(primary_topic)
                    item["topic_score"] = float(primary_score)

                    # Get enhanced topic terms (more terms for better understanding)
                    try:
                        topic_terms = lda_model.show_topic(primary_topic, topn=8)  # More terms
                        item["topic_terms"] = [term for term, _ in topic_terms]

                        # Add weighted topic terms (with probabilities)
                        weighted_terms = [(term, round(prob, 3)) for term, prob in topic_terms[:5]]
                        item["topic_terms_weighted"] = weighted_terms

                    except Exception as e:
                        logger.warning(f"Error getting topic terms for topic {primary_topic}: {e}")
                        item["topic_terms"] = []
                        item["topic_terms_weighted"] = []

                    # Store additional topic information for richer analysis
                    if len(topic_probs_sorted) > 1:
                        # Secondary topic (if exists and significant)
                        secondary_topic, secondary_score = topic_probs_sorted[1]
                        if secondary_score >= 0.15:  # Only if secondary topic is significant
                            item["secondary_topic_id"] = int(secondary_topic)
                            item["secondary_topic_score"] = float(secondary_score)

                        # Topic diversity (how focused vs. diverse the document is)
                        total_prob = sum(score for _, score in topic_probs_sorted)
                        topic_diversity = 1 - (primary_score / total_prob) if total_prob > 0 else 0
                        item["topic_diversity"] = round(float(topic_diversity), 3)

                    assigned_count += 1

            except Exception as e:
                logger.warning(f"Error assigning topic to document {item.get('file_name', 'unknown')}: {e}")
                continue

        logger.info(f"Enhanced topic assignment completed: {assigned_count} documents assigned topics")

        # Log some sample topic assignments for quality check
        try:
            sample_size = min(3, assigned_count)
            logger.info(f"Sample topic assignments (showing {sample_size}):")
            assigned_docs = [item for item in index if item.get('topic_id') is not None]
            for i, doc in enumerate(assigned_docs[:sample_size]):
                terms = ", ".join(doc.get('topic_terms', [])[:5])
                logger.info(f"  Doc '{doc.get('file_name', 'unknown')}': Topic {doc.get('topic_id')} "
                            f"(score: {doc.get('topic_score', 0):.3f}) - {terms}")
        except Exception as e:
            logger.warning(f"Error logging sample assignments: {e}")

    except Exception as e:
        logger.error(f"Error in enhanced topic assignment process: {e}")

def parse_xml_files_parallel(xml_files_to_process: List[str]) -> List[Dict[str, Any]]:
    """
    Parse XML files in parallel using ProcessPoolExecutor.

    Args:
        xml_files_to_process: List of XML file paths to parse

    Returns:
        List of successfully parsed document dictionaries
    """
    logger.info(f"Starting parallel XML parsing with {config.MAX_WORKERS} workers")
    parsed_docs = []
    
    try:
        with ProcessPoolExecutor(max_workers=config.MAX_WORKERS) as executor:
            # Submit all parsing tasks
            future_to_file = {
                executor.submit(parse_letter_xml, xml_file): xml_file 
                for xml_file in xml_files_to_process
            }
            
            # Collect results as they complete
            for i, future in enumerate(as_completed(future_to_file)):
                xml_file = future_to_file[future]
                try:
                    result = future.result()
                    if result is not None:
                        parsed_docs.append(result)
                    
                    # Progress reporting
                    if (i + 1) % config.PROGRESS_REPORT_INTERVAL == 0:
                        logger.info(f"Parsed {i + 1}/{len(xml_files_to_process)} files")
                        
                except Exception as e:
                    logger.error(f"Error parsing {xml_file}: {e}")
                continue

        logger.info(f"Parallel parsing completed: {len(parsed_docs)}/{len(xml_files_to_process)} files successfully parsed")
        return parsed_docs
        
    except Exception as e:
        logger.error(f"Error in parallel XML parsing: {e}")
        # Fallback to sequential parsing
        logger.info("Falling back to sequential parsing...")
        return parse_xml_files_sequential(xml_files_to_process)

def parse_xml_files_sequential(xml_files_to_process: List[str]) -> List[Dict[str, Any]]:
    """
    Fallback sequential XML parsing function.
    
    Args:
        xml_files_to_process: List of XML file paths to parse
        
    Returns:
        List of successfully parsed document dictionaries
    """
    logger.info("Starting sequential XML parsing")
    parsed_docs = []
    
    for i, xml_file in enumerate(xml_files_to_process):
        try:
            result = parse_letter_xml(xml_file)
            if result is not None:
                parsed_docs.append(result)

            # Progress reporting
            if (i + 1) % config.PROGRESS_REPORT_INTERVAL == 0:
                logger.info(f"Parsed {i + 1}/{len(xml_files_to_process)} files")

        except Exception as e:
            logger.error(f"Error parsing {xml_file}: {e}")
            continue

    logger.info(f"Sequential parsing completed: {len(parsed_docs)}/{len(xml_files_to_process)} files successfully parsed")
    return parsed_docs

def get_file_metadata(file_path: str) -> Dict[str, Any]:
    """
    Get metadata for a file for incremental indexing.
    
    Args:
        file_path: Path to the file
        
    Returns:
        Dictionary containing file metadata
    """
    try:
        stat = os.stat(file_path)
        
        # Calculate file hash for change detection
        with open(file_path, 'rb') as f:
            file_hash = hashlib.sha256(f.read()).hexdigest()
            
        return {
            'size': stat.st_size,
            'modified': stat.st_mtime,
            'hash': file_hash,
            'path': file_path
        }
    except Exception as e:
        logger.error(f"Error getting metadata for {file_path}: {e}")
        return {'path': file_path, 'error': str(e)}

def save_metadata(file_metadata: Dict[str, Dict[str, Any]], metadata_path: str) -> None:
    """
    Save file metadata for incremental indexing.
    
    Args:
        file_metadata: Dictionary of file path -> metadata mappings
        metadata_path: Path to save the metadata file
    """
    try:
        with open(metadata_path, 'wb') as f:
            pickle.dump(file_metadata, f)
        logger.info(f"Saved metadata for {len(file_metadata)} files to {metadata_path}")
        except Exception as e:
        logger.error(f"Error saving metadata: {e}")

def load_metadata(metadata_path: str) -> Dict[str, Dict[str, Any]]:
    """
    Load file metadata for incremental indexing.
    
    Args:
        metadata_path: Path to the metadata file
        
    Returns:
        Dictionary of file path -> metadata mappings
    """
    try:
        if os.path.exists(metadata_path):
            with open(metadata_path, 'rb') as f:
                metadata = pickle.load(f)
            logger.info(f"Loaded metadata for {len(metadata)} files from {metadata_path}")
            return metadata
    else:
            logger.info("No existing metadata file found")
            return {}
    except Exception as e:
        logger.error(f"Error loading metadata: {e}")
        return {}

def _create_and_save_index(xml_files_to_process: List[str], sentence_model, spacy_nlp_model, 
                          index_path: str, incremental: bool = False) -> List[Dict[str, Any]]:
    """
    Enhanced index creation with parallel processing, checkpointing, and incremental updates.
    
    Args:
        xml_files_to_process: List of XML file paths to process
        sentence_model: Loaded sentence transformer model
        spacy_nlp_model: Loaded spaCy model
        index_path: Path to save the index
        incremental: Whether this is an incremental update
        
    Returns:
        List of document dictionaries with embeddings and topics
    """
    start_time = time.time()
    logger.info(f"Starting {'incremental' if incremental else 'full'} index creation with {len(xml_files_to_process)} files")
    
    # Initialize model variables at the start
    lda_model = None
    lda_dictionary = None
    
    # Create checkpoint directory
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    
    # Step 1: Parallel XML parsing
    logger.info("Phase 1: Parallel XML parsing...")
    parsed_docs = parse_xml_files_parallel(xml_files_to_process)
    
    if not parsed_docs:
        logger.error("No documents were successfully parsed")
        return []
    
    logger.info(f"Successfully parsed {len(parsed_docs)} documents")
    
    # Step 2: Enhanced embedding generation with checkpointing
    logger.info("Phase 2: Generating embeddings with checkpointing...")
    texts_for_embedding = [
        f"{doc.get('title', '')} {doc.get('description', '')} {doc.get('full_text', '')}".strip()
        for doc in parsed_docs
    ]
    
    # Setup checkpoint path for embeddings
    embedding_checkpoint_path = os.path.join(config.CHECKPOINT_DIR, "embeddings_checkpoint.pkl")
    
    try:
        embeddings = generate_embeddings_batch(
            texts_for_embedding, 
            sentence_model,
            checkpoint_path=embedding_checkpoint_path
        )
        
        # Attach embeddings to documents
        for i, doc in enumerate(parsed_docs):
            if i < len(embeddings):
                doc['embedding'] = embeddings[i]
            else:
                # Fallback for missing embeddings
                embedding_dim = getattr(sentence_model, 'get_sentence_embedding_dimension', lambda: 768)()
                doc['embedding'] = [0.0] * embedding_dim
                
        successful_embeddings = sum(1 for doc in parsed_docs if doc.get('embedding') is not None)
        logger.info(f"Embeddings attached: {successful_embeddings}/{len(parsed_docs)} documents")
        
        except Exception as e:
        logger.error(f"Error in embedding generation: {e}")
        # Continue without embeddings for now
        for doc in parsed_docs:
            doc['embedding'] = None
    
    # Step 3: Enhanced topic modeling
    logger.info("Phase 3: Topic modeling...")
    try:
        # Train LDA model with potential sampling for development
        lda_model, lda_dictionary = train_lda_model(parsed_docs, spacy_nlp_model)
        
        if lda_model and lda_dictionary:
            logger.info("Assigning topics to documents...")
            assign_topics_to_documents(parsed_docs, lda_model, lda_dictionary)
            
            # Store models in global variables
            global global_lda_model, global_lda_dictionary
            global_lda_model = lda_model
            global_lda_dictionary = lda_dictionary
            
        else:
            logger.warning("LDA model training failed - continuing without topic assignments")
            
        except Exception as e:
        logger.error(f"Error in topic modeling: {e}")
    
    # Step 4: Advanced topic modeling (if enabled)
    if ADVANCED_GENSIM_AVAILABLE and len(parsed_docs) >= config.HIERARCHICAL_MIN_TOPICS:
        try:
            logger.info("Phase 4a: Hierarchical topic modeling...")
            hdp_model, hdp_dictionary = train_hierarchical_topic_model(parsed_docs, spacy_nlp_model)
            
            if hdp_model:
                global global_hdp_model, global_hdp_dictionary
                global_hdp_model = hdp_model
                global_hdp_dictionary = hdp_dictionary
                logger.info("Hierarchical topic model trained successfully")
                
        except Exception as e:
            logger.error(f"Error in hierarchical topic modeling: {e}")
        
        try:
            logger.info("Phase 4b: Dynamic topic modeling...")
            dtm_model, dtm_dictionary, time_slices = train_dynamic_topic_model(parsed_docs)
            
            if dtm_model:
                global global_dtm_model, global_dtm_dictionary, global_time_slices
                global_dtm_model = dtm_model
                global_dtm_dictionary = dtm_dictionary 
                global_time_slices = time_slices
                logger.info("Dynamic topic model trained successfully")
                
        except Exception as e:
            logger.error(f"Error in dynamic topic modeling: {e}")
    
    # Step 5: Create comprehensive index with metadata
    logger.info("Phase 5: Creating comprehensive index...")
    
    # Add processing metadata
    processing_metadata = {
        'version': '3.0',
        'created_at': datetime.now().isoformat(),
        'total_documents': len(parsed_docs),
        'successful_embeddings': sum(1 for doc in parsed_docs if doc.get('embedding') is not None),
        'documents_with_topics': sum(1 for doc in parsed_docs if doc.get('dominant_topic') is not None),
        'topics_available': lda_model.num_topics if lda_model else 0,
        'processing_time_seconds': time.time() - start_time,
        'incremental_update': incremental,
        'files_processed': len(xml_files_to_process)
    }
    
    # Create the comprehensive index
    comprehensive_index = {
        'documents': parsed_docs,
        'metadata': processing_metadata,
        'lda_model': lda_model,
        'lda_dictionary': lda_dictionary,
        'hdp_model': global_hdp_model,
        'hdp_dictionary': global_hdp_dictionary,
        'dtm_model': global_dtm_model,
        'dtm_dictionary': global_dtm_dictionary,
        'time_slices': global_time_slices
    }
    
    # Step 6: Save index and metadata
    logger.info("Phase 6: Saving index and metadata...")
    try:
        with open(index_path, 'wb') as f:
            pickle.dump(comprehensive_index, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        # Update file metadata for incremental updates
        metadata_path = config.METADATA_FILE
        file_metadata = {}
        
        for file_path in xml_files_to_process:
            file_metadata[file_path] = get_file_metadata(file_path)
        
        save_metadata(file_metadata, metadata_path)
        
        # Clean up checkpoint files
        try:
            if os.path.exists(embedding_checkpoint_path):
                os.remove(embedding_checkpoint_path)
        except:
            pass
        
        elapsed_time = time.time() - start_time
        logger.info(f"Index creation completed in {elapsed_time:.2f} seconds")
        logger.info(f"Index stats: {processing_metadata}")
        
        return parsed_docs
        
    except Exception as e:
        logger.error(f"Error saving index: {e}")
        return parsed_docs

def load_or_create_index(xml_files_to_process: List[str], sentence_model, spacy_nlp_model,
                         index_path: str, force_reindex: bool = False, incremental: bool = True) -> List[Dict[str, Any]]:
    """
    Enhanced index loading/creation with incremental update support.

    Args:
        xml_files_to_process: List of XML file paths to process
        sentence_model: Loaded sentence transformer model
        spacy_nlp_model: Loaded spaCy model
        index_path: Path to the index file
        force_reindex: Force complete reindexing
        incremental: Enable incremental updates (default True)

    Returns:
        List of document dictionaries
    """
    global global_lda_model, global_lda_dictionary, global_hdp_model, global_hdp_dictionary
    global global_dtm_model, global_dtm_dictionary, global_time_slices
    
    logger.info(f"Loading or creating index at '{index_path}'")
    logger.info(f"Mode: {'Force reindex' if force_reindex else 'Incremental' if incremental else 'Standard'}")
    
    # Check if index exists and force_reindex is not set
    if os.path.exists(index_path) and not force_reindex:
        try:
            # Get index file info
            index_stat = os.stat(index_path)
            index_size_mb = index_stat.st_size / (1024 * 1024)
            index_mtime = time.ctime(index_stat.st_mtime)
            
            logger.info(f"Loading existing index from '{index_path}'...")
            logger.info(f"Index file: {index_size_mb:.1f} MB, modified: {index_mtime}")
            
            # Load existing index
            with open(index_path, 'rb') as f:
                saved_data = pickle.load(f)
            
            # Handle different index versions
            if isinstance(saved_data, dict):
                # Version 2.0+ format
                if 'version' in saved_data.get('metadata', saved_data):
                    version = saved_data.get('metadata', saved_data).get('version', '2.0')
                    logger.info(f"Loaded index version {version}")
                    
                    if version == '3.0':
                        # New format with comprehensive metadata
                        documents = saved_data['documents']
                        metadata = saved_data['metadata']
                        
                        # Load all models
                        global_lda_model = saved_data.get('lda_model')
                        global_lda_dictionary = saved_data.get('lda_dictionary')
                        global_hdp_model = saved_data.get('hdp_model')
                        global_hdp_dictionary = saved_data.get('hdp_dictionary')
                        global_dtm_model = saved_data.get('dtm_model')
                        global_dtm_dictionary = saved_data.get('dtm_dictionary')
                        global_time_slices = saved_data.get('time_slices')
                        
                        logger.info(f"Index stats: {metadata}")
                        
                    else:
                        # Version 2.0 format (legacy)
                        documents = saved_data.get('index', [])
                        metadata = saved_data.get('stats', {})
                        
                        # Load models from legacy format
                        global_lda_model = saved_data.get('lda_model')
                        global_lda_dictionary = saved_data.get('lda_dictionary')
                        global_hdp_model = saved_data.get('hdp_model')
                        global_hdp_dictionary = saved_data.get('hdp_dictionary')
                        global_dtm_model = saved_data.get('dtm_model')
                        global_dtm_dictionary = saved_data.get('dtm_dictionary')
                        global_time_slices = saved_data.get('time_slices')
                        
                        logger.info(f"Index stats: {metadata}")
                else:
                    # Very old format
                    documents = saved_data.get('index', saved_data)
                    logger.info("Loaded legacy index format")
            else:
                # Ancient format - just a list
                documents = saved_data if isinstance(saved_data, list) else []
                logger.info("Loaded very old index format")
            
            # Log model loading status
            if global_lda_model:
                logger.info(f"LDA model loaded with {global_lda_model.num_topics} topics")
                if global_hdp_model:
                    try:
                    active_topics = len([t for t in global_hdp_model.show_topics(num_topics=-1, formatted=False)])
                        logger.info(f"HDP model loaded with {active_topics} active topics (sample)")
                except:
                    logger.info("HDP model loaded")
            if global_dtm_model:
                logger.info("DTM model loaded")
            
            # Check for incremental updates if enabled
            if incremental and xml_files_to_process:
                logger.info("Checking for incremental updates...")
                metadata_path = config.METADATA_FILE
                new_files, changed_files = identify_changed_files(xml_files_to_process, metadata_path)
                
                files_to_update = new_files + changed_files
                
                if files_to_update:
                    logger.info(f"Found {len(files_to_update)} files needing updates ({len(new_files)} new, {len(changed_files)} changed)")
                    
                    # Process incremental updates
                    updated_docs = _create_and_save_index(
                        files_to_update, 
                        sentence_model, 
                        spacy_nlp_model, 
                        index_path + ".incremental",
                        incremental=True
                    )
                    
                    if updated_docs:
                        # Merge with existing documents
                        existing_files = {doc.get('file_name'): doc for doc in documents}
                        
                        # Update/add new documents
                        for new_doc in updated_docs:
                            file_name = new_doc.get('file_name')
                            if file_name:
                                existing_files[file_name] = new_doc
                        
                        # Convert back to list
                        documents = list(existing_files.values())
                        
                        # Save updated index
                        logger.info("Saving merged incremental index...")
                        _create_and_save_index(
                            [],  # No files to process, just save existing
                            sentence_model,
                            spacy_nlp_model,
                            index_path,
                            incremental=True
                        )
                        
                        # Clean up temporary incremental file
                        try:
                            os.remove(index_path + ".incremental")
                        except:
                            pass
                        
                        logger.info(f"Incremental update completed: {len(documents)} total documents")
                else:
                    logger.info("No files need updating - using existing index")
            
            logger.info(f"Successfully loaded {len(documents)} items from saved index")
            
            # Verify embeddings
            embeddings_count = sum(1 for doc in documents if doc.get('embedding') is not None)
            logger.info(f"Embeddings available: {embeddings_count}/{len(documents)}")
            
            return documents
            
        except Exception as e:
            logger.error(f"Error loading existing index: {e}")
            logger.info("Will create new index instead")
    
    # Create new index if no existing index or force_reindex is True
    if not xml_files_to_process:
        logger.warning("No XML files provided for index creation")
        return []
    
    logger.info(f"Creating new index from {len(xml_files_to_process)} XML files...")
    return _create_and_save_index(xml_files_to_process, sentence_model, spacy_nlp_model, index_path, incremental=False)

def identify_changed_files(xml_files_to_process: List[str], metadata_path: str) -> Tuple[List[str], List[str]]:
    """
    Identify new and changed files for incremental indexing.
    
    Args:
        xml_files_to_process: List of XML file paths to check
        metadata_path: Path to the metadata file
        
    Returns:
        Tuple of (new_files, changed_files)
    """
    try:
        existing_metadata = load_metadata(metadata_path)
        new_files = []
        changed_files = []
        
        for file_path in xml_files_to_process:
            if file_path not in existing_metadata:
                # New file
                new_files.append(file_path)
            else:
                # Check if file has changed
                current_metadata = get_file_metadata(file_path)
                stored_metadata = existing_metadata[file_path]
                
                # Compare hash to detect changes
                if (current_metadata.get('hash') != stored_metadata.get('hash') or
                    current_metadata.get('modified') != stored_metadata.get('modified')):
                    changed_files.append(file_path)
        
        logger.info(f"File change detection: {len(new_files)} new, {len(changed_files)} changed")
        return new_files, changed_files
        
    except Exception as e:
        logger.error(f"Error identifying changed files: {e}")
        # Fallback: treat all files as new
        return xml_files_to_process, []

# --- Standard Search Functions ---
def search_by_sender(index: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
    """Search by sender with enhanced error handling."""
    if not query or not query.strip():
        logger.warning("Empty query provided for sender search")
        return []

    try:
        results = []
        query_lower = query.lower().strip()

        for item in index:
            sender = item.get('sender', "Unknown")
            if isinstance(sender, str) and query_lower in sender.lower():
                results.append(item)

        logger.debug(f"Sender search for '{query}': {len(results)} results")
        return results
    except Exception as e:
        logger.error(f"Error in sender search: {e}")
        return []

def search_by_recipient(index: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
    """Search by recipient with enhanced error handling."""
    if not query or not query.strip():
        logger.warning("Empty query provided for recipient search")
        return []

    try:
        results = []
        query_lower = query.lower().strip()

        for item in index:
            recipient = item.get('recipient', "Unknown")
            if isinstance(recipient, str) and query_lower in recipient.lower():
                results.append(item)

        logger.debug(f"Recipient search for '{query}': {len(results)} results")
        return results
    except Exception as e:
        logger.error(f"Error in recipient search: {e}")
        return []

def search_by_year(index: List[Dict[str, Any]], year_query: str) -> List[Dict[str, Any]]:
    """Search by year with enhanced validation."""
    if not year_query or not year_query.strip():
        logger.warning("Empty query provided for year search")
        return []

    try:
        year_int = int(year_query.strip())

        # Validate year range
        if year_int < 1700 or year_int > 2100: # Adjusted to be more realistic for historical docs
            logger.warning(f"Year {year_int} seems out of typical historical range")

        results = []
        for item in index:
            if item.get('year') == year_int:
                results.append(item)

        logger.debug(f"Year search for {year_int}: {len(results)} results")
        return results

    except ValueError:
        logger.error(f"Invalid year format: '{year_query}'. Please enter a 4-digit year.")
        return []
    except Exception as e:
        logger.error(f"Error in year search: {e}")
        return []

def search_by_place(index: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
    """Search by place with enhanced error handling."""
    if not query or not query.strip():
        logger.warning("Empty query provided for place search")
        return []

    try:
        results = []
        query_lower = query.lower().strip()

        for item in index:
            places = item.get('places', [])
            if isinstance(places, list):
                if any(query_lower in place.lower() for place in places if isinstance(place, str)):
                    results.append(item)

        logger.debug(f"Place search for '{query}': {len(results)} results")
        return results
    except Exception as e:
        logger.error(f"Error in place search: {e}")
        return []

def search_by_keyword(index: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
    """Enhanced keyword search with better text handling."""
    if not query or not query.strip():
        logger.warning("Empty query provided for keyword search")
        return []

    try:
        results = []
        query_lower = query.lower().strip()

        for item in index:
            # Search in title
            title = item.get('title', "")
            in_title = isinstance(title, str) and query_lower in title.lower()

            # Search in description
            description = item.get('description', "")
            in_desc = isinstance(description, str) and query_lower in description.lower()

            # Search in full text
            full_text = item.get('full_text', "")
            in_text = isinstance(full_text, str) and query_lower in full_text.lower()

            if in_title or in_desc or in_text:
                results.append(item)

        logger.debug(f"Keyword search for '{query}': {len(results)} results")
        return results
    except Exception as e:
        logger.error(f"Error in keyword search: {e}")
        return []

# --- Topic Modeling Search Functions ---
def list_available_topics(lda_model, top_n_terms: int = 5) -> Optional[List[Tuple[int, str]]]:
    """
    List available topics with error handling.

    Returns:
        List of (topic_id, terms_string) tuples or None if error
    """
    if not lda_model:
        logger.warning("LDA model not loaded.")
        print("LDA model not loaded. No topics to list.") # User feedback
        return None

    try:
        topics = []
        print("\n--- Available Topics ---")
        for tid in range(lda_model.num_topics):
            terms = ", ".join(t for t, _ in lda_model.show_topic(tid, topn=top_n_terms))
            topics.append((tid, terms))
            print(f"Topic {tid:02}: {terms}")
        print("---")
        return topics
    except Exception as e:
        logger.error(f"Error listing topics: {e}")
        print(f"Error listing topics: {e}") # User feedback
        return None

def search_by_topic_id(index: List[Dict[str, Any]], topic_id: str) -> List[Dict[str, Any]]:
    """Search by topic ID with validation."""
    if not topic_id or not topic_id.strip():
        logger.warning("Empty topic ID provided")
        return []

    try:
        tid = int(topic_id.strip())

        # Validate topic ID range
        if global_lda_model and (tid < 0 or tid >= global_lda_model.num_topics):
            logger.warning(f"Topic ID {tid} out of range (0-{global_lda_model.num_topics-1})")
            return []

        results = [item for item in index if item.get("topic_id") == tid]
        logger.debug(f"Topic ID search for {tid}: {len(results)} results")
        return results

    except ValueError:
        logger.error(f"Topic ID must be an integer, got: '{topic_id}'")
        return []
    except Exception as e:
        logger.error(f"Error in topic ID search: {e}")
        return []

def search_by_topic_terms(index: List[Dict[str, Any]], keyword: str) -> List[Dict[str, Any]]:
    """Search by topic terms with enhanced matching."""
    if not keyword or not keyword.strip():
        logger.warning("Empty keyword provided for topic terms search")
        return []

    try:
        kw_lower = keyword.lower().strip()
        results = []

        for item in index:
            topic_terms = item.get("topic_terms", [])
            if isinstance(topic_terms, list):
                if any(kw_lower in term.lower() for term in topic_terms if isinstance(term, str)):
                    results.append(item)

        logger.debug(f"Topic terms search for '{keyword}': {len(results)} results")
        return results
    except Exception as e:
        logger.error(f"Error in topic terms search: {e}")
        return []

# --- Enhanced AI-Powered Smart Search Functions (spaCy) with Keyword Weighting ---
def process_smart_query(query_text, nlp_model):
    """
    Enhanced smart query processing with keyword weighting for better relevance.
    Assigns higher weights to more important parts of speech.

    Args:
        query_text: User's search query
        nlp_model: spaCy NLP model

    Returns:
        Dictionary with extracted entities, keywords, and their weights
    """
    if not nlp_model or not SPACY_AVAILABLE : # Check SPACY_AVAILABLE
        # Simple fallback without weights
        keywords = [kw.lower() for kw in query_text.split() if len(kw) > 2]
        return {
            'keywords': keywords,
            'keyword_weights': {kw: 1.0 for kw in keywords},
            'persons': [],
            'year': None,
            'places': []
        }

    doc = nlp_model(query_text)
    filters = {
        'persons': [],
        'year': None,
        'places': [],
        'keywords': [],
        'keyword_weights': {}  # New: store weights for keywords
    }

    processed_tokens_for_keywords = set() 

    # Process named entities with high importance
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            filters['persons'].append(ent.text)
            filters['keyword_weights'][ent.text.lower()] = 3.0  # High weight for persons
            for token in ent:
                processed_tokens_for_keywords.add(token.lemma_)

        elif ent.label_ == "DATE":
            year_match = re.search(r'\b(\d{4})\b', ent.text)
            if year_match:
                filters['year'] = int(year_match.group(1))
                filters['keyword_weights']['year_' + year_match.group(1)] = 2.5  # High weight for years
            for token in ent:
                processed_tokens_for_keywords.add(token.lemma_)

        elif ent.label_ in ["GPE", "LOC"]:
            filters['places'].append(ent.text)
            filters['keyword_weights'][ent.text.lower()] = 2.5  # High weight for places
            for token in ent:
                processed_tokens_for_keywords.add(token.lemma_)

    # Process individual tokens with weighted importance
    for token in doc:
        if token.lemma_ not in processed_tokens_for_keywords and not token.is_stop and not token.is_punct:
            lemma = token.lemma_.lower()

            # Assign weights based on part of speech and importance
            if token.pos_ == "PROPN":  # Proper nouns (names, places)
                filters['keywords'].append(lemma)
                filters['keyword_weights'][lemma] = 2.0

            elif token.pos_ == "NOUN":  # Common nouns (subjects, objects)
                filters['keywords'].append(lemma)
                filters['keyword_weights'][lemma] = 1.5

            elif token.pos_ == "VERB" and len(lemma) > 3:  # Important verbs
                filters['keywords'].append(lemma)
                filters['keyword_weights'][lemma] = 1.0

            elif token.pos_ == "ADJ" and len(lemma) > 4:  # Descriptive adjectives
                filters['keywords'].append(lemma)
                filters['keyword_weights'][lemma] = 0.8

            # Skip less important parts of speech

    # Remove duplicates while preserving weights
    filters['keywords'] = list(set(filters['keywords']))
    filters['persons'] = list(set(filters['persons']))
    filters['places'] = list(set(filters['places']))

    return filters

def execute_smart_search(letter_index, filters):
    """
    Enhanced smart search execution with keyword weighting for improved relevance scoring.

    Args:
        letter_index: List of document dictionaries
        filters: Dictionary with search criteria and keyword weights

    Returns:
        List of matching documents with relevance scores
    """
    if not filters:
        return []

    results = []
    keyword_weights = filters.get('keyword_weights', {})

    for item in letter_index:
        match_score = 0.0
        base_match = True

        # Check person matches with high weight
        if filters.get('persons'):
            person_match = False
            for pq in filters['persons']:
                pql = pq.lower()
                if (pql in item.get('sender', '').lower() or
                        pql in item.get('recipient', '').lower() or
                        any(pql in p.lower() for p in item.get('people', []))):
                    person_match = True
                    match_score += keyword_weights.get(pql, 3.0)
                    break # Found a person match, can stop checking persons for this item
            if not person_match:
                base_match = False

        # Check year match with high weight
        if base_match and filters.get('year') is not None:
            if item.get('year') == filters['year']:
                match_score += 2.5  # High weight for exact year match
            else:
                base_match = False

        # Check place matches with high weight
        if base_match and filters.get('places'):
            place_match = False
            for pq in filters['places']:
                if any(pq.lower() in p.lower() for p in item.get('places', [])):
                    place_match = True
                    match_score += keyword_weights.get(pq.lower(), 2.5)
                    break # Found a place match
            if not place_match:
                base_match = False

        # Check keyword matches with weighted scoring
        if base_match and filters.get('keywords'):
            text_to_search = (
                item.get('title', '') + " " +
                item.get('description', '') + " " +
                item.get('full_text', '')
            ).lower()

            keyword_matches = 0
            # total_weight = 0 # Not used

            for kw in filters['keywords']:
                if kw.lower() in text_to_search:
                    weight = keyword_weights.get(kw.lower(), 1.0)
                    match_score += weight
                    keyword_matches += 1
                    # total_weight += weight # Not used

            # Require at least some keyword matches for relevance if keywords were specified
            if not filters.get('persons') and not filters.get('year') and not filters.get('places') and keyword_matches == 0:
                 base_match = False
            elif keyword_matches > 0 : # Boost if there are keywords and they matched
                 match_score += (keyword_matches / len(filters['keywords'])) * 0.5


        if base_match and match_score > 0:
            # Add the item with its relevance score
            item_with_score = item.copy()
            item_with_score['relevance_score'] = round(match_score, 3)
            results.append(item_with_score)

    # Sort by relevance score (highest first)
    results.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)

    return results

# --- Enhanced AI-Powered Semantic Search Functions (Sentence Transformers) ---
def execute_semantic_search(letter_index, query_text, sentence_model, top_n=5):
    """
    Enhanced semantic search with relevant snippet extraction and improved relevance scoring.

    Args:
        letter_index: List of document dictionaries
        query_text: User's search query
        sentence_model: Sentence transformer model
        top_n: Number of top results to return

    Returns:
        List of documents with similarity scores and relevant snippets
    """
    if not sentence_model or not util or not SENTENCE_TRANSFORMERS_AVAILABLE:
        print("Sentence model or util not available for semantic search.")
        logger.warning("Sentence model or util not available for semantic search.")
        return []
    if not query_text:
        print("Semantic search query is empty.")
        logger.warning("Semantic search query is empty.")
        return []

    try:
        query_embedding = sentence_model.encode(query_text, convert_to_tensor=True)
        
        temp_corpus_embeddings_list = []
        docs_for_semantic_search = [] # Store the actual document objects

        for doc_item in letter_index:
            embedding_data = doc_item.get('embedding')
            current_tensor = None

            if isinstance(embedding_data, list) and embedding_data and all(isinstance(x, (float, int, np.float32, np.float64)) for x in embedding_data):
                try:
                    current_tensor = util.torch.tensor(embedding_data, dtype=torch.float32)
                except Exception as e:
                    logger.warning(f"Could not convert list embedding to tensor for doc {doc_item.get('file_name', 'N/A')}: {e}")
            elif isinstance(embedding_data, np.ndarray):
                try:
                    current_tensor = util.torch.from_numpy(embedding_data.astype(np.float32))
                except Exception as e:
                    logger.warning(f"Could not convert np embedding to tensor for doc {doc_item.get('file_name', 'N/A')}: {e}")
            elif util.torch.is_tensor(embedding_data):
                current_tensor = embedding_data.to(dtype=torch.float32)
            
            if current_tensor is not None:
                if current_tensor.ndim == 1 and current_tensor.numel() > 0: # Ensure it's a 1D tensor and not empty
                    temp_corpus_embeddings_list.append(current_tensor)
                    docs_for_semantic_search.append(doc_item)
                else:
                    logger.warning(f"Embedding for doc {doc_item.get('file_name', 'N/A')} was not a valid 1D tensor (shape: {current_tensor.shape}). Skipping.")
            elif embedding_data is not None:
                logger.warning(f"Embedding for doc {doc_item.get('file_name', 'N/A')} was of unexpected type {type(embedding_data)} or invalid. Skipping.")
            # If embedding_data is None, it's silently skipped, which is fine.

        if not temp_corpus_embeddings_list:
            print("No valid tensor embeddings found in corpus for semantic search.") # This matches the error you saw
            logger.warning("No valid tensor embeddings found in corpus for semantic search after filtering and conversion attempts.")
            return []

        corpus_embeddings = util.torch.stack(temp_corpus_embeddings_list)
    
        # Ensure embeddings are on CPU if they are PyTorch tensors
        if hasattr(query_embedding, 'cpu'):
            query_embedding = query_embedding.cpu()
            if hasattr(corpus_embeddings, 'cpu'):
                corpus_embeddings = corpus_embeddings.cpu()

        hits = util.semantic_search(
            query_embedding,
            corpus_embeddings,
            top_k=min(top_n * 2, len(corpus_embeddings))
        )
    
        search_results = []
        if hits and hits[0]:
            for hit in hits[0][:top_n]:
                # hit['corpus_id'] is an index into docs_for_semantic_search
                original_item = docs_for_semantic_search[hit['corpus_id']].copy()
                original_item['similarity_score'] = hit['score'] 

                relevant_snippet = extract_relevant_snippet(
                        original_item, query_text, sentence_model
                )
                original_item['relevant_snippet'] = relevant_snippet
                search_results.append(original_item)
        return search_results

    except Exception as e:
        logger.error(f"Error in enhanced semantic search: {e}", exc_info=True)
        return []

def extract_relevant_snippet(document, query_text, sentence_model, max_snippet_length=300):
    """
    Extract the most relevant snippet from a document based on the query.

    Args:
        document: Document dictionary
        query_text: User's search query
        sentence_model: Sentence transformer model
        max_snippet_length: Maximum length of the snippet

    Returns:
        Most relevant snippet from the document
    """
    try:
        # Combine all text content
        full_text = f"{document.get('title', '')} {document.get('description', '')} {document.get('full_text', '')}"

        if not full_text or len(full_text.strip()) < 50: # Adjusted minimum length
            return "No sufficient text content available."

        # Split text into sentences for more granular analysis
        sentences = re.split(r'[.!?]+', full_text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 20]  # Filter very short sentences

        if not sentences:
            # Fallback to paragraph-based splitting
            paragraphs = full_text.split('\n\n')
            sentences = [p.strip() for p in paragraphs if len(p.strip()) > 50] # Increased min length for paragraphs

        if not sentences:
            # Final fallback - return truncated full text
            return full_text[:max_snippet_length] + "..." if len(full_text) > max_snippet_length else full_text

        # Encode query and all sentences
        query_embedding = sentence_model.encode(query_text, convert_to_tensor=True)
        sentence_embeddings = sentence_model.encode(sentences, convert_to_tensor=True)

        # Find most similar sentences
        similarities = util.cos_sim(query_embedding, sentence_embeddings)[0]

        # Get top 2-3 most similar sentences
        top_indices = similarities.argsort(descending=True)[:3]

        # Combine the most relevant sentences into a snippet
        relevant_sentences = []
        total_length = 0

        for idx_tensor in top_indices: # Iterate through tensor elements
            idx = idx_tensor.item() # Get Python int from tensor
            sentence = sentences[idx]
            if total_length + len(sentence) <= max_snippet_length:
                relevant_sentences.append(sentence)
                total_length += len(sentence)
            else:
                # Truncate last sentence if needed
                remaining_space = max_snippet_length - total_length
                if remaining_space > 50:  # Only add if meaningful space left
                    relevant_sentences.append(sentence[:remaining_space] + "...")
                break

        if relevant_sentences:
            snippet = " ".join(relevant_sentences)
            return snippet.strip()
        else:
            # Fallback to most similar single sentence
            best_sentence_idx = similarities.argmax().item()
            best_sentence = sentences[best_sentence_idx]
            if len(best_sentence) > max_snippet_length:
                return best_sentence[:max_snippet_length] + "..."
            return best_sentence

    except Exception as e:
        logger.warning(f"Error extracting relevant snippet: {e}")
        # Fallback to simple truncation
        full_text = f"{document.get('title', '')} {document.get('description', '')} {document.get('full_text', '')}"
        if len(full_text) > max_snippet_length:
            return full_text[:max_snippet_length] + "..."
        return full_text if full_text.strip() else "No text content available."


# --- Enhanced Display Function ---
def display_results(results, is_semantic_search=False):
    """
    Enhanced results display with relevance scoring and better snippet presentation.

    Args:
        results: List of matching documents
        is_semantic_search: Whether this is a semantic search (affects snippet display)
    """
    if not results:
        print("No matching letters found.")
        return

    print(f"\n--- Found {len(results)} Matching Letter(s) ---")

    for i, item in enumerate(results, 1):
        print(f"\nResult {i}:")
        print(f"  File:      {item.get('file_name', 'N/A')}")

        # Show different types of scores based on search type
        if is_semantic_search and 'similarity_score' in item:
            print(f"  Similarity: {item.get('similarity_score', 0.0):.4f}")
        elif 'relevance_score' in item:
            print(f"  Relevance:  {item.get('relevance_score', 0.0):.3f}")

        print(f"  Title:     {item.get('title', 'N/A')}")
        print(f"  Date:      {item.get('main_date', 'N/A')}")
        print(f"  Sender:    {item.get('sender', 'N/A')}")
        print(f"  Recipient: {item.get('recipient', 'N/A')}")
        print(f"  Places:    {', '.join(item.get('places', [])) or 'N/A'}")

        # Enhanced topic information
        if item.get('topic_id') is not None: # Check if topic_id exists and is not None
            topic_info = f"Topic {item['topic_id']}"
            if 'topic_terms' in item and item['topic_terms']:
                topic_info += f" ({', '.join(item['topic_terms'][:5])})"
            if 'topic_score' in item and item['topic_score'] is not None:
                topic_info += f" score={item['topic_score']:.2f}"
            print(f"  Topic:     {topic_info}")

        # Enhanced snippet display
        if is_semantic_search and 'relevant_snippet' in item:
            # Use the extracted relevant snippet for semantic search
            snippet = item['relevant_snippet']
            if len(snippet) > 250:
                snippet = snippet[:250] + '...'
            print(f"  Relevance: {snippet}") # Changed label to "Relevance" as in prompt
        else:
            # Use traditional snippet for other search types
            text_snippet = item.get('full_text', 'N/A')
            if text_snippet != 'N/A' and text_snippet:
                snippet = (text_snippet[:200] + '...') if len(text_snippet) > 200 else text_snippet
                # Fix f-string syntax by moving regex outside
                cleaned_snippet = re.sub(r'\s+', ' ', snippet).strip()
                print(f"  Snippet:   {cleaned_snippet}")
            else:
                print(f"  Snippet:   N/A")

    print("-" * 30)

# --- Help Function ---
def display_help():
    print("\n--- Chatbot Help ---")
    print("Enter your query in one of the following formats:")
    print("1. Specific field search: <type>:<query>")
    print("     Available types:")
    print("       sender, recipient, year, place, keyword")
    print("\n2. Smart search (spaCy NLP-based): smart:<your question>")
    print("     Example: smart:letters from McClure about cannons to Pettus in 1859")
    print("\n3. Semantic search (Sentence Transformer-based): semantic:<your question>")
    print("     Example: semantic:concerns about military readiness and arms")
    print("\n4. Topic search:")
    print("       topic:<id or keyword>     e.g. topic:7    or    topic:cotton")
    print("       list_topics               show all topic ids with top terms")
    print("\n5. Phase 2 Advanced Search:")
    print("       answer:<your question>    extractive QA for direct answers")
    print("       hybrid:<your query>       combines keyword + semantic search")
    print("       htopic:<keyword>          search hierarchical topics")
    print("       dtopic:<year>:<keyword>   search dynamic topics by year")
    print("       list_htopics              show hierarchical topics")
    print("       list_dtopics              show dynamic topic evolution")
    print("\nOther commands: help, quit")
    print("--------------------")

# --- Main Chat Loop ---
def run_chatbot(letter_index, spacy_nlp_model, sentence_model):
    if not letter_index: 
        print("Index is empty. Cannot start chatbot.")
        return

    print("\n--- Mississippi Letters Chatbot (Advanced AI with Persistent Index) ---")
    # Warnings for missing AI models
    if SPACY_AVAILABLE and not spacy_nlp_model: # Check if spacy itself is available
        print("Warning: spaCy model failed to load. 'smart:' search limited.")
    elif not SPACY_AVAILABLE:
        print("Warning: spaCy library not found. 'smart:' search disabled.")
        
    if SENTENCE_TRANSFORMERS_AVAILABLE and not sentence_model: # Check if sentence_transformers itself is available
        print(f"Warning: SentenceTransformer model '{SENTENCE_MODEL_NAME}' failed. 'semantic:' search disabled.")
    elif not SENTENCE_TRANSFORMERS_AVAILABLE:
        print("Warning: sentence-transformers library not loaded. 'semantic:' search disabled.")
        
    if GENSIM_AVAILABLE and global_lda_model:
        print(f"Topic modeling enabled with {global_lda_model.num_topics} topics.")
    else:
        print("Topic modeling not available.")
        
    # Phase 2: Status messages
    if global_hdp_model:
        topics = global_hdp_model.show_topics(num_topics=-1, formatted=False)
        active_topics = len([t for t in topics if t[1] and len(t[1]) > 0])
        print(f"Hierarchical topic modeling enabled with {active_topics} discovered topics.")

    if global_dtm_model and global_time_slices:
        print(f"Dynamic topic modeling enabled with {global_dtm_model.num_topics} topics across {len(global_time_slices)} time periods.")

    if global_qa_pipeline:
        print("Question Answering (QA) enabled for direct answer extraction.")

    if HF_TRANSFORMERS_AVAILABLE: # This was previously used for QA, now more generally for hybrid
        print("Advanced search features (QA, Hybrid) may be available if models loaded.")

        
    display_help()
    
    while True:
        user_input = input("\nEnter query (or 'help'/'quit'): ").strip()
        if user_input.lower() == 'quit':
            break
            
        if user_input.lower() == 'help':
            display_help()
            continue
            
        if user_input.lower() == 'list_topics':
            list_available_topics(global_lda_model)
            continue
            
        # Phase 2: Advanced topic listing commands
        if user_input.lower() == 'list_htopics':
            list_hierarchical_topics(global_hdp_model)
            continue

        if user_input.lower() == 'list_dtopics':
            list_dynamic_topic_evolution(global_dtm_model, global_time_slices)
            continue
            
        if ':' not in user_input:
            print("Invalid format. Use <type>:<query>, smart:<query>, or semantic:<query>. Type 'help'.")
            continue
            
        try:
            search_type, query = user_input.split(':', 1)
            search_type = search_type.strip().lower()
            query = query.strip()
        except ValueError:
            print("Invalid format. Use <type>:<query>, smart:<query>, or semantic:<query>.")
            continue
            
        if not query:
            print("Please provide a search term after the colon.")
            continue
            
        results = []
        is_semantic = False
        
        if search_type == 'smart':
            if spacy_nlp_model:
                filters = process_smart_query(query, spacy_nlp_model)
                print(f"Smart search filters: {filters}") 
                results = execute_smart_search(letter_index, filters)
            else:
                print("spaCy model not loaded. Smart search unavailable.")
                
        elif search_type == 'semantic':
            if sentence_model:
                results = execute_semantic_search(letter_index, query, sentence_model)
                is_semantic = True
            else:
                print("SentenceTransformer model not available for semantic search.")
                
        elif search_type == 'sender':
            results = search_by_sender(letter_index, query)
        elif search_type == 'recipient':
            results = search_by_recipient(letter_index, query)
        elif search_type == 'year':
            results = search_by_year(letter_index, query)
        elif search_type == 'place':
            results = search_by_place(letter_index, query)
        elif search_type == 'keyword':
            results = search_by_keyword(letter_index, query)
        elif search_type == 'topic':
            # Accept either id=<int> or term=<word>
            if query.isdigit():
                results = search_by_topic_id(letter_index, query)
            else:
                results = search_by_topic_terms(letter_index, query)

        # Phase 2: Advanced search types
        elif search_type == 'answer':
            if global_qa_pipeline and sentence_model:
                answers = execute_extractive_qa(letter_index, query, global_qa_pipeline, sentence_model)
                display_qa_results(answers)
                continue  # QA has its own display function
            else:
                print("QA pipeline not available. Check if transformers library is installed.")
                continue

        elif search_type == 'hybrid':
            if sentence_model and HF_TRANSFORMERS_AVAILABLE: # Check both sentence model and TFIDF dependencies
                results = execute_hybrid_search(letter_index, query, sentence_model)
                display_hybrid_results(results)
                continue  # Hybrid has its own display function
            else:
                print("Sentence model or TFIDF dependencies not available for hybrid search.")
                continue

        elif search_type == 'htopic':
            results = search_hierarchical_topics(letter_index, query)
            # This function prints its own results for now
            if not results: # if it returns an empty list it means it printed a message.
                print("(Note: Document retrieval for htopic matches is not fully implemented in this example version)")
            continue

        elif search_type == 'dtopic':
            # Parse year and keyword: dtopic:1865:military
            if ':' in query:
                try:
                    year_str, keyword = query.split(':', 1)
                    year_val = int(year_str.strip())
                    keyword_val = keyword.strip()
                    results = search_dynamic_topics_by_year(letter_index, year_val, keyword_val)
                    # This function prints its own results for now
                    if not results:
                         print("(Note: Document retrieval for dtopic matches is not fully implemented in this example version)")
                    continue
                except ValueError:
                    print("Invalid format for dtopic. Use: dtopic:<year>:<keyword>")
                    continue
            else:
                print("Dynamic topic search requires year and keyword: dtopic:<year>:<keyword>")
                continue

        else:
            print(f"Unknown search type: '{search_type}'. Type 'help'.")
            continue
            
        display_results(results, is_semantic_search=is_semantic)
        
    print("Exiting chatbot. Goodbye!")

def get_file_list_from_args(args):
    """Get the list of files to process based on command line arguments."""
    files_to_process = []
    
    if hasattr(args, 'xmldir') and args.xmldir:
        if os.path.isdir(args.xmldir):
            print(f"Target XML directory: {args.xmldir}")
            files_to_process = glob.glob(os.path.join(args.xmldir, '*.xml'))
        else:
            print(f"Error: Provided directory '{args.xmldir}' does not exist.")
    elif hasattr(args, 'files') and args.files:
        print("Target XML files (from command line):")
        for f_path in args.files: 
            print(f"  - {f_path}")
        files_to_process = args.files
    else:
        print("No XML directory or file list provided. Using default file list from current directory.")
        files_to_process = [f for f in DEFAULT_UPLOADED_FILES if os.path.exists(f)]
        missing_defaults = [f for f in DEFAULT_UPLOADED_FILES if not os.path.exists(f)]
        if missing_defaults:
            print(f"Warning: Some default files not found: {', '.join(missing_defaults)}")
            
    return files_to_process

def load_nlp_models():
    """
    Load and return NLP models needed for search functions.
    
    Returns:
        tuple: (spacy_nlp_model, sentence_transformer_model, qa_pipeline)
    """
    # Check for MPS (Metal Performance Shaders) availability for Apple Silicon GPUs
    # Force CPU mode to avoid MPS allocation errors
    mps_available = False  # Temporarily disable MPS
    # mps_available = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    device_to_use = "mps" if mps_available else "cpu" # This will be used for SentenceTransformer and QA
    gpu_preferred = False # Default to False

    if mps_available:
        logger.info("MPS (Metal Performance Shaders) backend is available. Will attempt to use GPU for spaCy.")
        if SPACY_AVAILABLE:
            try:
                gpu_preferred = spacy.prefer_gpu()
                if gpu_preferred:
                    logger.info("Successfully called spacy.prefer_gpu() and GPU is available.")
                else:
                    logger.info("Successfully called spacy.prefer_gpu() but GPU is not available or not used, or an error occurred.")
            except Exception as e:
                logger.warning(f"Could not set spaCy to prefer GPU (MPS): {e}")
    else:
        logger.info("MPS backend not available. Will use CPU for all models.")
        if SPACY_AVAILABLE:
            try:
                spacy.require_cpu()
                logger.info("Successfully called spacy.require_cpu().")
                gpu_preferred = False # Explicitly set to False
            except Exception as e:
                logger.warning(f"Could not set spaCy to require CPU: {e}")
    
    # Load spaCy model
    spacy_nlp = None
    if SPACY_AVAILABLE: 
        model_name_to_load = "en_core_web_trf"
        try:
            logger.info(f"Attempting to load spaCy model \'{model_name_to_load}\'...")
            spacy_nlp = spacy.load(model_name_to_load)
            # Determine accelerator based on the outcome of prefer_gpu or if require_cpu was called
            accelerator = "gpu" if gpu_preferred and mps_available else "cpu"
            print(f"spaCy model \'{model_name_to_load}\' loaded successfully. Using device: {accelerator}") # For immediate stdout
            logger.info(f"spaCy model \'{model_name_to_load}\' loaded successfully. Using device: {accelerator}")
        except OSError as ose:
            print(f"spaCy model '{model_name_to_load}' not found. Please download it: python -m spacy download {model_name_to_load}") # For immediate stdout
            logger.error(f"spaCy model '{model_name_to_load}' not found. OSError: {ose}", exc_info=True)
            spacy_nlp = None
        except Exception as e: 
            print(f"An unexpected error occurred while loading spaCy model '{model_name_to_load}': {e}") # For immediate stdout
            logger.error(f"An unexpected error occurred while loading spaCy model '{model_name_to_load}': {e}", exc_info=True)
            spacy_nlp = None
        except BaseException as be: 
            print(f"A critical error (BaseException) occurred while loading spaCy model '{model_name_to_load}': {be}") # For immediate stdout
            logger.critical(f"A critical error (BaseException) occurred while loading spaCy model '{model_name_to_load}': {be}", exc_info=True)
            spacy_nlp = None 
    else:
        logger.warning("spaCy library is not available. Cannot load spaCy model.")
        print("spaCy library is not available. spaCy model loading skipped.") # For immediate stdout

    # Load sentence transformer model
    sentence_transformer_model = None
    if SENTENCE_TRANSFORMERS_AVAILABLE: # Check if sentence_transformers library was imported
        try:
            sentence_transformer_model = SentenceTransformer(SENTENCE_MODEL_NAME, device=device_to_use)
            print(f"SentenceTransformer model '{SENTENCE_MODEL_NAME}' loaded on device: {device_to_use}.")
        except Exception as e:
            print(f"Error loading SentenceTransformer model '{SENTENCE_MODEL_NAME}' on device '{device_to_use}': {e}")
            # Try CPU as fallback if MPS loading failed
            if device_to_use == "mps":
                print("Retrying SentenceTransformer with CPU...")
                try:
                    sentence_transformer_model = SentenceTransformer(SENTENCE_MODEL_NAME, device="cpu")
                    print(f"SentenceTransformer model '{SENTENCE_MODEL_NAME}' loaded on device: cpu (fallback).")
                except Exception as e_cpu:
                    print(f"Error loading SentenceTransformer model on CPU (fallback): {e_cpu}")
                    sentence_transformer_model = None
            else:
                sentence_transformer_model = None
            
    # Load QA pipeline (Phase 2)
    # Pass the determined device_to_use to initialize_qa_model
    qa_pipeline_model = initialize_qa_model(device_to_use=device_to_use)
    if qa_pipeline_model:
        # The pipeline object itself doesn't directly expose the device of its internal model easily,
        # but initialize_qa_model will log the device it attempted to use.
        print(f"Question Answering pipeline initialized (attempted device: {device_to_use}).")
    
    return spacy_nlp, sentence_transformer_model, qa_pipeline_model

def train_hierarchical_topic_model(index: List[Dict[str, Any]], spacy_nlp_model) -> Tuple[Any, Any]:
    """
    Train a Hierarchical Dirichlet Process (HDP) model to discover topic hierarchies.
    This automatically determines the number of topics and their relationships.

    Args:
        index: List of document dictionaries with LDA tokens
        spacy_nlp_model: spaCy model for preprocessing (currently unused, relies on existing lda_tokens)

    Returns:
        Tuple of (hdp_model, hdp_dictionary)
    """
    if not ADVANCED_GENSIM_AVAILABLE or not index:
        logger.warning("Advanced Gensim not available or no documents for hierarchical topic modeling")
        return None, None

    try:
        logger.info("Starting hierarchical topic modeling with HDP...")

        # Use existing LDA tokens if available
        documents_with_tokens = [item for item in index if item.get('lda_tokens')]

        if len(documents_with_tokens) < config.HIERARCHICAL_MIN_TOPICS:
            logger.warning(f"Insufficient documents ({len(documents_with_tokens)}) for hierarchical modeling")
            return None, None

        all_tokens = [item['lda_tokens'] for item in documents_with_tokens]

        # Create dictionary and corpus
        hdp_dictionary = corpora.Dictionary(all_tokens)

        # Filter dictionary for HDP
        hdp_dictionary.filter_extremes(
            no_below=5,  # Must appear in at least 5 documents
            no_above=0.4,  # Must not appear in more than 40% of documents
            keep_n=2000  # Keep top 2000 most frequent
        )

        corpus = [hdp_dictionary.doc2bow(tokens) for tokens in all_tokens]
        corpus = [doc for doc in corpus if doc and len(doc) >= 3] # Filter out empty documents

        if len(corpus) < config.HIERARCHICAL_MIN_TOPICS:
            logger.warning("Insufficient valid documents after corpus creation for HDP")
            return None, None

        logger.info(f"Training HDP model on {len(corpus)} documents...")

        # Train HDP model
        hdp_model = HdpModel(
            corpus=corpus,
            id2word=hdp_dictionary,
            alpha=config.HDP_ALPHA,
            gamma=config.HDP_GAMMA,
            # max_chunks=None, # Default
            # max_time=None, # Default
            chunksize=100, # Smaller for potentially better quality with smaller datasets
            kappa=1.0, # Default
            tau=64.0, # Default
            K=config.HIERARCHICAL_MAX_TOPICS, # Upper bound on number of topics at one level
            T=100,  # Number of top level truncation (default 150)
            random_state=42
        )

        # Get the actual topics discovered
        topics = hdp_model.show_topics(num_topics=-1, num_words=8, formatted=False)
        active_topics = [t for t in topics if t[1] and len(t[1]) > 0]

        logger.info(f"HDP discovered {len(active_topics)} active topics")

        # Log sample hierarchical topics
        sample_topics_count = min(5, len(active_topics)) # Renamed variable
        logger.info("Sample hierarchical topics:")
        for i in range(sample_topics_count):
            topic_id, words = active_topics[i]
            terms = ", ".join([word for word, _ in words[:8]])
            logger.info(f"  H-Topic {topic_id}: {terms}")

        return hdp_model, hdp_dictionary

    except Exception as e:
        logger.error(f"Error in hierarchical topic modeling: {e}")
        return None, None

def train_dynamic_topic_model(index: List[Dict[str, Any]]) -> Tuple[Any, Any, List[int]]:
    """
    Train a Dynamic Topic Model to track topic evolution over time.
    Uses year information to create time slices and track topic changes.

    Args:
        index: List of document dictionaries with year information

    Returns:
        Tuple of (dtm_model, dtm_dictionary, time_slices)
    """
    if not ADVANCED_GENSIM_AVAILABLE or not index:
        logger.warning("Advanced Gensim not available or no documents for dynamic topic modeling")
        return None, None, None

    try:
        logger.info("Starting dynamic topic modeling...")

        # Filter documents with valid years and tokens
        docs_with_years = [
            item for item in index
            if item.get('year') and item.get('lda_tokens') and
            1800 <= item['year'] <= 1900  # Example: Focus on a specific century like Civil War & Reconstruction
        ]

        if len(docs_with_years) < 100: # Increased minimum
            logger.warning(f"Insufficient documents with years ({len(docs_with_years)}) for dynamic modeling")
            return None, None, None

        # Group documents by year and create time slices
        year_docs = {}
        for item in docs_with_years:
            year = item['year']
            if year not in year_docs:
                year_docs[year] = []
            year_docs[year].append(item['lda_tokens'])

        # Filter years with sufficient documents
        valid_years = [year for year, docs in year_docs.items()
                       if len(docs) >= config.DYNAMIC_MIN_DOCS_PER_SLICE]
        valid_years.sort()

        if len(valid_years) < 3: # Need at least a few time slices
            logger.warning(f"Insufficient years with adequate documents ({len(valid_years)}) for DTM")
            return None, None, None

        logger.info(f"Dynamic modeling across {len(valid_years)} years: {valid_years[0]}-{valid_years[-1]}")

        # Create time-ordered corpus and time slices
        time_ordered_corpus_tokens = [] # Renamed
        time_slices = []

        for year in valid_years:
            year_docs_tokens = year_docs[year]
            time_ordered_corpus_tokens.extend(year_docs_tokens)
            time_slices.append(len(year_docs_tokens))
            logger.info(f"  Year {year}: {len(year_docs_tokens)} documents")

        # Create dictionary from all documents
        dtm_dictionary = corpora.Dictionary(time_ordered_corpus_tokens)
        dtm_dictionary.filter_extremes(
            no_below=config.MIN_CORPUS_TERMS, # Use config
            no_above=0.5,
            keep_n=1500
        )

        # Create corpus
        corpus = [dtm_dictionary.doc2bow(tokens) for tokens in time_ordered_corpus_tokens]
        # Filter out completely empty documents from corpus if any after dictionary filtering
        corpus_time_slices_pairs = [(doc, ts_idx) for ts_idx, ts_len in enumerate(time_slices) for doc_idx, doc in enumerate(corpus[sum(time_slices[:ts_idx]):sum(time_slices[:ts_idx+1])]) if doc]
        
        if not corpus_time_slices_pairs:
            logger.warning("No valid documents in corpus for DTM after filtering.")
            return None, None, None

        # Reconstruct corpus and time_slices based on non-empty documents
        new_corpus = [pair[0] for pair in corpus_time_slices_pairs]
        # This time_slices reconstruction is tricky if documents are removed.
        # For simplicity, we'll proceed if the filtering isn't too aggressive.
        # A more robust way would be to re-calculate time_slices or ensure DTM handles empty docs in slices.
        # For now, we assume initial time_slices are okay if corpus is not drastically reduced.
        if len(new_corpus) < sum(time_slices) * 0.8: # Heuristic: if more than 20% docs removed
            logger.warning("Significant number of documents removed after dictionary filtering, DTM might be unstable.")
        
        corpus = new_corpus


        logger.info(f"Training DTM on {len(corpus)} documents across {len(time_slices)} time periods...")

        # Train Dynamic Topic Model
        num_topics = min(20, len(valid_years) * 3)  # Reasonable number of topics

        dtm_model = LdaSeqModel(
            corpus=corpus, # Use the potentially filtered corpus
            id2word=dtm_dictionary,
            time_slice=time_slices, # Original time_slices, assuming LdaSeqModel can handle if some docs in slice are now empty
            num_topics=num_topics,
            passes=config.DTM_PASSES,
            random_state=42,
            lda_inference_max_iter=25, # Default
            em_max_iter=20, # Default
            em_min_iter=6, # Default for convergence check
            chain_variance=0.005 # Default DTM variance
        )

        logger.info(f"DTM training completed with {num_topics} topics across {len(time_slices)} time periods")

        # Log topic evolution sample
        logger.info("Sample topic evolution:")
        for topic_id_iter in range(min(3, num_topics)): # Renamed variable
            logger.info(f"  Topic {topic_id_iter} evolution:")
            for time_idx, year_val in enumerate(valid_years): # Renamed variable
                if time_idx < len(time_slices):
                    try:
                        # DTM doesn't have show_topic method, use print_topic instead
                        topic_terms = dtm_model.print_topic(topic_id_iter, time=time_idx, top_terms=5)
                        logger.info(f"    {year_val}: {topic_terms}")
                    except Exception as e:
                        logger.warning(f"    {year_val}: Error getting topic terms: {e}")
                        # Fallback: try to get terms another way
                        try:
                            terms_prob = dtm_model.show_topics(time=time_idx, topics=[topic_id_iter], topn=5, formatted=False)
                            if terms_prob and len(terms_prob) > 0:
                                terms_str = ", ".join([word for word, _ in terms_prob[0][1]])
                    logger.info(f"    {year_val}: {terms_str}")
                            else:
                                logger.info(f"    {year_val}: No terms available")
                        except Exception as e2:
                            logger.info(f"    {year_val}: Unable to retrieve topic terms")

        return dtm_model, dtm_dictionary, time_slices # Return original time_slices

    except Exception as e:
        logger.error(f"Error in dynamic topic modeling: {e}")
        return None, None, None

def initialize_qa_model(device_to_use="cpu"):
    """
    Initialize the Question Answering model for extractive QA.

    Args:
        device_to_use (str): The device to attempt to load the model on ('mps' or 'cpu').

    Returns:
        QA pipeline or None if not available
    """
    if not HF_TRANSFORMERS_AVAILABLE:
        logger.warning("Transformers library not available - QA features disabled")
        return None

    # Determine the device index for Hugging Face pipeline (0 for GPU, -1 for CPU)
    # PyTorch device strings ('mps', 'cpu') are different from pipeline device integers.
    pipeline_device_arg = 0 if device_to_use == "mps" else -1

    try:
        logger.info(f"Loading QA model: {config.QA_MODEL_NAME} on device: {device_to_use} (pipeline arg: {pipeline_device_arg})")
        qa_pipeline_model = pipeline(
            "question-answering",
            model=config.QA_MODEL_NAME,
            tokenizer=config.QA_MODEL_NAME,
            device=pipeline_device_arg # Use integer device for pipeline
            # handle_impossible_answer=True # This is often default or can be set at call time
        )
        logger.info(f"QA model loaded successfully (attempted device: {device_to_use})")
        return qa_pipeline_model
    except Exception as e:
        logger.error(f"Error loading QA model on device {device_to_use}: {e}")
        if device_to_use == "mps": # If MPS loading failed, try CPU as a fallback
            logger.warning("Retrying QA model loading on CPU as MPS failed...")
            try:
                qa_pipeline_model = pipeline(
                    "question-answering",
                    model=config.QA_MODEL_NAME,
                    tokenizer=config.QA_MODEL_NAME,
                    device=-1 # Explicitly CPU
                )
                logger.info("QA model loaded successfully on CPU (fallback)")
                return qa_pipeline_model
            except Exception as e_cpu:
                logger.error(f"Error loading QA model on CPU (fallback): {e_cpu}")
                return None
        return None

def execute_extractive_qa(letter_index, query_text, qa_pipeline_model, sentence_model, top_n=5): # qa_pipeline renamed
    """
    Perform extractive question answering on the historical letters.
    First finds relevant contexts, then extracts specific answers.

    Args:
        letter_index: List of document dictionaries
        query_text: User's question
        qa_pipeline_model: Hugging Face QA pipeline
        sentence_model: Sentence transformer for context retrieval
        top_n: Number of contexts to analyze (and max answers to return)

    Returns:
        List of answer dictionaries with context, score, and metadata
    """
    if not qa_pipeline_model or not sentence_model:
        logger.warning("QA pipeline or sentence model not available")
        return []

    try:
        logger.info(f"Processing QA query: {query_text}")

        # First, find the most relevant documents using semantic search
        logger.info(f"Processing QA query: {query_text}")
        relevant_docs = execute_semantic_search(
            letter_index, query_text, sentence_model, top_n=config.QA_TOP_K_CONTEXTS
        )

        if not relevant_docs:
            logger.warning("No relevant documents found for QA query")
            return []

        answers = []

        for doc in relevant_docs:
            # Get the full text content
                context = doc.get('full_text', '')
            
            if not context or len(context.strip()) < 50:  # Skip very short contexts
                    continue

            # Process in larger chunks for better context
            max_context_char_length = 3000  # Increased back to 3000 for more context
                if len(context) > max_context_char_length:
                # Try to find a good breaking point
                words = context.split()
                truncated_words = []
                char_count = 0

                for word in words:
                    if char_count + len(word) + 1 > max_context_char_length:
                        break
                    truncated_words.append(word)
                    char_count += len(word) + 1

                context = ' '.join(truncated_words)
            
            try:
                # Get answer from QA model
                qa_result = qa_pipeline_model(
                    question=query_text,
                    context=context,
                    max_answer_len=config.QA_MAX_ANSWER_LENGTH,
                    handle_impossible_answer=False  # Changed to False to force answers
                )

                # Extract answer details
                answer_text = qa_result.get('answer', '').strip()
                confidence_score = qa_result.get('score', 0.0)

                # Debug logging
                logger.info(f"Raw QA result for context {len(context)} chars: answer='{answer_text[:50]}...', score={confidence_score}")

                # Filter out clearly bad answers but be very lenient
                is_valid_answer = (
                    answer_text and 
                    len(answer_text.strip()) > 1 and  # At least 2 characters
                    answer_text.strip() not in ['...', '.', ',', ';', ':', '!', '?', '-'] and  # Not just punctuation
                    not answer_text.strip().isdigit() or len(answer_text.strip()) >= 4  # Accept years (4+ digits) but not single digits
                )
                
                logger.info(f"Answer validity check: '{answer_text}' -> {is_valid_answer}")

                # Accept almost all answers now - no score threshold
                if is_valid_answer:
                    answer_data = {
                        'answer': answer_text,
                        'confidence': confidence_score,
                        'context_snippet': _extract_answer_context(
                            context, answer_text, qa_result.get('start', 0)
                        ),
                        'document_title': doc.get('title', 'Unknown'),
                        'document_year': doc.get('year', 'Unknown'),
                        'document_sender': doc.get('sender', 'Unknown'),
                        'document_file': doc.get('file_name', 'Unknown'),
                        'semantic_similarity': doc.get('similarity_score', 0) # From context retrieval step
                    }
                    answers.append(answer_data)

            except Exception as e:
                logger.warning(f"Error processing QA for document {doc.get('file_name', 'unknown')}: {e}")
                continue

        # Sort answers by confidence score
        answers.sort(key=lambda x: x['confidence'], reverse=True)

        logger.info(f"QA extracted {len(answers)} answers from {len(relevant_docs)} contexts")
        return answers[:top_n]  # Return top N answers

    except Exception as e:
        logger.error(f"Error in extractive QA: {e}")
        return []

def _extract_answer_context(full_context: str, answer: str, answer_start: int, context_window: int = 200) -> str:
    """
    Extract a snippet around the answer for better context display.

    Args:
        full_context: Complete document text
        answer: Extracted answer text
        answer_start: Start position of answer in context
        context_window: Characters to include before/after answer

    Returns:
        Context snippet with answer highlighted
    """
    try:
        # Calculate snippet boundaries
        start = max(0, answer_start - context_window)
        end = min(len(full_context), answer_start + len(answer) + context_window)

        snippet = full_context[start:end]

        # Add ellipsis if truncated
        prefix = "..." if start > 0 else ""
        suffix = "..." if end < len(full_context) else ""
        
        snippet = prefix + snippet + suffix

        # Highlight the answer in the snippet (case-insensitive replace for robustness if casing differs slightly)
        # This is tricky if answer casing in snippet is different. A regex might be better.
        # For simplicity, direct replace first.
        try:
            # Attempt to find the exact answer string to replace
            answer_in_snippet_start = snippet.lower().find(answer.lower())
            if answer_in_snippet_start != -1:
                original_answer_text_in_snippet = snippet[answer_in_snippet_start : answer_in_snippet_start + len(answer)]
                snippet = snippet.replace(original_answer_text_in_snippet, f"**{original_answer_text_in_snippet}**", 1)
            else: # Fallback if answer not found as a whole (e.g. due to truncation or ellipsis)
                 snippet = snippet.replace(answer, f"**{answer}**",1) # Try exact match
        except Exception: # If replace fails, just return snippet
            pass


        return snippet.strip()

    except Exception:
        # Fallback to simple answer return
        return f"...**{answer}**..."

def execute_hybrid_search(letter_index, query_text, sentence_model, top_n=10):
    """
    Execute hybrid search combining keyword (TF-IDF) and semantic search.
    Merges and re-ranks results for more comprehensive retrieval.

    Args:
        letter_index: List of document dictionaries
        query_text: Search query
        sentence_model: Sentence transformer model
        top_n: Number of results to return

    Returns:
        List of documents with hybrid relevance scores
    """
    if not sentence_model or not HF_TRANSFORMERS_AVAILABLE: # Check for TFIDF dependencies
        logger.warning("Sentence model or TFIDF dependencies not available for hybrid search")
        return []

    try:
        logger.info(f"Executing hybrid search for: {query_text}")

        # Step 1: Semantic search
        semantic_results = execute_semantic_search(
            letter_index, query_text, sentence_model,
            top_n=config.HYBRID_TOP_N
        )

        # Step 2: Keyword search using TF-IDF
        keyword_results = _execute_tfidf_search(
            letter_index, query_text,
            top_n=config.HYBRID_TOP_N
        )

        # Step 3: Merge and re-rank results
        hybrid_results = _merge_search_results(
            semantic_results, keyword_results, query_text
        )

        # Sort by combined score and return top results
        hybrid_results.sort(key=lambda x: x.get('hybrid_score', 0), reverse=True)

        logger.info(f"Hybrid search returned {len(hybrid_results)} results (before final top_n)")
        return hybrid_results[:top_n]

    except Exception as e:
        logger.error(f"Error in hybrid search: {e}")
        return []

def _execute_tfidf_search(letter_index, query_text, top_n=10):
    """
    Execute TF-IDF based keyword search on the document collection.

    Args:
        letter_index: List of document dictionaries
        query_text: Search query
        top_n: Number of results to return

    Returns:
        List of documents with TF-IDF scores
    """
    try:
        # Prepare document texts
        documents = []
        doc_indices_map = {} # To map vectorizer index back to letter_index

        for i, doc in enumerate(letter_index):
            # Combine title, description, and full text
            text_parts = [
                doc.get('title', ''),
                doc.get('description', ''),
                doc.get('full_text', '')
            ]
            combined_text = ' '.join(part for part in text_parts if part and part != 'N/A')

            if combined_text.strip():
                documents.append(combined_text)
                doc_indices_map[len(documents)-1] = i # Store original index

        if not documents:
            return []

        # Create TF-IDF vectorizer
        vectorizer = TfidfVectorizer(
            max_features=config.TFIDF_MAX_FEATURES,
            min_df=config.TFIDF_MIN_DF,
            max_df=config.TFIDF_MAX_DF,
            stop_words='english',
            ngram_range=(1, 2),  # Include bigrams
            lowercase=True
        )

        # Fit and transform documents
        tfidf_matrix = vectorizer.fit_transform(documents)

        # Transform query
        query_vector = vectorizer.transform([query_text])

        # Calculate cosine similarity
        similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()

        # Get top results
        # argsort returns indices that would sort the array. We want top N largest.
        num_results_to_consider = min(top_n, len(similarities))
        top_vectorizer_indices = similarities.argsort()[-num_results_to_consider:][::-1]


        results = []
        for vec_idx in top_vectorizer_indices:
            if similarities[vec_idx] > 0:  # Only include positive scores
                original_doc_idx = doc_indices_map[vec_idx]
                doc = letter_index[original_doc_idx].copy()
                doc['tfidf_score'] = float(similarities[vec_idx])
                results.append(doc)

        return results

    except Exception as e:
        logger.error(f"Error in TF-IDF search: {e}")
        return []

def _merge_search_results(semantic_results, keyword_results, query_text): # query_text not used here currently
    """
    Merge semantic and keyword search results with intelligent scoring.

    Args:
        semantic_results: Results from semantic search
        keyword_results: Results from TF-IDF search
        query_text: Original query for additional scoring (currently unused)

    Returns:
        List of merged results with hybrid scores
    """
    # Create dictionaries for easy lookup using a unique identifier (file_name)
    # Ensure all results have 'file_name'
    semantic_dict = {doc['file_name']: doc for doc in semantic_results if 'file_name' in doc}
    keyword_dict = {doc['file_name']: doc for doc in keyword_results if 'file_name' in doc}


    # Get all unique documents based on file_name
    all_files = set(semantic_dict.keys()) | set(keyword_dict.keys())

    merged_results = []

    for file_name in all_files:
        # Get scores from both searches
        semantic_score = 0.0 # Default to float
        keyword_score = 0.0  # Default to float

        # Determine the base document to copy attributes from
        # Prefer semantic result if available, else keyword, else create minimal
        if file_name in semantic_dict:
            base_doc = semantic_dict[file_name].copy()
            semantic_score = base_doc.get('similarity_score', 0.0)
        elif file_name in keyword_dict:
            base_doc = keyword_dict[file_name].copy()
        else:
            # This case should ideally not happen if all_files is derived correctly
            # and input results have 'file_name'.
            # However, to be safe, create a minimal doc.
            base_doc = {'file_name': file_name}


        if file_name in keyword_dict:
            keyword_score = keyword_dict[file_name].get('tfidf_score', 0.0)
            # If base_doc came from semantic, update it with keyword score if not present
            if 'tfidf_score' not in base_doc:
                 base_doc['tfidf_score'] = keyword_score


        # Calculate hybrid score
        hybrid_score = (
            config.HYBRID_SEMANTIC_WEIGHT * semantic_score +
            config.HYBRID_KEYWORD_WEIGHT * keyword_score
        )

        # Add scoring metadata
        base_doc['semantic_score'] = semantic_score # Ensure it's there
        base_doc['keyword_score'] = keyword_score   # Ensure it's there
        base_doc['hybrid_score'] = hybrid_score
        base_doc['search_type_contribution'] = _determine_search_type(semantic_score, keyword_score) # Renamed key

        merged_results.append(base_doc)

    return merged_results

def _determine_search_type(semantic_score, keyword_score):
    """Determine which search type contributed most to the result."""
    # Normalize scores if they are on different scales - assuming they are somewhat comparable [0,1] for now
    # This is a simple heuristic
    if semantic_score > 0 and keyword_score <= 0.01: # Primarily semantic if keyword score is negligible
        return "Semantic"
    if keyword_score > 0 and semantic_score <= 0.01: # Primarily keyword if semantic score is negligible
        return "Keyword"

    # More nuanced comparison if both have scores
    if semantic_score > keyword_score * 1.5 : # Significantly more semantic
        return "Primarily Semantic"
    elif keyword_score > semantic_score * 1.5: # Significantly more keyword
        return "Primarily Keyword"
    elif semantic_score > 0 and keyword_score > 0 : # Both contributed significantly
        return "Hybrid Match"
    elif semantic_score > 0 : # Only semantic had a score
        return "Semantic"
    elif keyword_score > 0 : # Only keyword had a score
        return "Keyword"
    else:
        return "Undetermined"


# --- Phase 2: Advanced Search Functions ---

def search_hierarchical_topics(index: List[Dict[str, Any]], keyword: str) -> List[Dict[str, Any]]:
    """Search using hierarchical topic model."""
    if not global_hdp_model or not global_hdp_dictionary: # Check dictionary too
        print("Hierarchical topic model or dictionary not available.")
        return []

    try:
        # Find topics containing the keyword
        # HDP model's show_topics can be slow if called repeatedly or for many topics
        # This is a conceptual search. Real HDP search might involve assigning topics to query.
        all_hdp_topics = global_hdp_model.show_topics(num_topics=-1, num_words=10, formatted=False)
        active_hdp_topics = [t for t in all_hdp_topics if t[1] and len(t[1]) > 0]


        matching_topic_ids = []
        keyword_lower = keyword.lower()

        for topic_id, words_probs in active_hdp_topics:
            topic_words = [word for word, prob in words_probs]
            if any(keyword_lower in word.lower() for word in topic_words):
                matching_topic_ids.append(topic_id)

        if not matching_topic_ids:
            print(f"No hierarchical topics found containing '{keyword}'")
            return []

        print(f"Found {len(matching_topic_ids)} hierarchical topics containing '{keyword}':")
        for topic_id in matching_topic_ids[:5]:  # Show top 5 matching topics
            # Find the specific topic from active_hdp_topics to display its words
            display_words = "N/A"
            for tid_lookup, words_probs_lookup in active_hdp_topics:
                if tid_lookup == topic_id:
                    display_words = ", ".join([word for word, _ in words_probs_lookup[:8]])
                    break
            print(f"  H-Topic {topic_id}: {display_words}")
        
        # Placeholder for document retrieval:
        # To retrieve documents, you'd typically get HDP topic distributions for each document
        # and then filter documents that have a high probability for the matching_topic_ids.
        # This is not implemented here for brevity as it requires storing HDP topic distributions per doc.
        print("\n(Document retrieval based on HDP topics is not fully implemented in this example)")
        print("The following are documents that simply contain the keyword, for illustrative purposes:")

        # Fallback to simple keyword search for now to show *some* documents
        results = []
        for item in index:
            text_to_search = (item.get('title', '') + " " + item.get('full_text', '')).lower()
            if keyword_lower in text_to_search:
                results.append(item)
        
        display_results(results[:5]) # Show a few keyword matches
        return results[:5] # Return some results

    except Exception as e:
        logger.error(f"Error in hierarchical topic search: {e}")
        return []

def search_dynamic_topics_by_year(index: List[Dict[str, Any]], year: int, keyword: str) -> List[Dict[str, Any]]:
    """Search dynamic topics for a specific year and keyword."""
    if not global_dtm_model or not global_time_slices or not global_dtm_dictionary: # Check dictionary too
        print("Dynamic topic model, time slices, or dictionary not available.")
        return []

    try:
        print(f"Dynamic topic search for year {year} with keyword '{keyword}':")
        
        # Find the time slice index corresponding to the given year
        # This assumes global_time_slices corresponds to an ordered list of document counts per unique year used in DTM
        # And that we have a way to map 'year' to the 'time_idx' for dtm_model.show_topic
        # This part is complex as DTM's `time_slice` is just counts, not the years themselves.
        # We need the original `valid_years` list used during DTM training.
        # For this example, we'll assume `global_time_slices` implies an ordering that can be mapped.
        # A more robust solution would store the year mapping with the DTM model.

        # Find time_idx for the given year - this is a placeholder logic
        # You would need the actual list of years that correspond to each slice.
        # Let's assume `valid_years` from DTM training was implicitly stored or can be inferred.
        # For this demo, we'll iterate through time slices and check topics.
        
        print(f"Searching topics for year {year} containing '{keyword.lower()}':")
        found_topics_for_year = False
        
        # This is a conceptual search. We need to find which `time_idx` corresponds to `year`.
        # Let's assume `global_dtm_model.idx_to_year_map` existed (it doesn't by default).
        # For now, we'll iterate time slices and if keyword matches topic, state it.
        num_dtm_topics = global_dtm_model.num_topics
        num_dtm_time_slices = len(global_dtm_model.time_slice) # Use model's internal time_slice count

        matching_topic_details_for_year = []

        for time_idx in range(num_dtm_time_slices):
            # Here, you'd need to map time_idx back to a year or year range.
            # For this example, we can't directly filter by the input `year` without that mapping.
            # We'll show topics for *all* periods that contain the keyword.
            for topic_id in range(num_dtm_topics):
                try:
                    terms_probs = global_dtm_model.show_topic(topic_id, time=time_idx, topn=10)
                    terms = [term for term, prob in terms_probs]
                    if any(keyword.lower() in term.lower() for term in terms):
                        # If we had a year mapping for time_idx and it matched input `year`:
                        # matching_topic_details_for_year.append(...)
                        # For now, just print if found in any period for illustration
                        terms_str = ", ".join(terms[:5])
                        print(f"  - Keyword '{keyword}' found in Topic {topic_id} (Period {time_idx}): {terms_str}")
                        found_topics_for_year = True
                except IndexError: # If time_idx is out of bounds for a topic
                    continue
        
        if not found_topics_for_year:
            print(f"No dynamic topics found containing '{keyword.lower()}' across available time periods.")
            return []

        print("\n(Document retrieval for specific year-topic matches in DTM is complex and not fully implemented here.)")
        print("Showing general keyword matches in documents from the specified year as an illustration:")
        
        results_for_year = []
        for item in index:
            if item.get('year') == year:
                text_to_search = (item.get('title', '') + " " + item.get('full_text', '')).lower()
                if keyword.lower() in text_to_search:
                    results_for_year.append(item)
        
        display_results(results_for_year[:5])
        return results_for_year[:5] # Return some results

    except Exception as e:
        logger.error(f"Error in dynamic topic search: {e}")
        return []

def list_hierarchical_topics(hdp_model_global, top_n_terms: int = 8) -> None: # Parameter renamed
    """List available hierarchical topics."""
    if not hdp_model_global: # Use the parameter
        print("Hierarchical topic model not loaded.")
        return

    try:
        topics = hdp_model_global.show_topics(num_topics=-1, num_words=top_n_terms, formatted=False)
        active_topics = [t for t in topics if t[1] and len(t[1]) > 0]

        print(f"\n--- Hierarchical Topics ({len(active_topics)} discovered) ---")
        for i, (topic_id, words) in enumerate(active_topics[:15]):  # Show top 15
            terms = ", ".join([word for word, _ in words[:top_n_terms]])
            print(f"H-Topic {topic_id:02}: {terms}")

        if len(active_topics) > 15:
            print(f"... and {len(active_topics) - 15} more topics")
        print("---")

    except Exception as e:
        logger.error(f"Error listing hierarchical topics: {e}")

def list_dynamic_topic_evolution(dtm_model_global, time_slices_global, top_n_terms: int = 5) -> None: # Parameters renamed
    """Show how topics evolved over time."""
    if not dtm_model_global or not time_slices_global: # Use parameters
        print("Dynamic topic model or time slices not loaded.")
        return

    try:
        num_topics = dtm_model_global.num_topics
        num_time_slices = len(time_slices_global) # Use parameter

        print(f"\n--- Dynamic Topic Evolution ({num_topics} topics, {num_time_slices} time periods) ---")
        # Note: `time_slices_global` here is the list of doc counts per period.
        # We need a list of actual years for display if available from DTM training.
        # Assuming an ordered sequence of periods for now.

        topics_to_show = min(5, num_topics)
        for topic_id in range(topics_to_show):
            print(f"\nTopic {topic_id} Evolution:")
            for time_idx in range(min(5, num_time_slices)):  # Show first 5 time periods
                try:
                    terms = dtm_model_global.show_topic(topic_id, time=time_idx, topn=top_n_terms)
                    terms_str = ", ".join([word for word, _ in terms])
                    # Ideally, map time_idx to actual year here if that mapping is stored
                    print(f"  Period {time_idx}: {terms_str}")
                except IndexError: # Should not happen if time_idx is < num_time_slices
                    print(f"  Period {time_idx}: (Error retrieving topic for this period)")
                except Exception as e_topic:
                    print(f"  Period {time_idx}: (Error: {e_topic})")


        if num_topics > topics_to_show:
            print(f"\n... and {num_topics - topics_to_show} more topics with evolution data")
        print("---")

    except Exception as e:
        logger.error(f"Error showing topic evolution: {e}")

def display_qa_results(answers):
    """Display results from extractive QA."""
    if not answers:
        print("No answers found.")
        return

    print(f"\n--- Found {len(answers)} Answer(s) ---")

    for i, answer_item in enumerate(answers, 1): # Renamed variable
        print(f"\nAnswer {i}:")
        print(f"  Answer:     {answer_item.get('answer', 'N/A')}") # Corrected spacing
        print(f"  Confidence: {answer_item.get('confidence', 0):.3f}")
        print(f"  Source:     {answer_item.get('document_title', 'Unknown')} ({answer_item.get('document_year', 'Unknown')})")
        print(f"  From:       {answer_item.get('document_sender', 'Unknown')}") # Corrected spacing
        print(f"  File:       {answer_item.get('document_file', 'Unknown')}") # Corrected spacing

        context = answer_item.get('context_snippet', '')
        if context:
            # Limit context display length
            if len(context) > 300:
                context = context[:300] + "..."
            print(f"  Context:    {context}") # Corrected spacing

    print("-" * 30)

def display_hybrid_results(results):
    """Display results from hybrid search with score breakdown."""
    if not results:
        print("No matching letters found.")
        return

    print(f"\n--- Found {len(results)} Matching Letter(s) (Hybrid Search) ---")

    for i, item in enumerate(results, 1):
        print(f"\nResult {i}:")
        print(f"  File:         {item.get('file_name', 'N/A')}") # Corrected spacing
        print(f"  Title:        {item.get('title', 'N/A')}") # Corrected spacing
        print(f"  Date:         {item.get('main_date', 'N/A')}") # Corrected spacing
        print(f"  Sender:       {item.get('sender', 'N/A')}") # Corrected spacing

        # Show hybrid scoring breakdown
        hybrid_score = item.get('hybrid_score', 0)
        semantic_score = item.get('semantic_score', 0)
        keyword_score = item.get('keyword_score', 0)
        search_type_contrib = item.get('search_type_contribution', 'Unknown') # Matched key from _merge_search_results

        print(f"  Hybrid Score: {hybrid_score:.3f} ({search_type_contrib})")
        print(f"    Semantic:   {semantic_score:.3f}") # Corrected spacing
        print(f"    Keyword:    {keyword_score:.3f}") # Corrected spacing

        # Show snippet
        text_snippet = item.get('full_text', 'N/A')
        if text_snippet != 'N/A' and text_snippet:
            snippet = (text_snippet[:200] + '...') if len(text_snippet) > 200 else text_snippet
            # Fix f-string syntax by moving regex outside
            cleaned_snippet = re.sub(r'\s+', ' ', snippet).strip()
            print(f"  Snippet:      {cleaned_snippet}")

    print("-" * 30)

# --- Main Entry Point ---
def main():
    """
    Main entry point for the command-line interface.
    """
    parser = argparse.ArgumentParser(
        description="AI-Powered Historical Document Analyzer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create index with optimizations
  %(prog)s --create-index --xmldir xmlfiles --workers 6 --batch-size 12 --checkpoint-interval 500 --num-topics 30

  # Force full re-index
  %(prog)s --create-index --xmldir xmlfiles --force-reindex --verbose

  # Create index with sampling for development
  %(prog)s --create-index --xmldir xmlfiles --sample-size 500 --enable-topic-sampling

  # Run interactive chat interface
  %(prog)s --xmldir xmlfiles --interactive

  # Check index statistics
  %(prog)s --stats
        """
    )
    
    # Mode selection
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument('--create-index', action='store_true',
                           help='Create or update the document index')
    mode_group.add_argument('--interactive', action='store_true',
                           help='Run interactive chat interface')
    mode_group.add_argument('--stats', action='store_true',
                           help='Show index statistics')
    
    # XML source configuration
    parser.add_argument('--xmldir', type=str, default='xmlfiles',
                       help='Directory containing XML files (default: xmlfiles)')
    parser.add_argument('--files', nargs='+', type=str,
                       help='Specific XML files to process')
    
    # Index management
    parser.add_argument('--index-path', type=str, default=config.SAVED_INDEX_FILE,
                       help=f'Path to save/load index (default: {config.SAVED_INDEX_FILE})')
    parser.add_argument('--force-reindex', action='store_true',
                       help='Force complete re-indexing, ignore existing data')
    parser.add_argument('--no-incremental', action='store_true',
                       help='Disable incremental updates')
    
    # Performance optimization settings
    parser.add_argument('--workers', type=int, default=config.MAX_WORKERS,
                       help=f'Number of parallel workers (default: {config.MAX_WORKERS})')
    parser.add_argument('--batch-size', type=int, default=config.OPTIMAL_BATCH_SIZE_EMBEDDING,
                       help=f'Batch size for embedding generation (default: {config.OPTIMAL_BATCH_SIZE_EMBEDDING})')
    parser.add_argument('--checkpoint-interval', type=int, default=config.EMBEDDING_CHECKPOINT_INTERVAL,
                       help=f'Save checkpoints every N documents (default: {config.EMBEDDING_CHECKPOINT_INTERVAL})')
    
    # Topic modeling settings
    parser.add_argument('--num-topics', type=int, default=config.NUM_TOPICS,
                       help=f'Number of topics for LDA (default: {config.NUM_TOPICS})')
    parser.add_argument('--sample-size', type=int,
                       help='Use subset of documents for development (for topic training)')
    parser.add_argument('--enable-topic-sampling', action='store_true',
                       help='Enable topic model sampling for faster development iteration')
    
    # Advanced features
    parser.add_argument('--enable-hierarchical', action='store_true',
                       help='Enable hierarchical topic modeling (HDP)')
    parser.add_argument('--enable-dynamic', action='store_true',
                       help='Enable dynamic topic modeling over time')
    parser.add_argument('--enable-qa', action='store_true',
                       help='Enable question answering pipeline')
    
    # Logging and output
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')
    parser.add_argument('--profile', action='store_true',
                       help='Enable performance profiling')
    parser.add_argument('--log-file', type=str, default='chat_analyzer.log',
                       help='Log file path (default: chat_analyzer.log)')
    
    args = parser.parse_args()

    # Configure logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.info("Verbose logging enabled")
    
    # Update configuration based on arguments
    config.MAX_WORKERS = args.workers
    config.OPTIMAL_BATCH_SIZE_EMBEDDING = args.batch_size
    config.EMBEDDING_CHECKPOINT_INTERVAL = args.checkpoint_interval
    config.NUM_TOPICS = args.num_topics
    
    if args.sample_size:
        config.SAMPLE_SIZE_FOR_TOPIC_TRAINING = args.sample_size
    if args.enable_topic_sampling:
        config.ENABLE_TOPIC_SAMPLING = True
    
    logger.info(f"Configuration: workers={config.MAX_WORKERS}, batch_size={config.OPTIMAL_BATCH_SIZE_EMBEDDING}, topics={config.NUM_TOPICS}")
    
    # Performance profiling setup
    if args.profile:
        import cProfile
        import pstats
        profiler = cProfile.Profile()
        profiler.enable()
    
    try:
        start_time = time.time()
        
        if args.stats:
            # Show index statistics
            show_index_statistics(args.index_path)
            
        elif args.create_index:
            # Create or update index
            create_index_cli(args)
            
        elif args.interactive:
            # Run interactive interface
            run_interactive_cli(args)
            
        elapsed_time = time.time() - start_time
        logger.info(f"Total execution time: {elapsed_time:.2f} seconds")
        
    except KeyboardInterrupt:
        logger.info("Operation interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        if args.verbose:
            logger.error(traceback.format_exc())
        return 1
    finally:
        if args.profile:
            profiler.disable()
            # Save profiling results
            profile_file = f"profile_{datetime.now().strftime('%Y%m%d_%H%M%S')}.prof"
            profiler.dump_stats(profile_file)
            logger.info(f"Profiling data saved to {profile_file}")
            
            # Print top time consumers
            stats = pstats.Stats(profiler)
            stats.sort_stats('cumulative')
            stats.print_stats(10)
    
    return 0

def show_index_statistics(index_path: str):
    """Show statistics about the existing index."""
    if not os.path.exists(index_path):
        logger.error(f"Index file not found: {index_path}")
        return
    
    try:
        with open(index_path, 'rb') as f:
            comprehensive_index = pickle.load(f)
        
        if isinstance(comprehensive_index, dict) and 'documents' in comprehensive_index:
            # New format with metadata
            documents = comprehensive_index['documents']
            metadata = comprehensive_index.get('metadata', {})
            
            print(f"\n📊 Index Statistics for {index_path}")
            print(f"📁 Index file size: {os.path.getsize(index_path) / (1024*1024):.1f} MB")
            print(f"🕒 Last modified: {datetime.fromtimestamp(os.path.getmtime(index_path))}")
            print(f"\n📋 Document Statistics:")
            print(f"   Total documents: {metadata.get('total_documents', len(documents))}")
            print(f"   Successful embeddings: {metadata.get('successful_embeddings', 'Unknown')}")
            print(f"   Documents with topics: {metadata.get('documents_with_topics', 'Unknown')}")
            print(f"   Topics available: {metadata.get('topics_available', 'Unknown')}")
            
            if metadata.get('processing_time_seconds'):
                print(f"   Processing time: {metadata['processing_time_seconds']:.2f} seconds")
            
            # Analyze years
            years = [doc.get('year') for doc in documents if doc.get('year')]
            if years:
                print(f"\n📅 Time Period Coverage:")
                print(f"   Year range: {min(years)} - {max(years)}")
                print(f"   Documents with dates: {len(years)}")
            
            # Analyze senders
            senders = [doc.get('sender') for doc in documents if doc.get('sender') and doc.get('sender') != 'Unknown']
            if senders:
                from collections import Counter
                top_senders = Counter(senders).most_common(5)
                print(f"\n✍️ Top Senders:")
                for sender, count in top_senders:
                    print(f"   {sender}: {count} letters")
                    
        else:
            # Legacy format
            documents = comprehensive_index if isinstance(comprehensive_index, list) else []
            print(f"\n📊 Index Statistics (Legacy Format)")
            print(f"📁 Index file size: {os.path.getsize(index_path) / (1024*1024):.1f} MB")
            print(f"📋 Total documents: {len(documents)}")
            
    except Exception as e:
        logger.error(f"Error reading index statistics: {e}")

def create_index_cli(args):
    """Create or update the document index via CLI."""
    logger.info("Starting index creation/update...")
    
    # Get list of files to process
    xml_files_to_process = get_file_list_from_args(args)
    
    if not xml_files_to_process:
        logger.error("No XML files found to process")
        return
    
    logger.info(f"Found {len(xml_files_to_process)} XML files to process")
    
    # Load NLP models
    logger.info("Loading NLP models...")
    spacy_nlp_model, sentence_model, qa_pipeline = load_nlp_models()
    
    if not sentence_model:
        logger.error("Failed to load sentence transformer model")
        return
    
    # Create or update index
    incremental = not args.no_incremental
    letter_index = load_or_create_index(
        xml_files_to_process,
        sentence_model,
        spacy_nlp_model,
        args.index_path,
        force_reindex=args.force_reindex,
        incremental=incremental
    )
    
    if letter_index:
        logger.info(f"Index creation completed successfully with {len(letter_index)} documents")
        
        # Load advanced models if requested
        if args.enable_qa:
            logger.info("QA model already loaded during initialization")
            global global_qa_pipeline
            global_qa_pipeline = qa_pipeline
            
        # Show basic statistics
        show_index_statistics(args.index_path)
    else:
        logger.error("Index creation failed")

def run_interactive_cli(args):
    """Run the interactive chat interface via CLI."""
    logger.info("Starting interactive chat interface...")
    
    # Load existing index
    if not os.path.exists(args.index_path):
        logger.error(f"Index file not found: {args.index_path}. Please run --create-index first.")
        return
    
    # Get list of files (for potential updates)
    xml_files_to_process = get_file_list_from_args(args)
    
    # Load NLP models
    spacy_nlp_model, sentence_model, qa_pipeline = load_nlp_models()
    
    # Load index
    letter_index = load_or_create_index(
        xml_files_to_process,
        sentence_model,
        spacy_nlp_model,
        args.index_path,
        force_reindex=False,
        incremental=True
    )
    
    if not letter_index:
        logger.error("Failed to load index")
        return
    
    # Initialize advanced models if requested
    if args.enable_qa:
        logger.info("QA model already loaded during initialization")
        global global_qa_pipeline
        global_qa_pipeline = qa_pipeline
    
    # Run chatbot
    run_chatbot(letter_index, spacy_nlp_model, sentence_model)

# Import the modern topic modeling
try:
    from modern_topics import replace_slow_topic_modeling
    MODERN_TOPICS_AVAILABLE = True
except ImportError:
    MODERN_TOPICS_AVAILABLE = False

# Global variable to store results of modern topic modeling for app access
global_modern_topic_results: Optional[Dict[str, Any]] = None

def run_modern_topic_discovery(current_letter_index: List[Dict[str, Any]]):
    """
    Runs the modern topic discovery on a loaded index and stores results globally.
    This is intended to be called by the Streamlit app after loading an existing index.
    """
    global global_modern_topic_results, global_lda_model # We can reuse global_lda_model for the new model object

    if not MODERN_TOPICS_AVAILABLE:
        logger.warning("Modern topic modeling module not available. Skipping.")
        global_modern_topic_results = None
        global_lda_model = None
        return

    if not current_letter_index:
        logger.warning("Letter index is empty. Skipping modern topic discovery.")
        global_modern_topic_results = None
        global_lda_model = None
        return

    logger.info(f"Running modern topic discovery on {len(current_letter_index)} loaded documents...")
    try:
        # The replace_slow_topic_modeling function returns (model_object, results_dict)
        model_object, results_dict = replace_slow_topic_modeling(current_letter_index, config) # Pass the global config
        
        if model_object and results_dict:
            global_modern_topic_results = results_dict
            global_lda_model = model_object # Store the new model object here for compatibility
            
            # Assign topic_id and topic_name to documents in the main index
            # The 'document_topics' in results_dict is a list of (doc_identifier, topic_id)
            # The 'topics' in results_dict is a dict of topic_id -> {name, keywords}
            
            # Create a mapping from file_name to document for efficient update
            doc_map = {doc.get('file_name'): doc for doc in current_letter_index if doc.get('file_name')}

            if 'document_topics' in results_dict and 'topics' in results_dict:
                for doc_identifier, topic_id in results_dict['document_topics']:
                    # Assuming doc_identifier is file_name, which was used in modern_topics.py
                    if doc_identifier in doc_map:
                        doc_map[doc_identifier]['topic_id'] = topic_id
                        topic_info = results_dict['topics'].get(topic_id)
                        if topic_info:
                            doc_map[doc_identifier]['topic_name'] = topic_info.get('name', f'Topic {topic_id}')
                            doc_map[doc_identifier]['topic_terms'] = topic_info.get('keywords', [])
            
            logger.info(f"Modern topic discovery successful. Found {model_object.num_topics} topics. Results stored globally.")
            logger.info(f"Sample modern topics: {list(results_dict.get('topics', {}).values())[:2]}")

    else:
            logger.error("Modern topic discovery did not return expected model or results.")
            global_modern_topic_results = None
            global_lda_model = None

    except Exception as e:
        logger.error(f"Error during modern topic discovery: {e}", exc_info=True)
        global_modern_topic_results = None
        global_lda_model = None

# Add geocoding imports
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderServiceError

# Geocoding functionality for map integration
def geocode_locations(place_names: List[str], delay_seconds: float = 1.1) -> List[Dict[str, Any]]:
    """
    Convert place names to latitude/longitude coordinates using OpenStreetMap/Nominatim.
    
    Args:
        place_names: List of place names to geocode
        delay_seconds: Delay between API calls to respect Nominatim usage policy
        
    Returns:
        List of dictionaries with geocoding results
    """
    if not place_names:
        return []
    
    geocoded_results = []
    
    try:
        # Initialize geocoder with proper user agent
        geolocator = Nominatim(user_agent="CivilWarLetterAnalyzer/1.0")
        
        logger.info(f"Starting geocoding for {len(place_names)} places...")
        
        for i, place_name in enumerate(place_names):
            if not place_name or not place_name.strip():
                continue
                
            try:
                # Add delay to respect Nominatim usage policy (max 1 request/second)
                if i > 0:  # Don't delay before the first request
                    time.sleep(delay_seconds)
                
                logger.debug(f"Geocoding place {i+1}/{len(place_names)}: '{place_name}'")
                
                # Perform geocoding with timeout
                location = geolocator.geocode(place_name, timeout=10)
                
                if location:
                    result = {
                        'name': place_name,
                        'lat': location.latitude,
                        'lon': location.longitude,
                        'raw_address': location.address,
                        'geocoded': True
                    }
                    logger.debug(f"Successfully geocoded '{place_name}' to ({location.latitude}, {location.longitude})")
                else:
                    result = {
                        'name': place_name,
                        'lat': None,
                        'lon': None,
                        'raw_address': None,
                        'geocoded': False
                    }
                    logger.warning(f"Could not geocode location: '{place_name}'")
                
                geocoded_results.append(result)
                
            except (GeocoderTimedOut, GeocoderServiceError) as e:
                logger.warning(f"Geocoding error for '{place_name}': {e}")
                # Add failed result to maintain consistency
                geocoded_results.append({
                    'name': place_name,
                    'lat': None,
                    'lon': None,
                    'raw_address': None,
                    'geocoded': False,
                    'error': str(e)
                })
                
            except Exception as e:
                logger.error(f"Unexpected error geocoding '{place_name}': {e}")
                geocoded_results.append({
                    'name': place_name,
                    'lat': None,
                    'lon': None,
                    'raw_address': None,
                    'geocoded': False,
                    'error': str(e)
                })
        
        successful_geocodes = sum(1 for r in geocoded_results if r.get('geocoded', False))
        logger.info(f"Geocoding completed: {successful_geocodes}/{len(geocoded_results)} places successfully geocoded")
        
        return geocoded_results
        
    except Exception as e:
        logger.error(f"Fatal error in geocoding process: {e}")
        return []

# Add a migration function for existing indices
def migrate_index_with_geocoding(letter_index: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Migrate existing index by adding geocoding to documents that don't have it.
    
    Args:
        letter_index: Existing letter index
        
    Returns:
        Updated letter index with geocoded places
    """
    logger.info("Starting geocoding migration for existing index...")
    
    docs_updated = 0
    docs_with_places = 0
    total_new_geocodes = 0
    
    for doc in letter_index:
        # Skip if already has geocoded places
        if doc.get('geocoded_places'):
            continue
            
        places = doc.get('places', [])
        if places:
            docs_with_places += 1
            try:
                logger.info(f"Geocoding places for document: {doc.get('file_name', 'unknown')}")
                geocoded_places = geocode_locations(places)
                doc['geocoded_places'] = geocoded_places
                
                successful_geocodes = sum(1 for place in geocoded_places if place.get('geocoded', False))
                total_new_geocodes += successful_geocodes
                docs_updated += 1
                
            except Exception as e:
                logger.error(f"Error geocoding places for document {doc.get('file_name', 'unknown')}: {e}")
                # Set empty geocoded_places to indicate attempt was made
                doc['geocoded_places'] = []
        else:
            # No places to geocode
            doc['geocoded_places'] = []
    
    logger.info(f"Geocoding migration completed:")
    logger.info(f"  - Documents processed: {docs_updated}")
    logger.info(f"  - Documents with places: {docs_with_places}")
    logger.info(f"  - Total new geocodes: {total_new_geocodes}")
    
    return letter_index

def add_geocoding_to_existing_index(current_letter_index: List[Dict[str, Any]]):
    """
    Helper function to add geocoding to an existing index and save it.
    This function is called from the Streamlit UI.
    
    Args:
        current_letter_index: The current letter index to migrate
    """
    if not current_letter_index:
        logger.warning("Cannot add geocoding: letter index is empty")
        return False
    
    try:
        logger.info(f"Starting geocoding migration for {len(current_letter_index)} documents...")
        
        # Perform the migration
        migrated_index = migrate_index_with_geocoding(current_letter_index)
        
        # Save the updated index
        logger.info("Saving updated index with geocoding...")
        with gzip.open(SAVED_INDEX_FILE, 'wb') as f:
            pickle.dump(migrated_index, f)
        
        logger.info(f"Successfully saved updated index to {SAVED_INDEX_FILE}")
        
        # Update the current index in place
        current_letter_index.clear()
        current_letter_index.extend(migrated_index)
        
        return True
        
    except Exception as e:
        logger.error(f"Error during geocoding migration: {e}")
        return False

if __name__ == "__main__":
    sys.exit(main())
