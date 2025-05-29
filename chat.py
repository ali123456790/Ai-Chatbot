import os
# Prevent HuggingFace tokenizers warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import xml.etree.ElementTree as ET
import re
import glob
import argparse
import pickle
import logging
from typing import Optional, List, Dict, Any, Tuple
from pathlib import Path
import time

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
    NUM_TOPICS = 35  # Optimized base number for n-gram enhanced topics

    # Default list of known uploaded files (for testing)
    DEFAULT_UPLOADED_FILES = [
        "42251.xml", "42252.xml", "42253.xml", "42254.xml",
        "42255.xml", "42256.xml", "42257.xml", "42258.xml", "42259.xml"
    ]

    # Model for sentence embeddings
    SENTENCE_MODEL_NAME = 'all-MiniLM-L6-v2'  # A good general-purpose model

    # File to save/load the processed index
    SAVED_INDEX_FILE = "letter_index.pkl"

    # XML parsing constants
    TEI_NAMESPACE = {'tei': 'http://www.tei-c.org/ns/1.0'}

    # Enhanced performance settings for historical document analysis with n-grams
    BATCH_SIZE_EMBEDDING = 32
    MAX_TOKENS_FOR_GENSIM = 2000  # Optimized for n-gram processing
    PROGRESS_REPORT_INTERVAL = 10

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
    QA_MODEL_NAME = 'distilbert-base-cased-distilled-squad'  # Pre-trained QA model
    QA_MAX_ANSWER_LENGTH = 512  # Maximum length of extracted answers
    QA_MIN_SCORE_THRESHOLD = 0.1  # Minimum confidence score for answers
    QA_TOP_K_CONTEXTS = 3  # Number of top contexts to search for answers

    # Hybrid Search Settings
    TFIDF_MAX_FEATURES = 5000  # Maximum features for TF-IDF vectorizer
    TFIDF_MIN_DF = 2  # Minimum document frequency for TF-IDF
    TFIDF_MAX_DF = 0.8  # Maximum document frequency for TF-IDF
    HYBRID_SEMANTIC_WEIGHT = 0.6  # Weight for semantic search in hybrid
    HYBRID_KEYWORD_WEIGHT = 0.4  # Weight for keyword search in hybrid
    HYBRID_TOP_N = 10  # Number of results to consider from each search type

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
        extracted_data['places'] = extract_places_from_xml(root, namespaces)

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

def preprocess_for_gensim(text: str, nlp=None, phrases_model=None) -> List[str]:
    """
    Enhanced text preprocessing for Gensim with n-gram support for historical content.
    Detects meaningful multi-word expressions and produces more mature topics.

    Args:
        text: Input text to preprocess
        nlp: spaCy NLP model (optional)
        phrases_model: Trained gensim Phrases model for detecting n-grams (optional)

    Returns:
        List of cleaned, meaningful tokens including n-grams
    """
    if not text or text.strip() == "N/A":
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

    try:
        if nlp and SPACY_AVAILABLE:  # Enhanced spaCy processing with n-grams
            doc = nlp(text)
            tokens = []

            for token in doc:
                # Skip if basic conditions not met
                if (token.is_stop or token.is_punct or token.like_num or
                        token.is_space or len(token.lemma_) < 3):
                    continue

                # Focus on meaningful content words
                if token.pos_ in ['NOUN', 'PROPN', 'ADJ', 'VERB']:
                    lemma = token.lemma_.lower()

                    # Skip custom historical stop words
                    if lemma in historical_stop_words:
                        continue

                    # Only keep alphabetic tokens
                    if lemma.isalpha():
                        # Prefer proper nouns and nouns for better topics
                        if token.pos_ in ['PROPN', 'NOUN']:
                            tokens.append(lemma)
                        # Include meaningful adjectives and verbs
                        elif token.pos_ in ['ADJ', 'VERB'] and len(lemma) > 4:
                            tokens.append(lemma)

            # Apply n-gram detection if phrases model is available
            if phrases_model and tokens:
                # Check if phrases_model is callable (function) or subscriptable (Phraser object)
                if callable(phrases_model):
                    tokens = phrases_model(tokens)  # Call as function
                else:
                    tokens = phrases_model[tokens]  # Use as Phraser object

            # Remove duplicates while preserving order
            seen = set()
            unique_tokens = []
            for token in tokens:
                if token not in seen:
                    seen.add(token)
                    unique_tokens.append(token)

            return unique_tokens[:config.MAX_TOKENS_FOR_GENSIM]

        else:
            # Enhanced fallback processing with basic n-gram detection
            # Remove common punctuation and normalize
            text = re.sub(r'[^\w\s]', ' ', text.lower())

            # Extract alphabetic tokens of reasonable length
            tokens = re.findall(r'\b[a-z]{4,}\b', text)  # Minimum 4 characters

            # Filter out historical stop words
            tokens = [t for t in tokens if t not in historical_stop_words]

            # Apply n-gram detection if phrases model is available
            if phrases_model and tokens:
                # Check if phrases_model is callable (function) or subscriptable (Phraser object)
                if callable(phrases_model):
                    tokens = phrases_model(tokens)  # Call as function
                else:
                    tokens = phrases_model[tokens]  # Use as Phraser object

            # Remove duplicates while preserving order
            seen = set()
            unique_tokens = []
            for token in tokens:
                if token not in seen:
                    seen.add(token)
                    unique_tokens.append(token)

            return unique_tokens[:config.MAX_TOKENS_FOR_GENSIM]

    except Exception as e:
        logger.warning(f"Error in enhanced text preprocessing with n-grams: {e}")
        # Simple fallback
        try:
            tokens = re.findall(r"\\b[a-zA-Z]{4,}\\b", text.lower())
            return [t for t in tokens if t not in historical_stop_words][:config.MAX_TOKENS_FOR_GENSIM]
        except Exception:
            return []

def generate_embeddings_batch(texts: List[str], sentence_model) -> List[Any]:
    """
    Generate embeddings in batches for better performance.

    Args:
        texts: List of texts to embed
        sentence_model: Sentence transformer model

    Returns:
        List of embeddings
    """
    if not sentence_model or not SENTENCE_TRANSFORMERS_AVAILABLE:
        return [None] * len(texts)

    embeddings = []
    batch_size = config.BATCH_SIZE_EMBEDDING

    try:
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            try:
                batch_embeddings = sentence_model.encode(
                    batch,
                    convert_to_tensor=True,
                    show_progress_bar=False
                )

                # Convert to list if it's a tensor
                if hasattr(batch_embeddings, 'cpu'):
                    batch_embeddings = [emb for emb in batch_embeddings.cpu().numpy()] # Ensure it's a list of NumPy arrays
                elif isinstance(batch_embeddings, np.ndarray) and batch_embeddings.ndim == 1: # Single embedding
                     batch_embeddings = [batch_embeddings]
                elif isinstance(batch_embeddings, np.ndarray) and batch_embeddings.ndim > 1: # Multiple embeddings already as ndarray
                     batch_embeddings = [emb for emb in batch_embeddings]


                embeddings.extend(batch_embeddings)

            except Exception as e:
                logger.error(f"Error generating embeddings for batch {i//batch_size + 1}: {e}")
                # Add None placeholders for failed batch
                embeddings.extend([None] * len(batch))

        return embeddings

    except Exception as e:
        logger.error(f"Critical error in batch embedding generation: {e}")
        return [None] * len(texts)

def train_lda_model(index: List[Dict[str, Any]], spacy_nlp_model) -> Tuple[Any, Any]:
    """
    Enhanced LDA model training with n-grams and coherence optimization for historical documents.
    Produces more mature and specific topics through n-gram detection and parameter optimization.

    Args:
        index: List of document dictionaries
        spacy_nlp_model: spaCy model for preprocessing

    Returns:
        Tuple of (lda_model, lda_dictionary)
    """
    if not GENSIM_AVAILABLE or not index:
        logger.warning("Gensim not available or no documents for topic modeling")
        return None, None

    try:
        logger.info("Starting enhanced LDA topic modeling with n-grams for historical documents...")

        # Phase 1: Initial preprocessing to build n-gram models
        logger.info("Phase 1: Building n-gram models...")
        all_token_lists = []

        for item in index:
            # Combine title, description, and full text for richer context
            combined_text = f"{item.get('title', '')} {item.get('description', '')} {item.get('full_text', '')}"
            tokens = preprocess_for_gensim(combined_text, spacy_nlp_model)  # Without phrases model first

            if tokens and len(tokens) >= 5:  # Require minimum meaningful tokens
                all_token_lists.append(tokens)

        if len(all_token_lists) < 10:
            logger.warning(f"Only {len(all_token_lists)} valid documents for n-gram modeling. Need at least 10.")
            return None, None

        # Build bigram and trigram models
        logger.info("Training bigram model...")
        bigram_model = models.Phrases(
            all_token_lists,
            min_count=config.BIGRAM_MIN_COUNT,
            threshold=config.BIGRAM_THRESHOLD,
            connector_words=frozenset(['of', 'the', 'and', 'in', 'to', 'for', 'with'])
        )
        bigram_mod = models.phrases.Phraser(bigram_model)

        logger.info("Training trigram model...")
        # Apply bigrams first, then find trigrams
        bigram_docs = [bigram_mod[doc] for doc in all_token_lists]
        trigram_model = models.Phrases(
            bigram_docs,
            min_count=config.TRIGRAM_MIN_COUNT,
            threshold=config.TRIGRAM_THRESHOLD,
            connector_words=frozenset(['of', 'the', 'and', 'in', 'to', 'for', 'with'])
        )
        trigram_mod = models.phrases.Phraser(trigram_model)

        # Combined phrases model
        def phrases_model(tokens):
            """Apply bigram and trigram detection"""
            return trigram_mod[bigram_mod[tokens]]

        logger.info("N-gram models trained successfully")

        # Phase 2: Reprocess all texts with n-gram models
        logger.info("Phase 2: Reprocessing texts with n-gram detection...")
        all_tokens = []
        valid_docs = 0

        for item in index:
            combined_text = f"{item.get('title', '')} {item.get('description', '')} {item.get('full_text', '')}"
            tokens = preprocess_for_gensim(combined_text, spacy_nlp_model, phrases_model)

            if tokens and len(tokens) >= config.MIN_TOKENS_PER_DOCUMENT:
                all_tokens.append(tokens)
                item["lda_tokens"] = tokens
                valid_docs += 1
            else:
                item["lda_tokens"] = []

        logger.info(f"Enhanced preprocessing with n-grams completed. {valid_docs} documents with valid tokens.")

        # Phase 3: Enhanced dictionary creation with n-gram aware filtering
        logger.info("Phase 3: Creating enhanced dictionary with n-gram support...")
        lda_dictionary = corpora.Dictionary(all_tokens)
        original_token_count = len(lda_dictionary)

        # Calculate corpus statistics for smarter filtering
        corpus_size = len(all_tokens)

        # More sophisticated filtering for n-gram enhanced corpus
        min_doc_freq = max(config.MIN_CORPUS_TERMS, corpus_size // 100)  # Appear in at least 1% of docs or MIN_CORPUS_TERMS
        max_doc_freq = 0.3  # More restrictive for n-gram corpus (30% max)
        keep_n = min(3000, config.MAX_TOKENS_FOR_GENSIM * 2)  # Keep more tokens for n-gram richness

        lda_dictionary.filter_extremes(
            no_below=min_doc_freq,
            no_above=max_doc_freq,
            keep_n=keep_n
        )

        logger.info(f"Enhanced n-gram dictionary: {original_token_count} -> {len(lda_dictionary)} tokens")
        logger.info(f"Dictionary filtering: min_freq={min_doc_freq}, max_freq={max_doc_freq:.1%}, keep_n={keep_n}")

        # Phase 4: Create corpus and find optimal number of topics
        corpus = [lda_dictionary.doc2bow(tokens) for tokens in all_tokens]
        corpus = [doc for doc in corpus if doc and len(doc) >= config.MIN_CORPUS_TERMS]

        if len(corpus) < config.MIN_TOPIC_DOCUMENTS:
            logger.warning("Insufficient valid documents after enhanced corpus creation.")
            return None, None

        # Phase 5: Train LDA with optimal parameters
        logger.info("Phase 5: Training optimized LDA model...")

        # Smart topic number calculation for n-gram enhanced corpus
        base_topics = min(config.NUM_TOPICS * 2, corpus_size // 4)
        num_topics = max(25, min(50, base_topics))

        logger.info(f"Training enhanced LDA model with {num_topics} topics on {len(corpus)} documents...")

        # Optimized LDA parameters for n-gram enhanced historical document analysis
        lda_model = models.LdaModel(
            corpus=corpus,
            id2word=lda_dictionary,
            num_topics=num_topics,
            chunksize=max(50, len(corpus) // 20),
            passes=20,
            iterations=300,
            alpha='asymmetric',
            eta=0.01,
            random_state=42,
            eval_every=None,
            per_word_topics=False,
            minimum_probability=0.01,
            minimum_phi_value=0.01,
            dtype=np.float32 if np else None # Check if np is available
        )

        logger.info("Enhanced n-gram LDA model training completed")

        # Phase 6: Evaluate topic quality with coherence metrics
        try:
            if len(corpus) > 50 and ADVANCED_GENSIM_AVAILABLE:  # Only calculate coherence for larger corpora
                logger.info("Calculating topic coherence...")
                coherence_model = CoherenceModel(
                    model=lda_model,
                    texts=all_tokens,
                    dictionary=lda_dictionary,
                    coherence='c_v'
                )
                coherence_score = coherence_model.get_coherence()
                logger.info(f"Topic coherence score (C_v): {coherence_score:.4f}")

                # Store coherence score for future reference
                lda_model.coherence_score = coherence_score
        except Exception as e:
            logger.warning(f"Error calculating topic coherence: {e}")

        # Log enhanced topic quality information
        try:
            logger.info(f"Enhanced model trained with {lda_model.num_topics} topics")

            # Sample topics to show n-gram effectiveness
            sample_topics = min(5, lda_model.num_topics)
            logger.info("Sample n-gram enhanced topics:")
            for i in range(sample_topics):
                terms = ", ".join([term for term, _ in lda_model.show_topic(i, topn=10)])
                logger.info(f"  Topic {i}: {terms}")

        except Exception as e:
            logger.warning(f"Error logging enhanced topic quality info: {e}")

        return lda_model, lda_dictionary

    except Exception as e:
        logger.error(f"Error in enhanced n-gram LDA model training: {e}")
        return None, None

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

def _create_and_save_index(xml_files_to_process: List[str], sentence_model, spacy_nlp_model, index_path: str) -> List[Dict[str, Any]]:
    """
    Enhanced indexing function with better error handling and progress tracking.
    Parses XML files, generates embeddings, trains LDA model, and saves the index.

    Args:
        xml_files_to_process: List of XML file paths to process
        sentence_model: Sentence transformer model
        spacy_nlp_model: spaCy NLP model
        index_path: Path to save the index file

    Returns:
        List of processed document dictionaries
    """
    index = []

    if not xml_files_to_process:
        logger.warning("No XML files provided to process for new index.")
        return index

    logger.info(f"Creating new index from {len(xml_files_to_process)} XML files...")

    # Phase 1: Parse XML files
    parsed_count = 0
    failed_count = 0

    for i, file_path in enumerate(xml_files_to_process):
        try:
            if not os.path.exists(file_path):
                logger.warning(f"File not found - '{file_path}'. Skipping.")
                failed_count += 1
                continue

            if not os.access(file_path, os.R_OK):
                logger.warning(f"File not readable - '{file_path}'. Skipping.")
                failed_count += 1
                continue
        
            parsed_item = parse_letter_xml(file_path)
            if parsed_item:
                index.append(parsed_item)
                parsed_count += 1
            else:
                failed_count += 1

            # Progress reporting
            if (i + 1) % config.PROGRESS_REPORT_INTERVAL == 0:
                logger.info(f"Parsing progress: {i + 1}/{len(xml_files_to_process)} files processed "
                            f"({parsed_count} successful, {failed_count} failed)")

        except Exception as e:
            logger.error(f"Unexpected error processing {file_path}: {e}")
            failed_count += 1
            continue

    logger.info(f"XML parsing completed: {parsed_count} successful, {failed_count} failed")

    if not index:
        logger.error("No documents were successfully parsed. Cannot create index.")
        return index

    # Phase 2: Generate embeddings
    if sentence_model and SENTENCE_TRANSFORMERS_AVAILABLE:
        logger.info("Generating embeddings...")
        try:
            # Prepare texts for embedding
            texts_for_embedding = []
            for item in index:
                text_for_embedding = f"{item.get('title', '')} {item.get('description', '')} {item.get('full_text', '')}"
                texts_for_embedding.append(text_for_embedding.strip())

            # Generate embeddings in batches
            embeddings = generate_embeddings_batch(texts_for_embedding, sentence_model)

            # Assign embeddings to items
            for item, embedding in zip(index, embeddings):
                item['embedding'] = embedding

            successful_embeddings = sum(1 for emb in embeddings if emb is not None)
            logger.info(f"Embedding generation completed: {successful_embeddings}/{len(index)} successful")

        except Exception as e:
            logger.error(f"Error in embedding generation: {e}")
            # Continue without embeddings
            for item in index:
                item['embedding'] = None
    else:
        logger.warning("Sentence transformer not available. Skipping embedding generation.")
        for item in index:
            item['embedding'] = None

    # Phase 3: Topic modeling
    lda_model, lda_dictionary = train_lda_model(index, spacy_nlp_model)

    if lda_model and lda_dictionary:
        assign_topics_to_documents(index, lda_model, lda_dictionary)

    # Phase 3.5: Advanced Topic Modeling (Phase 2 features)
    hdp_model, hdp_dictionary = None, None
    dtm_model, dtm_dictionary, time_slices = None, None, None

    if ADVANCED_GENSIM_AVAILABLE and index:
        logger.info("Training advanced topic models...")

        # Hierarchical Topic Modeling
        try:
            hdp_model, hdp_dictionary = train_hierarchical_topic_model(index, spacy_nlp_model)
        except Exception as e:
            logger.warning(f"Hierarchical topic modeling failed: {e}")

        # Dynamic Topic Modeling
        try:
            dtm_model, dtm_dictionary, time_slices = train_dynamic_topic_model(index)
        except Exception as e:
            logger.warning(f"Dynamic topic modeling failed: {e}")

    # Phase 4: Save index
    if index:
        try:
            payload = {
                "index": index,
                "lda_model": lda_model,
                "lda_dictionary": lda_dictionary,
                "hdp_model": hdp_model,
                "hdp_dictionary": hdp_dictionary,
                "dtm_model": dtm_model,
                "dtm_dictionary": dtm_dictionary,
                "time_slices": time_slices,
                "creation_timestamp": time.time(),
                "version": "2.0",  # Version for future compatibility
                "stats": {
                    "total_documents": len(index),
                    "successful_embeddings": sum(1 for item in index if item.get('embedding') is not None),
                    "documents_with_topics": sum(1 for item in index if item.get('topic_id') is not None),
                    "topics_available": lda_model.num_topics if lda_model else 0
                }
            }

            # Create backup if index already exists
            if os.path.exists(index_path):
                backup_path = f"{index_path}.backup"
                try:
                    os.rename(index_path, backup_path)
                    logger.info(f"Created backup: {backup_path}")
                except Exception as e:
                    logger.warning(f"Could not create backup: {e}")

            with open(index_path, "wb") as f:
                pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

            logger.info(f"New index + LDA saved to '{index_path}'")
            logger.info(f"Index stats: {payload['stats']}")

        except Exception as e:
            logger.error(f"Error saving index to '{index_path}': {e}")

    return index

def load_or_create_index(xml_files_to_process: List[str], sentence_model, spacy_nlp_model,
                         index_path: str, force_reindex: bool = False) -> List[Dict[str, Any]]:
    """
    Enhanced index loading/creation with better error handling and validation.

    Args:
        xml_files_to_process: List of XML files to process
        sentence_model: Sentence transformer model
        spacy_nlp_model: spaCy model
        index_path: Path to the index file
        force_reindex: Whether to force recreation of the index

    Returns:
        List of processed document dictionaries
    """
    global global_lda_model, global_lda_dictionary
    global global_hdp_model, global_hdp_dictionary
    global global_dtm_model, global_dtm_dictionary, global_time_slices
    global global_qa_pipeline

    # Clear global models at start
    global_lda_model = None
    global_lda_dictionary = None
    global_hdp_model = None
    global_hdp_dictionary = None
    global_dtm_model = None
    global_dtm_dictionary = None
    global_time_slices = None
    global_qa_pipeline = None # Already done in load_nlp_models, but good to be explicit
    
    if not force_reindex and os.path.exists(index_path):
        logger.info(f"Loading existing index from '{index_path}'...")
        try:
            # Check file size and modification time
            file_size = os.path.getsize(index_path)
            mod_time = os.path.getmtime(index_path)
            logger.info(f"Index file: {file_size / (1024*1024):.1f} MB, modified: {time.ctime(mod_time)}")

            with open(index_path, 'rb') as f:
                data = pickle.load(f)
            
            # Validate loaded data
            if isinstance(data, dict):
                # New format with metadata
                if 'index' not in data:
                    raise ValueError("Invalid index format: missing 'index' key")

                index = data['index']
                global_lda_model = data.get('lda_model')
                global_lda_dictionary = data.get('lda_dictionary')

                # Phase 2: Load advanced topic models
                global_hdp_model = data.get('hdp_model')
                global_hdp_dictionary = data.get('hdp_dictionary')
                global_dtm_model = data.get('dtm_model')
                global_dtm_dictionary = data.get('dtm_dictionary')
                global_time_slices = data.get('time_slices')

                # Log statistics
                stats = data.get('stats', {})
                version = data.get('version', '1.0')

                logger.info(f"Loaded index version {version}")
                if stats:
                    logger.info(f"Index stats: {stats}")

                # Validate LDA model
                if global_lda_model:
                    try:
                        num_topics = global_lda_model.num_topics
                        logger.info(f"LDA model loaded with {num_topics} topics")
                    except Exception as e:
                        logger.warning(f"LDA model validation failed: {e}")
                        global_lda_model = None

                # Validate HDP model
                if global_hdp_model:
                    try:
                        # Test HDP model functionality
                        topics = global_hdp_model.show_topics(num_topics=5, formatted=False)
                        active_topics = len([t for t in topics if t[1] and len(t[1]) > 0])
                        logger.info(f"HDP model loaded with {active_topics} active topics (sample)")
                    except Exception as e:
                        logger.warning(f"HDP model validation failed: {e}")
                        global_hdp_model = None

                # Validate DTM model
                if global_dtm_model and global_time_slices:
                    try:
                        num_topics = global_dtm_model.num_topics
                        logger.info(f"DTM model loaded with {num_topics} topics across {len(global_time_slices)} time periods")
                    except Exception as e:
                        logger.warning(f"DTM model validation failed: {e}")
                        global_dtm_model = None
                        global_time_slices = None

            elif isinstance(data, list):
                # Old format - just the index
                logger.info("Loading legacy index format")
                index = data
                global_lda_model = None
                global_lda_dictionary = None
            else:
                raise ValueError(f"Invalid index format: expected dict or list, got {type(data)}")

            # Validate index content
            if not index:
                logger.warning("Loaded index is empty")
                return []

            # Check if index items have required fields
            sample_item = index[0]
            required_fields = ['file_name', 'title', 'full_text']
            missing_fields = [field for field in required_fields if field not in sample_item]

            if missing_fields:
                logger.warning(f"Index items missing required fields: {missing_fields}")

            # Check embedding availability
            embeddings_available = sum(1 for item in index if item.get('embedding') is not None)
            logger.info(f"Successfully loaded {len(index)} items from saved index")
            logger.info(f"Embeddings available: {embeddings_available}/{len(index)}")

            # Warn if embeddings are missing but sentence model is available
            if sentence_model and embeddings_available == 0 and len(index) > 0 :
                logger.warning("No embeddings found in index but sentence model is available. Consider re-indexing.")

            return index

        except (pickle.PickleError, EOFError) as e:
            logger.error(f"Error loading pickled index from '{index_path}': {e}")
        except ValueError as e:
            logger.error(f"Index validation error: {e}")
        except Exception as e:
            logger.error(f"Unexpected error loading index from '{index_path}': {e}")
    
        logger.info("Will attempt to create a new index due to loading failure")

    # Create new index
    logger.info("Creating new index...")
    index = _create_and_save_index(xml_files_to_process, sentence_model, spacy_nlp_model, index_path)

    # Set global LDA variables from newly saved file (if creation was successful)
    if index and os.path.exists(index_path): # Check if index was created and file exists
        try:
            with open(index_path, 'rb') as f:
                data = pickle.load(f)
                if isinstance(data, dict):
                    global_lda_model = data.get('lda_model')
                    global_lda_dictionary = data.get('lda_dictionary')
                    # Also load HDP and DTM models into global variables
                    global_hdp_model = data.get('hdp_model')
                    global_hdp_dictionary = data.get('hdp_dictionary')
                    global_dtm_model = data.get('dtm_model')
                    global_dtm_dictionary = data.get('dtm_dictionary')
                    global_time_slices = data.get('time_slices')

                    # Log what was loaded
                    if global_lda_model:
                        logger.info("LDA model from new index loaded successfully into global variables")
                    if global_hdp_model:
                        logger.info("HDP model from new index loaded successfully into global variables")
                    if global_dtm_model and global_time_slices:
                        logger.info(f"DTM model from new index loaded successfully with {len(global_time_slices)} time slices")

        except Exception as e:
            logger.warning(f"Could not load models from newly created and saved index: {e}")
    
    return index


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
        return []
    if not query_text:
        print("Semantic search query is empty.")
        return []

    try:
        query_embedding = sentence_model.encode(query_text, convert_to_tensor=True)
        items_with_embeddings = [
            item for item in letter_index
            if item.get('embedding') is not None # Check if embedding exists
        ]
    
        if not items_with_embeddings:
            print("No documents with valid embeddings found in the index for semantic search.")
            return []
        
        # Prepare corpus embeddings, ensuring they are tensors
        corpus_embeddings_list = []
        valid_items_indices = []
        for i, item_emb in enumerate(item['embedding'] for item in items_with_embeddings):
            if isinstance(item_emb, np.ndarray):
                corpus_embeddings_list.append(util.torch.tensor(item_emb))
                valid_items_indices.append(i)
            elif util.torch.is_tensor(item_emb):
                corpus_embeddings_list.append(item_emb)
                valid_items_indices.append(i)
        
        if not corpus_embeddings_list:
            print("No valid tensor embeddings found in corpus.")
            return []

        corpus_embeddings = util.torch.stack(corpus_embeddings_list)
    
        # Ensure embeddings are on CPU if they are PyTorch tensors
        if hasattr(query_embedding, 'cpu'):
            query_embedding = query_embedding.cpu()
            if hasattr(corpus_embeddings, 'cpu'):
                corpus_embeddings = corpus_embeddings.cpu()

        hits = util.semantic_search(
            query_embedding,
            corpus_embeddings,
            top_k=min(top_n * 2, len(corpus_embeddings))  # Get more results for snippet analysis
        )
    
        search_results = []
        if hits and hits[0]:
            for hit in hits[0][:top_n]:  # Only keep top_n final results
                original_item_index = valid_items_indices[hit['corpus_id']]
                original_item = items_with_embeddings[original_item_index].copy()
                original_item['similarity_score'] = hit['score'] 

                # Extract the most relevant snippet from the document
                relevant_snippet = extract_relevant_snippet(
                        original_item, query_text, sentence_model
                )
                original_item['relevant_snippet'] = relevant_snippet

                search_results.append(original_item)

        return search_results

    except Exception as e:
        logger.error(f"Error in enhanced semantic search: {e}")
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
    
    if args.dir:
        if os.path.isdir(args.dir):
            print(f"Target XML directory: {args.dir}")
            files_to_process = glob.glob(os.path.join(args.dir, '*.xml'))
        else:
            print(f"Error: Provided directory '{args.dir}' does not exist.")
    elif args.files:
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
    # Load spaCy model
    spacy_nlp = None
    if SPACY_AVAILABLE: # Check if spacy library was imported
        try:
            spacy_nlp = spacy.load("en_core_web_sm")
            print("spaCy model 'en_core_web_sm' loaded.")
        except OSError:
            print("spaCy model 'en_core_web_sm' not found. Download: python -m spacy download en_core_web_sm")
            spacy_nlp = None
    
    # Load sentence transformer model
    sentence_transformer_model = None
    if SENTENCE_TRANSFORMERS_AVAILABLE: # Check if sentence_transformers library was imported
        try:
            sentence_transformer_model = SentenceTransformer(SENTENCE_MODEL_NAME)
            print(f"SentenceTransformer model '{SENTENCE_MODEL_NAME}' loaded.")
        except Exception as e:
            print(f"Error loading SentenceTransformer model '{SENTENCE_MODEL_NAME}': {e}")
            sentence_transformer_model = None
            
    # Load QA pipeline (Phase 2)
    qa_pipeline_model = initialize_qa_model() # Renamed to avoid conflict
    if qa_pipeline_model:
        print("Question Answering pipeline loaded successfully.")
    # else: # initialize_qa_model already logs
    #     print("QA pipeline not available.")

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
                    terms = dtm_model.show_topic(topic_id_iter, time=time_idx, topn=5)
                    terms_str = ", ".join([word for word, _ in terms])
                    logger.info(f"    {year_val}: {terms_str}")

        return dtm_model, dtm_dictionary, time_slices # Return original time_slices

    except Exception as e:
        logger.error(f"Error in dynamic topic modeling: {e}")
        return None, None, None

def initialize_qa_model():
    """
    Initialize the Question Answering model for extractive QA.

    Returns:
        QA pipeline or None if not available
    """
    if not HF_TRANSFORMERS_AVAILABLE:
        logger.warning("Transformers library not available - QA features disabled")
        return None

    try:
        logger.info(f"Loading QA model: {config.QA_MODEL_NAME}")
        qa_pipeline_model = pipeline( # Renamed variable
            "question-answering",
            model=config.QA_MODEL_NAME,
            tokenizer=config.QA_MODEL_NAME,
            # return_scores=True, # Default for pipeline is to return scores
            handle_impossible_answer=True # Important for robust QA
        )
        logger.info("QA model loaded successfully")
        return qa_pipeline_model
    except Exception as e:
        logger.error(f"Error loading QA model: {e}")
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

        # Step 1: Find relevant contexts using semantic search
        relevant_docs = execute_semantic_search(
            letter_index, query_text, sentence_model,
            top_n=config.QA_TOP_K_CONTEXTS
        )

        if not relevant_docs:
            logger.warning("No relevant contexts found for QA")
            return []

        answers = []

        # Step 2: Extract answers from each relevant context
        for doc in relevant_docs:
            try:
                # Use full text as context for QA
                context = doc.get('full_text', '')
                if not context or len(context.strip()) < 20:
                    continue

                # Truncate context if too long for the model
                # Model's max sequence length is usually 512 tokens.
                # A simple char limit is a heuristic. Better to use tokenizer.
                max_context_char_length = 3000 # A conservative character limit
                if len(context) > max_context_char_length:
                    context = context[:max_context_char_length] + "..."


                # Get answer from QA model
                qa_result = qa_pipeline_model(
                    question=query_text,
                    context=context,
                    max_answer_len=config.QA_MAX_ANSWER_LENGTH,
                    handle_impossible_answer=True
                )
                
                # The pipeline returns a dict or a list of dicts. We expect one.
                if isinstance(qa_result, list): # if top_k > 1 in pipeline
                    if not qa_result: continue
                    qa_result = qa_result[0]


                # Check if answer meets quality threshold
                if qa_result and qa_result.get('score', 0) >= config.QA_MIN_SCORE_THRESHOLD and qa_result.get('answer'):
                    answer_data = {
                        'answer': qa_result['answer'],
                        'confidence': qa_result['score'],
                        'context_snippet': _extract_answer_context(
                            context, qa_result['answer'], qa_result.get('start', 0)
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
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chatbot for querying TEI XML letters with AI and persistent index.")
    parser.add_argument("--dir", help="Directory containing XML letter files.")
    parser.add_argument("--files", nargs="+", help="List of specific XML files to process.")
    parser.add_argument("--reindex", action="store_true", help="Force re-creation of the index, ignoring any saved index file.")
    args = parser.parse_args()

    # Get files to process
    files_to_process = get_file_list_from_args(args)
    
    # Load NLP models
    spacy_nlp, sentence_transformer_model, qa_pipeline_model = load_nlp_models()

    # Store QA pipeline globally
    global_qa_pipeline = qa_pipeline_model # Assign to global variable

    # Determine if we should create a new index
    # If reindex is forced, files_to_process will be used.
    # If not reindexing and index file exists, files_for_indexing will be empty (load_or_create handles this).
    # If not reindexing and index file doesn't exist, files_to_process will be used.
    if args.reindex:
        files_for_indexing = files_to_process
    elif not os.path.exists(SAVED_INDEX_FILE):
        files_for_indexing = files_to_process
        if not files_for_indexing: # No files provided and no index exists
            logger.warning(f"No XML files specified to create an index, and '{SAVED_INDEX_FILE}' not found.")
            print(f"No XML files specified to create an index, and '{SAVED_INDEX_FILE}' not found.")
            print("Please specify XML files/directory or ensure a saved index exists.")
            exit() # Exit if no source for index
    else: # Not reindexing and index file exists, so we will try to load it.
        files_for_indexing = [] # Pass empty list, load_or_create_index will handle loading

    letter_data_index = load_or_create_index(
        files_for_indexing, # This will be empty if we are loading, populated if creating
        sentence_transformer_model, 
        spacy_nlp, 
        SAVED_INDEX_FILE, 
        force_reindex=args.reindex
    )
    
    # Start chatbot if we have data
    if letter_data_index:
        run_chatbot(letter_data_index, spacy_nlp, sentence_transformer_model)
    else:
        print("\nFailed to load or create any letter data. Exiting.")
