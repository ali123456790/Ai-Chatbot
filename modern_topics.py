# Modern Fast Topic Discovery Using Embeddings + Clustering
# This replaces slow LDA with instant clustering of existing embeddings

import numpy as np
from sklearn.cluster import HDBSCAN, KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from collections import Counter
import re
from typing import List, Dict, Any, Tuple
import logging
import time
import torch

logger = logging.getLogger(__name__)

class ModernTopicDiscovery:
    """
    Ultra-fast topic discovery using sentence embeddings + clustering.
    No preprocessing needed - uses embeddings we already computed!
    """
    
    def __init__(self, n_topics: int = 20, min_cluster_size: int = 10):
        self.n_topics = n_topics
        self.min_cluster_size = min_cluster_size
        self.cluster_model = None
        self.topic_keywords = {}
        self.topic_names = {}
        
    def discover_topics(self, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Discover topics instantly using existing embeddings.
        
        Args:
            documents: List of document dicts with 'embedding' and 'full_text'
            
        Returns:
            Dictionary with topic info and document assignments
        """
        start_time = time.time()
        
        embeddings_for_clustering = []
        texts_for_clustering = []
        valid_docs_for_clustering = []
        
        for i, doc in enumerate(documents):
            doc_id = doc.get('id', doc.get('file_name', f'doc_index_{i}'))
            embedding_data = doc.get('embedding')
            current_embedding_np = None

            if embedding_data is None:
                logger.debug(f"Doc ID: {doc_id} has no embedding. Skipping.")
                continue

            # Attempt to convert to a consistent NumPy float32 format
            try:
                if isinstance(embedding_data, np.ndarray):
                    if embedding_data.ndim == 1 and embedding_data.size > 0:
                        current_embedding_np = embedding_data.astype(np.float32)
                    else:
                        logger.warning(f"Doc ID: {doc_id} has NumPy embedding with invalid shape {embedding_data.shape} or size {embedding_data.size}. Skipping.")
                elif isinstance(embedding_data, list):
                    if embedding_data and all(isinstance(x, (float, int, np.float32, np.float64)) for x in embedding_data):
                        current_embedding_np = np.array(embedding_data, dtype=np.float32)
                        if current_embedding_np.ndim != 1 or current_embedding_np.size == 0:
                            logger.warning(f"Doc ID: {doc_id} has list embedding that converted to invalid shape {current_embedding_np.shape} or size {current_embedding_np.size}. Skipping.")
                            current_embedding_np = None # Invalidate if shape is wrong after conversion
                    else:
                        logger.warning(f"Doc ID: {doc_id} has list embedding with non-numeric data or is empty. Skipping.")
                elif torch.is_tensor(embedding_data): # Handle PyTorch tensors
                    tensor_1d = embedding_data.cpu().numpy().astype(np.float32)
                    if tensor_1d.ndim == 1 and tensor_1d.size > 0:
                        current_embedding_np = tensor_1d
                    else:
                         logger.warning(f"Doc ID: {doc_id} has PyTorch tensor embedding with invalid shape {tensor_1d.shape} or size {tensor_1d.size} after conversion. Skipping.")
                else:
                    logger.warning(f"Doc ID: {doc_id} has embedding of unexpected type: {type(embedding_data)}. Skipping.")
            except Exception as e:
                logger.error(f"Error converting embedding for Doc ID: {doc_id} - {type(embedding_data)}: {e}. Skipping.", exc_info=True)
                current_embedding_np = None

            if current_embedding_np is not None:
                if np.isnan(current_embedding_np).any() or np.isinf(current_embedding_np).any():
                    logger.warning(f"Doc ID: {doc_id} embedding contains NaN or Inf values. Skipping.")
                else:
                    embeddings_for_clustering.append(current_embedding_np)
                    texts_for_clustering.append(doc.get('full_text', ''))
                    valid_docs_for_clustering.append(doc)
            
        if len(embeddings_for_clustering) < self.min_cluster_size:
            logger.warning(f"Not enough documents with valid embeddings for clustering: {len(embeddings_for_clustering)} found, need at least {self.min_cluster_size}.")
            return {'topics': [], 'document_topics': [], 'processing_time': time.time() - start_time, 'method': 'embedding_clustering', 'error': 'Insufficient valid embeddings'}
        
        final_embeddings_np = np.array(embeddings_for_clustering) # Should already be list of 1D np.arrays
        logger.info(f"Using {len(final_embeddings_np)} documents with valid embeddings for topic discovery (shape: {final_embeddings_np.shape}).")
        
        logger.info("Performing fast clustering on embeddings...")
        try:
            clusterer = HDBSCAN(
                min_cluster_size=self.min_cluster_size,
                metric='cosine',
                cluster_selection_method='eom'
            )
            cluster_labels = clusterer.fit_predict(final_embeddings_np)
        except Exception as e_hdbscan:
            logger.error(f"HDBSCAN clustering failed: {e_hdbscan}. Falling back to KMeans.", exc_info=True)
            # Fallback to KMeans if HDBSCAN fails for any reason (e.g., all points considered noise)
            try:
                num_kmeans_clusters = min(self.n_topics, max(1, len(final_embeddings_np) // self.min_cluster_size)) # Ensure at least 1 cluster
                if num_kmeans_clusters == 0: num_kmeans_clusters = 1 # Safety for very few docs
                logger.info(f"Falling back to KMeans with n_clusters={num_kmeans_clusters}")
                clusterer = KMeans(n_clusters=num_kmeans_clusters, random_state=42, n_init='auto')
                cluster_labels = clusterer.fit_predict(final_embeddings_np)
            except Exception as e_kmeans:
                logger.error(f"KMeans fallback clustering also failed: {e_kmeans}. Cannot perform topic discovery.", exc_info=True)
                return {'topics': [], 'document_topics': [], 'processing_time': time.time() - start_time, 'method': 'embedding_clustering', 'error': 'Clustering failed'}

        unique_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        if unique_clusters < 1 and not isinstance(clusterer, KMeans): # If HDBSCAN found no real clusters, and we haven't tried KMeans yet
            logger.info("HDBSCAN found no significant clusters (or only noise). Attempting KMeans as fallback...")
            try:
                num_kmeans_clusters = min(self.n_topics, max(1, len(final_embeddings_np) // self.min_cluster_size))
                if num_kmeans_clusters == 0: num_kmeans_clusters = 1
                logger.info(f"Falling back to KMeans with n_clusters={num_kmeans_clusters}")
                clusterer = KMeans(n_clusters=num_kmeans_clusters, random_state=42, n_init='auto')
                cluster_labels = clusterer.fit_predict(final_embeddings_np)
            except Exception as e_kmeans_fallback:
                logger.error(f"KMeans fallback clustering also failed: {e_kmeans_fallback}. Cannot perform topic discovery.", exc_info=True)
                return {'topics': [], 'document_topics': [], 'processing_time': time.time() - start_time, 'method': 'embedding_clustering', 'error': 'Clustering failed'}
        
        topics = self._generate_topic_info(texts_for_clustering, cluster_labels)
        
        document_topics = []
        for i, doc in enumerate(valid_docs_for_clustering):
            if i < len(cluster_labels):
                topic_id = int(cluster_labels[i])
                if topic_id != -1:  # -1 is noise in HDBSCAN
                    doc['topic_id'] = topic_id
                    doc['topic_name'] = topics.get(topic_id, {}).get('name', f'Topic {topic_id}')
                    document_topics.append((doc.get('file_name', f'doc_index_{i}'), topic_id))
        
        processing_time = time.time() - start_time
        logger.info(f"Fast topic discovery completed in {processing_time:.2f} seconds, found {len(topics)} topics.")
        
        return {
            'topics': topics,
            'document_topics': document_topics,
            'cluster_model': clusterer,
            'processing_time': processing_time,
            'method': 'embedding_clustering'
        }
    
    def _generate_topic_info(self, texts: List[str], cluster_labels: np.ndarray) -> Dict[int, Dict[str, Any]]:
        """Generate topic keywords and names using TF-IDF on clustered documents."""
        topics = {}
        
        # Group texts by cluster
        cluster_texts = {}
        for text, label in zip(texts, cluster_labels):
            if label != -1:  # Skip noise
                if label not in cluster_texts:
                    cluster_texts[label] = []
                cluster_texts[label].append(text)
        
        # Generate keywords for each topic using TF-IDF
        for cluster_id, cluster_docs in cluster_texts.items():
            num_docs_in_cluster = len(cluster_docs)
            
            if num_docs_in_cluster < 3: # Keep this minimum, TF-IDF needs some docs
                logger.info(f"Cluster {cluster_id} has only {num_docs_in_cluster} documents. Skipping keyword generation.")
                topics[cluster_id] = {
                    'id': cluster_id,
                    'name': f'Small Cluster {cluster_id}',
                    'keywords': ['cluster too small'],
                    'doc_count': num_docs_in_cluster,
                    'representative_text': cluster_docs[0][:200] + "..." if cluster_docs else ""
                }
                continue
                
            # Combine all documents in cluster
            combined_text_for_vectorizer = ' '.join(cluster_docs) # This is for fitting TF-IDF *within* the cluster
            
            # Adjust TF-IDF parameters based on cluster size
            current_min_df = 1 if num_docs_in_cluster < 20 else 2 # More lenient for small clusters
            current_max_df = 0.95 # Be less restrictive
            current_ngram_range = (1, 1) if num_docs_in_cluster < 10 else (1, 2) # No bigrams for very small clusters

            logger.info(f"Cluster {cluster_id} ({num_docs_in_cluster} docs): TF-IDF params: min_df={current_min_df}, max_df={current_max_df}, ngrams={current_ngram_range}")

            vectorizer = TfidfVectorizer(
                max_features=100, # Keep max_features to limit keyword list size
                stop_words='english',
                ngram_range=current_ngram_range,
                min_df=current_min_df,
                max_df=current_max_df
            )
            
            try:
                # Fit TF-IDF on the documents *within the current cluster*
                # This helps find terms that are characteristic of *this specific cluster*
                tfidf_matrix = vectorizer.fit_transform(cluster_docs) # Fit on the list of docs in cluster
                feature_names = vectorizer.get_feature_names_out()
                
                # To get representative keywords, average TF-IDF scores across docs in cluster or sum them
                # For simplicity, let's sum the TF-IDF scores for each term across all docs in the cluster
                summed_tfidf_scores = tfidf_matrix.sum(axis=0).A1 # Sum scores for each term, .A1 converts to 1D array
                
                top_indices = summed_tfidf_scores.argsort()[-10:][::-1] # Get top 10 terms by summed score
                keywords = [feature_names[i] for i in top_indices if summed_tfidf_scores[i] > 0]
                
                if not keywords:
                    logger.warning(f"No keywords found for cluster {cluster_id} after TF-IDF. Assigning default.")
                    keywords = ['terms unavailable']

                topic_name = self._generate_topic_name(keywords[:3])
                
                topics[cluster_id] = {
                    'id': cluster_id,
                    'name': topic_name,
                    'keywords': keywords,
                    'doc_count': num_docs_in_cluster,
                    'representative_text': cluster_docs[0][:200] + "..." if cluster_docs else ""
                }
                
            except ValueError as ve:
                # This specific ValueError often happens with min_df/max_df issues
                logger.warning(f"TF-IDF ValueError for cluster {cluster_id} ({num_docs_in_cluster} docs): {ve}. Assigning default keywords.")
                topics[cluster_id] = {
                    'id': cluster_id,
                    'name': f'Cluster {cluster_id} (TF-IDF Issue)',
                    'keywords': ['vectorizer error'],
                    'doc_count': num_docs_in_cluster,
                    'representative_text': cluster_docs[0][:200] + "..." if cluster_docs else ""
                }
            except Exception as e:
                logger.error(f"Unexpected error generating keywords for cluster {cluster_id}: {e}", exc_info=True)
                topics[cluster_id] = {
                    'id': cluster_id,
                    'name': f'Cluster {cluster_id} (Error)',
                    'keywords': ['processing error'],
                    'doc_count': num_docs_in_cluster,
                    'representative_text': cluster_docs[0][:200] + "..." if cluster_docs else ""
                }
        
        return topics
    
    def _generate_topic_name(self, keywords: List[str]) -> str:
        """Generate a readable topic name from keywords."""
        if not keywords:
            return "Miscellaneous"
        
        # Clean and join keywords
        clean_keywords = []
        for kw in keywords[:3]:
            # Remove common prefixes/suffixes, capitalize
            clean_kw = re.sub(r'^(the|and|for|with|from)\s+', '', kw.lower())
            clean_kw = re.sub(r'\s+(and|for|with|from)$', '', clean_kw)
            if len(clean_kw) > 2:
                clean_keywords.append(clean_kw.title())
        
        if len(clean_keywords) == 1:
            return clean_keywords[0]
        elif len(clean_keywords) == 2:
            return f"{clean_keywords[0]} & {clean_keywords[1]}"
        else:
            return f"{clean_keywords[0]}, {clean_keywords[1]} & {clean_keywords[2]}"

def fast_topic_modeling(documents: List[Dict[str, Any]], n_topics: int = 20) -> Dict[str, Any]:
    """
    Main function for fast topic modeling using existing embeddings.
    
    Args:
        documents: List of documents with embeddings already computed
        n_topics: Target number of topics (approximate)
        
    Returns:
        Topic modeling results
    """
    logger.info(f"Starting fast topic modeling on {len(documents)} documents...")
    
    discovery = ModernTopicDiscovery(n_topics=n_topics, min_cluster_size=max(5, len(documents)//100))
    results = discovery.discover_topics(documents)
    
    logger.info(f"Discovered {len(results['topics'])} topics in {results['processing_time']:.2f} seconds")
    
    return results

# Integration function for the main chat.py
def replace_slow_topic_modeling(working_index: List[Dict[str, Any]], config) -> Tuple[Any, Dict[str, Any]]:
    """
    Drop-in replacement for the slow LDA topic modeling.
    Uses embeddings we already computed - no preprocessing needed!
    """
    logger.info("Using modern fast topic modeling (embedding clustering)")
    
    # Use existing embeddings for instant topic discovery
    results = fast_topic_modeling(working_index, n_topics=config.NUM_TOPICS)
    
    # Create a simple model object for compatibility
    class FastTopicModel:
        def __init__(self, topics_dict):
            self.topics = topics_dict
            self.num_topics = len(topics_dict)
            
        def print_topics(self, num_topics=10):
            for topic_id, topic_info in list(self.topics.items())[:num_topics]:
                keywords = ', '.join(topic_info['keywords'][:5])
                print(f"Topic {topic_id}: {keywords}")
    
    model = FastTopicModel(results['topics'])
    
    return model, results 