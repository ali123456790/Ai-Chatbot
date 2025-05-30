import os
# Prevent HuggingFace tokenizers warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import sys
import streamlit as st

# Configure Streamlit page FIRST - before any other Streamlit commands
st.set_page_config(
    page_title="📜 Historical Document Analyzer",
    page_icon="📜",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/your-repo/historical-analyzer',
        'Report a bug': 'https://github.com/your-repo/historical-analyzer/issues',
        'About': "AI-Powered Historical Document Analyzer - Explore Civil War era letters using advanced NLP"
    }
)

import traceback
import datetime
import time
from typing import Optional, List, Dict, Any
import pandas as pd
import folium
from streamlit_folium import st_folium

# --- Add the directory containing chat.py to Python's path ---
CHAT_PY_DIR = os.path.dirname(os.path.abspath(__file__))
if CHAT_PY_DIR not in sys.path:
    sys.path.append(CHAT_PY_DIR)

# --- Import necessary functions and variables from chat.py ---
try:
    from chat import (
        load_or_create_index,
        load_nlp_models,
        SAVED_INDEX_FILE,
        SENTENCE_MODEL_NAME,
        # Search functions
        search_by_sender, search_by_recipient, search_by_year,
        search_by_place, search_by_keyword,
        process_smart_query,
        execute_smart_search,
        execute_semantic_search,
        # Topic modeling functions & globals
        search_by_topic_id,
        search_by_topic_terms,
        list_available_topics as cli_list_available_topics,
        # Phase 2: Advanced search functions
        execute_extractive_qa,
        execute_hybrid_search,
        search_hierarchical_topics,
        search_dynamic_topics_by_year,
        # Map functionality
        add_geocoding_to_existing_index,
    )
    import chat  # To access global_lda_model and global_lda_dictionary
except ImportError as e:
    st.error(f"🚨 **Critical Error**: Could not import from chat.py: {e}")
    st.markdown("""
    **Troubleshooting Steps:**
    1. Ensure `chat.py` is in the same directory as this file
    2. Check that all dependencies are installed: `pip install -r requirements.txt`
    3. Verify the main CLI logic in `chat.py` is under `if __name__ == '__main__':`
    """)
    st.stop()

# --- Enhanced Configuration ---
class Config:
    """Enhanced configuration settings for the application."""
    XML_DIR_DEFAULT = "./xmlfiles_full/"
    PAGE_TITLE = "📜 Historical Document Analyzer"
    MAX_RESULTS_DISPLAY = 25
    SNIPPET_LENGTH = 600
    
    # UI Settings
    THEME_COLOR = "#1f77b4"
    SUCCESS_COLOR = "#28a745"
    WARNING_COLOR = "#ffc107"
    ERROR_COLOR = "#dc3545"
    
    # Performance Settings
    CACHE_TTL = 3600  # 1 hour
    MAX_SEARCH_HISTORY = 15
    
    # Search Settings
    DEFAULT_MAX_RESULTS = 20
    MIN_SIMILARITY_SCORE = 0.1

@st.cache_data(ttl=Config.CACHE_TTL)
def get_config():
    """Get cached configuration."""
    return Config()

# --- Enhanced Session State Management ---
def initialize_session_state():
    """Initialize all session state variables with better defaults."""
    defaults = {
        'letter_index': None,
        'models_loaded_flag': False,
        'sentence_model_instance': None,
        'spacy_nlp_instance': None,
        'qa_pipeline_instance': None,
        'show_topics': False,
        'last_search_results': [],
        'search_history': [],
        'current_page': 'search',
        'tutorial_shown': False,
        'advanced_mode': False,
        'favorites': [],
        'search_stats': {
            'total_searches': 0, 
            'successful_searches': 0, # Overall successful searches
            'searches_by_type': {} # Detailed stats per search type
        },
        'user_preferences': {
            'show_similarity_scores': True,
            'auto_expand_results': 3,
            'theme': 'default'
        }
    }
    
    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

def ensure_models_loaded() -> tuple:
    """Enhanced model loading with better user feedback."""
    if not st.session_state.models_loaded_flag:
        try:
            with st.spinner("🤖 Loading AI models..."):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                status_text.text("Loading spaCy language model...")
                progress_bar.progress(20)
                
                spacy_instance, sentence_instance, qa_pipeline = load_nlp_models()
                
                status_text.text("Loading sentence transformer...")
                progress_bar.progress(60)
                
                status_text.text("Initializing QA pipeline...")
                progress_bar.progress(80)
                
            st.session_state.spacy_nlp_instance = spacy_instance
            st.session_state.sentence_model_instance = sentence_instance
            st.session_state.qa_pipeline_instance = qa_pipeline
            st.session_state.models_loaded_flag = True

            progress_bar.progress(100)
            status_text.text("✅ All models loaded successfully!")
            time.sleep(1)
            status_text.empty()
            progress_bar.empty()
            
        except Exception as e:
            st.error(f"❌ **Model Loading Failed**: {e}")
            st.session_state.models_loaded_flag = False
            return None, None, None
    
    return (st.session_state.spacy_nlp_instance, 
            st.session_state.sentence_model_instance, 
            st.session_state.qa_pipeline_instance)

# --- Enhanced Welcome Screen ---
def render_welcome_screen():
    """Render an attractive welcome screen with tutorial."""
    st.markdown("""
    <div style="text-align: center; padding: 2rem 0;">
        <h1 style="color: #1f77b4; font-size: 3rem; margin-bottom: 1rem;">
            📜 Historical Document Analyzer
        </h1>
        <h3 style="color: #666; font-weight: 300; margin-bottom: 2rem;">
            Explore Civil War Era Letters with AI-Powered Search
        </h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Quick stats if index is loaded
    if st.session_state.letter_index:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("📚 Total Documents", len(st.session_state.letter_index))
        with col2:
            st.metric("🔍 Search Types", "8+")
        with col3:
            topics = chat.global_lda_model.num_topics if chat.global_lda_model else 0
            st.metric("🏷️ Topics Available", topics)
        with col4:
            st.metric("🧠 AI Features", "QA, Semantic, Smart")
    
    # Feature showcase
    st.markdown("### 🚀 **What You Can Do**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **🔍 Smart Search Options:**
        - **Keyword Search** - Traditional text matching
        - **Smart Search** - AI understands your questions
        - **Semantic Search** - Find similar content by meaning
        - **Question Answering** - Get direct answers from documents
        
        **📊 Advanced Analysis:**
        - **Topic Modeling** - Automatic theme discovery
        - **Hybrid Search** - Combined keyword + semantic
        - **Historical Timeline** - Search by time periods
        """)
    
    with col2:
        st.markdown("""
        **🎯 Precise Filters:**
        - Search by **sender** or **recipient**
        - Filter by **year** or **location**
        - Explore **thematic topics**
        - Export results to **CSV**
        
        **💡 User-Friendly Features:**
        - **Search history** and favorites
        - **Auto-complete** suggestions
        - **Interactive results** with previews
        - **Modern, responsive** interface
        """)
    
    # Quick start guide
    with st.expander("🎯 **Quick Start Guide**", expanded=not st.session_state.tutorial_shown):
        st.markdown("""
        ### **Step 1: Load Your Data** 📁
        - Use the sidebar to load historical documents
        - Choose your XML files directory
        - Click "📚 Load/Index Documents"
        
        ### **Step 2: Choose Search Type** 🔍
        - **New users**: Start with "Keyword" or "Smart Search"
        - **Advanced users**: Try "Semantic Search" or "Question Answering"
        - **Researchers**: Explore "Topic Modeling" features
        
        ### **Step 3: Enter Your Query** ✏️
        - Type naturally - the AI understands context
        - Use the suggested formats for different search types
        - Check "Advanced Options" for more control
        
        ### **Step 4: Explore Results** 📊
        - Click on results to expand details
        - Export interesting findings to CSV
        - Save favorites for later reference
        """)
        
        if st.button("✅ Got it! Take me to search", type="primary"):
            st.session_state.tutorial_shown = True
            st.session_state.current_page = 'search'
            st.rerun()

# --- Enhanced Results Display ---
def display_enhanced_results(results: List[Dict[str, Any]], search_type: str = "", is_semantic_search: bool = False):
    """Enhanced results display with modern UI and better user experience."""
    if not results:
        st.markdown("""
        <div style="text-align: center; padding: 2rem; background: #f8f9fa; border-radius: 10px; border-left: 4px solid #17a2b8;">
            <h3 style="color: #17a2b8;">🔍 No Results Found</h3>
            <p>Try these suggestions:</p>
            <ul style="text-align: left; display: inline-block;">
                <li>Check your spelling</li>
                <li>Try broader search terms</li>
                <li>Use a different search type</li>
                <li>Remove filters or increase result limit</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        return
    
    config = get_config()
    
    # Update search stats
    st.session_state.search_stats['successful_searches'] += 1
    
    # Results header with stats
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        st.markdown(f"""
        <div style="background: linear-gradient(90deg, #1f77b4, #17a2b8); color: white; padding: 1rem; border-radius: 8px; margin-bottom: 1rem;">
            <h3 style="margin: 0; color: white;">📋 Found {len(results)} Documents</h3>
            <p style="margin: 0; opacity: 0.9;">Search type: {search_type}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        if st.button("📥 Export All Results", help="Download results as CSV file"):
            export_enhanced_results(results, search_type)
    
    with col3:
        if st.button("⭐ Save Search", help="Save this search to favorites"):
            save_search_to_favorites(search_type, results)
    
    # Results per page selector
    col1, col2 = st.columns([3, 1])
    with col2:
        results_per_page = st.selectbox("Results per page:", [10, 20, 50], index=1, key="results_per_page")
    
    # Pagination
    total_pages = max(1, (len(results) - 1) // results_per_page + 1)
    
    if total_pages > 1:
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            current_page = st.number_input("Page", 1, total_pages, 1) - 1
    else:
        current_page = 0
    
    # Get results for current page
    start_idx = current_page * results_per_page
    end_idx = min(start_idx + results_per_page, len(results))
    page_results = results[start_idx:end_idx]
    
    # Display results with enhanced formatting
    for i, item in enumerate(page_results, start_idx + 1):
        try:
            display_single_result(item, i, is_semantic_search)
        except Exception as e:
            st.error(f"❌ Error displaying result {i}: {e}")
            continue

def display_single_result(item: Dict[str, Any], index: int, is_semantic: bool = False):
    """Display a single result with enhanced formatting."""
    config = get_config()
    
    # Initialize expanded state tracking in session state
    if 'expanded_items' not in st.session_state:
        st.session_state.expanded_items = set()
    
    # Prepare title and metadata
    title = item.get('title', 'Untitled Document')
    if len(title) > 100:
        title = title[:97] + "..."
    
    # Create result card
    with st.container():
        # Header with title, score badge, and action buttons in one row
        header_col1, header_col2, header_col3, header_col4 = st.columns([3, 1, 0.5, 0.5])
        
        with header_col1:
            # Similarity score badge for semantic search
            if is_semantic and 'similarity_score' in item:
                score = item.get('similarity_score', 0.0)
                score_color = "#28a745" if score > 0.7 else "#ffc107" if score > 0.5 else "#dc3545"
                score_badge = f"<span style='background: {score_color}; color: white; padding: 2px 8px; border-radius: 12px; font-size: 0.8em; margin-right: 8px;'>Match: {score:.1%}</span>"
            else:
                score_badge = ""
            
            st.markdown(f"""
            <div style="background: white; border: 1px solid #dee2e6; border-radius: 8px; padding: 1rem; margin-bottom: 1rem; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                <h4 style="margin: 0 0 0.5rem 0; color: #1f77b4;">
                    📄 Result {index}: {score_badge}{title}
                </h4>
            </div>
            """, unsafe_allow_html=True)
        
        with header_col2:
            st.write("")  # Spacer
        
        with header_col3:
            if st.button("⭐", key=f"fav_{index}", help="Add to favorites"):
                add_to_favorites(item)
        
        with header_col4:
            # Use unique identifier for each item
            item_id = f"item_{item.get('file_name', index)}"
            
            if st.button("👁️", key=f"view_{index}", help="View details"):
                # Toggle expanded state
                if item_id in st.session_state.expanded_items:
                    st.session_state.expanded_items.remove(item_id)
                else:
                    st.session_state.expanded_items.add(item_id)
                st.rerun()  # Refresh to show the change
        
        # Determine if this item should be expanded
        item_id = f"item_{item.get('file_name', index)}"
        auto_expand = index <= st.session_state.user_preferences.get('auto_expand_results', 3)
        is_manually_expanded = item_id in st.session_state.expanded_items
        should_expand = auto_expand or is_manually_expanded
        
        # Expandable content - USE STREAMLIT'S BUILT-IN EXPANDER
        expanded_key = f"expander_{item_id}"
        
        # Initialize expander state in session state
        if expanded_key not in st.session_state:
            st.session_state[expanded_key] = should_expand
        
        # Create expander with proper state management
        with st.expander("📋 View Full Document Details", expanded=st.session_state[expanded_key]):
            # File information at top
            st.markdown("**📁 Document Information:**")
            col_info1, col_info2 = st.columns(2)
            
            with col_info1:
                st.text(f"📄 File: {item.get('file_name', 'N/A')}")
                st.text(f"📅 Date: {item.get('main_date', 'Unknown')}")
                if item.get('year'):
                    st.text(f"📊 Year: {item.get('year')}")
            
            with col_info2:
                st.text(f"📤 From: {item.get('sender', 'Unknown')}")
                st.text(f"📥 To: {item.get('recipient', 'Unknown')}")
                
                # Places
                places = item.get('places', [])
                if places:
                    st.text(f"📍 Places: {', '.join(places[:2])}")
                else:
                    st.text("📍 No places mentioned")
            
            # Topic information if available
            if item.get('topic_id') is not None and item.get('topic_terms'):
                st.markdown("**🏷️ Topic Analysis:**")
                topic_score = item.get('topic_score')
                
                # Ensure topic_score is a float for st.progress and min()
                if topic_score is None:
                    display_score = 0.0 # Default to 0.0 if None
                    # Optionally, log or indicate that the score was missing
                    # st.caption("Topic score not available") 
                else:
                    try:
                        display_score = float(topic_score)
                    except (ValueError, TypeError):
                        # logger.warning(f"Could not convert topic_score '{topic_score}' to float for item {item.get('file_name')}. Defaulting to 0.0.")
                        st.warning(f"Could not convert topic_score '{topic_score}' to float for item {item.get('file_name')}. Defaulting to 0.0.")
                        display_score = 0.0
                
                st.progress(min(display_score, 1.0)) 
                
                topic_name = item.get('topic_name', f"Topic {item['topic_id']}")
                st.caption(f"{topic_name}: {', '.join(item['topic_terms'][:5])}")
            
            st.markdown("---")
            
            # Content preview with better formatting
            st.markdown("**📄 Letter Content:**")
            content = item.get('full_text', 'No content available')
            
            if content and content not in ['No content available', 'N/A', '']:
                # Show content in text area for better readability
                st.text_area(
                    "Full Letter Text", 
                    value=content, 
                    height=300, 
                    key=f"letter_content_{item_id}_{index}",
                    help="Complete letter content from historical document"
                )
                
                # Document statistics
                st.caption(f"📊 Document length: {len(content)} characters | {len(content.split())} words")
            else:
                st.warning("📝 No readable content available for this document")
                
            # Action buttons at bottom
            col_action1, col_action2, col_action3 = st.columns(3)
            
            with col_action1:
                if st.button(f"📋 Copy Text", key=f"copy_{item_id}"):
                    st.session_state[f"copied_{item_id}"] = content
                    st.success("Text copied to session!")
            
            with col_action2:
                if st.button(f"⭐ Add to Favorites", key=f"fav_detail_{item_id}"):
                    add_to_favorites(item)
            
            with col_action3:
                if st.button(f"🔍 Search Similar", key=f"similar_{item_id}"):
                    # Trigger semantic search with this document's content
                    if len(content) > 50:
                        similar_query = content[:200] + "..."
                        st.session_state[f"search_similar_{item_id}"] = similar_query
                        st.info(f"Search for similar documents with: '{similar_query[:50]}...'")
                    else:
                        st.warning("Document too short for similarity search")

# --- Enhanced Utility Functions ---
def export_enhanced_results(results: List[Dict[str, Any]], search_type: str):
    """Enhanced CSV export with better formatting."""
    try:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"historical_search_results_{search_type.lower().replace(' ', '_')}_{timestamp}.csv"
        
        # Prepare comprehensive data
        export_data = []
        for item in results:
            row = {
                'Search_Type': search_type,
                'Export_Date': datetime.datetime.now().isoformat(),
                'File_Name': item.get('file_name', 'N/A'),
                'Document_Title': item.get('title', 'N/A'),
                'Date': item.get('main_date', 'N/A'),
                'Year': item.get('year', 'N/A'),
                'Sender': item.get('sender', 'N/A'),
                'Recipient': item.get('recipient', 'N/A'),
                'Places': '; '.join(item.get('places', [])),
                'Description': item.get('description', 'N/A'),
                'Topic_ID': item.get('topic_id', ''),
                'Topic_Score': item.get('topic_score', ''),
                'Topic_Terms': '; '.join(item.get('topic_terms', [])),
                'Similarity_Score': item.get('similarity_score', ''),
                'Content_Preview': (item.get('full_text', '')[:500] + "...") if len(item.get('full_text', '')) > 500 else item.get('full_text', ''),
                'Character_Count': len(item.get('full_text', ''))
            }
            export_data.append(row)
        
        df = pd.DataFrame(export_data)
        csv_data = df.to_csv(index=False)
        
        st.download_button(
            label=f"💾 Download {filename}",
            data=csv_data,
            file_name=filename,
            mime="text/csv",
            help=f"Download {len(results)} search results as CSV file"
        )
        
        st.success(f"✅ Export ready! {len(results)} results prepared for download.")
        
    except Exception as e:
        st.error(f"❌ Export failed: {e}")

def save_search_to_favorites(search_type: str, results: List[Dict[str, Any]]):
    """Save search results to user favorites."""
    try:
        timestamp = datetime.datetime.now().isoformat()
        favorite = {
            'timestamp': timestamp,
            'search_type': search_type,
            'result_count': len(results),
            'preview': results[0].get('title', 'No title')[:50] + "..." if results else "No results"
        }
        
        if 'favorites' not in st.session_state:
            st.session_state.favorites = []
        
        st.session_state.favorites.append(favorite)
        
        # Keep only last 20 favorites
        if len(st.session_state.favorites) > 20:
            st.session_state.favorites = st.session_state.favorites[-20:]
        
        st.success("⭐ Search saved to favorites!")
        
    except Exception as e:
        st.error(f"❌ Failed to save to favorites: {e}")

def add_to_favorites(item: Dict[str, Any]):
    """Add individual document to favorites."""
    try:
        timestamp = datetime.datetime.now().isoformat()
        favorite_doc = {
            'timestamp': timestamp,
            'type': 'document',
            'title': item.get('title', 'Untitled'),
            'file_name': item.get('file_name', 'Unknown'),
            'date': item.get('main_date', 'Unknown'),
            'sender': item.get('sender', 'Unknown')
        }
        
        if 'favorites' not in st.session_state:
            st.session_state.favorites = []
        
        # Check if already in favorites
        existing = any(
            fav.get('file_name') == item.get('file_name') 
            for fav in st.session_state.favorites 
            if fav.get('type') == 'document'
        )
        
        if not existing:
            st.session_state.favorites.append(favorite_doc)
            st.success("⭐ Document added to favorites!")
        else:
            st.info("📌 Document already in favorites")
        
    except Exception as e:
        st.error(f"❌ Failed to add to favorites: {e}")

# --- Enhanced Sidebar ---
def render_enhanced_sidebar():
    """Render an enhanced sidebar with modern design and better organization."""
    st.sidebar.markdown("""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 1rem; border-radius: 8px; margin-bottom: 1rem;">
        <h2 style="margin: 0; text-align: center;">⚙️ Control Panel</h2>
    </div>
    """, unsafe_allow_html=True)

    # Navigation tabs
    tab1, tab2, tab3 = st.sidebar.tabs(["🤖 AI Models", "📁 Data", "⭐ Favorites"])
    
    with tab1:
        render_models_section()
    
    with tab2:
        render_data_section()
    
    with tab3:
        render_favorites_section()

def render_models_section():
    """Render the AI models section in sidebar."""
    st.markdown("### Model Status")
    
    if st.button("🔄 Load/Reload Models", help="Load all AI models", use_container_width=True):
        st.session_state.models_loaded_flag = False
    
    ensure_models_loaded()

    # Model status indicators
    models = [
        ("🧠 spaCy NLP", st.session_state.spacy_nlp_instance),
        ("🎯 Sentence Transformer", st.session_state.sentence_model_instance), 
        ("❓ QA Pipeline", st.session_state.qa_pipeline_instance)
    ]
    
    for name, model in models:
        if model:
            st.success(f"✅ {name}")
        else:
            st.error(f"❌ {name}")

def render_data_section():
    """Render the data management section in sidebar."""
    st.markdown("### Document Management")
    
    config = get_config()
    xml_dir = st.text_input(
        "📁 XML Directory:", 
        value=config.XML_DIR_DEFAULT, 
        help="Path to historical documents"
    )
    
    force_reindex = st.checkbox(
        "🔄 Force Re-index", 
        value=False, 
        help="Rebuild entire index (slow)"
    )
    
    # Index status
    if os.path.exists(SAVED_INDEX_FILE):
        try:
            size_mb = os.path.getsize(SAVED_INDEX_FILE) / (1024 * 1024)
            st.info(f"📊 Index: {size_mb:.1f} MB")
        except:
            st.warning("📊 Index: Unable to read size")
    else:
        st.warning("📊 No index found")
    
    if st.button("📚 Load Documents", type="primary", use_container_width=True):
        load_enhanced_data(xml_dir, force_reindex)
    
    # Geocoding migration section
    st.markdown("### 🗺️ Map Integration")
    
    if st.session_state.letter_index:
        # Check if documents have geocoded places
        docs_with_geocoding = len([doc for doc in st.session_state.letter_index if doc.get('geocoded_places')])
        total_docs = len(st.session_state.letter_index)
        
        if docs_with_geocoding == 0:
            st.warning("No geocoded locations found")
            if st.button("🌍 Add Geocoding to Index", help="Add geographic coordinates to existing documents", use_container_width=True):
                with st.spinner("🗺️ Adding geocoding to existing documents..."):
                    success = add_geocoding_to_existing_index()
                    if success:
                        st.success("✅ Geocoding added successfully! Please reload the documents.")
                        # Refresh the session state
                        st.session_state.letter_index = None
                        st.rerun()
                    else:
                        st.error("❌ Geocoding migration failed. Check logs for details.")
        else:
            st.success(f"✅ {docs_with_geocoding}/{total_docs} documents have geocoding")
            if docs_with_geocoding < total_docs:
                if st.button("🔄 Update Geocoding", help="Add geocoding to remaining documents", use_container_width=True):
                    with st.spinner("🗺️ Updating geocoding for remaining documents..."):
                        success = add_geocoding_to_existing_index()
                        if success:
                            st.success("✅ Geocoding updated successfully! Please reload the documents.")
                            st.session_state.letter_index = None
                            st.rerun()
                        else:
                            st.error("❌ Geocoding update failed. Check logs for details.")
    else:
        st.info("Load documents first to enable geocoding")

def render_favorites_section():
    """Render the favorites section in sidebar."""
    st.markdown("### Your Favorites")
    
    if not st.session_state.get('favorites', []):
        st.info("No favorites yet. Start searching to add favorites!")
        return
    
    favorites = st.session_state.favorites[-10:]  # Show last 10
    
    for i, fav in enumerate(reversed(favorites)):
        if fav.get('type') == 'document':
            if st.button(f"📄 {fav.get('title', 'Untitled')[:30]}...", key=f"fav_doc_{i}"):
                st.info(f"Document: {fav.get('file_name', 'Unknown')}")
        else:
            if st.button(f"🔍 {fav.get('search_type', 'Search')} ({fav.get('result_count', 0)} results)", key=f"fav_search_{i}"):
                st.info(f"Search saved on {fav.get('timestamp', 'Unknown')[:10]}")

    if st.button("🗑️ Clear Favorites", help="Remove all favorites"):
        st.session_state.favorites = []
        st.success("Favorites cleared!")

def load_enhanced_data(xml_dir: str, force_reindex: bool):
    """Enhanced data loading with better progress tracking."""
    spacy_model, sentence_model, qa_pipeline = ensure_models_loaded()

    if not os.path.isdir(xml_dir):
        st.error(f"❌ Directory '{xml_dir}' does not exist.")
        return
    
    # Check for Gensim
    try:
        import gensim
        gensim_available = True
    except ImportError:
        gensim_available = False
    
    xml_files = [f for f in os.listdir(xml_dir) if f.lower().endswith('.xml')]
    
    if not xml_files:
        st.warning(f"⚠️ No XML files found in '{xml_dir}'")
        return
    
    # Limit to 3500 files for processing
    max_files_to_process = 3500
    if len(xml_files) > max_files_to_process:
        xml_files = xml_files[:max_files_to_process]
        st.info(f"📁 Found {len(os.listdir(xml_dir))} XML files, limiting to {max_files_to_process} for processing")
    else:
        st.info(f"📁 Found {len(xml_files)} XML files")
    
    try:
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        if force_reindex or not os.path.exists(SAVED_INDEX_FILE):
            files_for_indexing = [os.path.join(xml_dir, f) for f in xml_files]
            status_text.text(f"🔄 Processing {len(files_for_indexing)} files...")
        else:
            files_for_indexing = []
            status_text.text("📖 Loading existing index...")
        
        progress_bar.progress(20)

        st.session_state.letter_index = load_or_create_index(
            xml_files_to_process=files_for_indexing,
            sentence_model=sentence_model,
            spacy_nlp_model=spacy_model,
            index_path=SAVED_INDEX_FILE,
            force_reindex=force_reindex
        )

        progress_bar.progress(80)
        
        if st.session_state.letter_index:
            status_text.text("✅ Index loaded successfully!")
            progress_bar.progress(100)
            
            st.success(f"✅ Loaded {len(st.session_state.letter_index)} documents")
            
            # --- Run Modern Topic Discovery ---
            if chat.MODERN_TOPICS_AVAILABLE and st.session_state.sentence_model_instance:
                with st.spinner("🔬 Discovering modern topics from embeddings..."):
                    chat.run_modern_topic_discovery(st.session_state.letter_index)
                if chat.global_modern_topic_results and chat.global_lda_model:
                    st.info(f"💡 Modern Topics Discovered: {chat.global_lda_model.num_topics} themes found.")
                    # Optionally, log some sample topics to Streamlit UI if desired
                    # sample_display_topics = list(chat.global_modern_topic_results.get('topics', {}).values())[:3]
                    # if sample_display_topics:
                    #    st.caption(f"Sample themes: {[(t.get('name'), t.get('keywords')[:2]) for t in sample_display_topics]}")
                else:
                    st.warning("Modern topic discovery ran but no results were stored.")
            # --- End Modern Topic Discovery ---
            
            # Check if topics are available (don't fail if they're not)
            if chat.global_lda_model: # This will now reflect the modern topic model
                st.info(f"🏷️ {chat.global_lda_model.num_topics} topics available (modern)")
            else:
                st.info("ℹ️ Topic analysis could not be completed.") # Changed message slightly
            
        else:
            st.error("❌ Failed to load index")
        
        # Clean up progress indicators
        time.sleep(1)
        progress_bar.empty()
        status_text.empty()
        
    except Exception as e:
        st.error(f"❌ Loading failed: {e}")

# --- Enhanced Search Interface ---
def render_enhanced_search_interface():
    """Render the main search interface with modern design."""
    if st.session_state.letter_index is None:
        st.markdown("""
        <div style="text-align: center; padding: 3rem; background: #f8f9fa; border-radius: 15px; border: 2px dashed #dee2e6;">
            <h2 style="color: #6c757d;">📁 No Documents Loaded</h2>
            <p style="color: #6c757d; font-size: 1.1em;">Please use the sidebar to load your historical documents first.</p>
            <p style="color: #6c757d;">👈 Click "Load Documents" in the Control Panel</p>
        </div>
        """, unsafe_allow_html=True)
        return
    
    # Search header
    st.markdown("""
    <div style="background: linear-gradient(90deg, #28a745, #20c997); color: white; padding: 2rem; border-radius: 12px; margin-bottom: 2rem; text-align: center;">
        <h1 style="margin: 0; font-size: 2.5rem;">🔍 Smart Document Search</h1>
        <p style="margin: 0.5rem 0 0 0; font-size: 1.2em; opacity: 0.9;">Find historical letters using AI-powered search</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Search form
    col1, col2 = st.columns([1, 2])
    
    with col1:
        search_type = st.selectbox(
            "🎯 Search Method:", 
            get_available_search_types(),
            help="Choose how you want to search the documents"
        )
    
    with col2:
        query = st.text_input(
            "🔍 Your Query:", 
            placeholder=get_search_placeholder(search_type),
            help=get_search_help(search_type),
            key="main_search_input"
        )
    
    # Search button and options
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        search_clicked = st.button(
            "🚀 Search Documents", 
            type="primary", 
            use_container_width=True,
            help="Execute your search query"
        )
    
    # Advanced options
    with st.expander("⚙️ Search Options"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            max_results = st.slider("📊 Max Results:", 5, 100, 25)
        
        with col2:
            st.session_state.user_preferences['auto_expand_results'] = st.slider(
                "📖 Auto-expand results:", 0, 10, 3
            )
        
        with col3:
            sort_option = st.selectbox("📈 Sort by:", ["Relevance", "Date", "Alphabetical"])
    
    # Quick search buttons
    if st.session_state.letter_index:
        st.markdown("### 🎯 Quick Searches")
        
        quick_searches = [
            ("🏛️ Civil War", "Semantic Search", "civil war battles military conflict"),
            ("📜 Personal Letters", "Smart Search (spaCy)", "personal family letters home"),
            ("📅 1863 Events", "Year", "1863"),
            ("🗺️ Virginia", "Place", "Virginia")
        ]
        
        cols = st.columns(len(quick_searches))
        
        for i, (label, search_type_quick, query_quick) in enumerate(quick_searches):
            with cols[i]:
                if st.button(label, use_container_width=True, key=f"quick_{i}"):
                    perform_enhanced_search(search_type_quick, query_quick, max_results)
    
    # Execute search
    if search_clicked and query.strip():
        perform_enhanced_search(search_type, query, max_results)
    
    # Search history and suggestions
    render_search_history()

def get_available_search_types() -> List[str]:
    """Get list of available search types based on loaded models."""
    base_types = ["Keyword", "Sender", "Recipient", "Year", "Place"]
    
    if st.session_state.spacy_nlp_instance:
        base_types.append("Smart Search (spaCy)")
    
    if st.session_state.sentence_model_instance:
        base_types.append("Semantic Search")
    
    if st.session_state.qa_pipeline_instance and st.session_state.sentence_model_instance:
        base_types.append("Question Answering")
    
    if st.session_state.sentence_model_instance:
        base_types.append("Hybrid Search")
    
    # Topic modeling - check for both traditional and modern approaches
    try:
        import gensim
        gensim_available = True
    except ImportError:
        gensim_available = False
    
    # Add topic search if we have any topic modeling capability
    if st.session_state.letter_index and len(st.session_state.letter_index) > 0:
        # Check if traditional topic modeling is available
        if chat.global_lda_model and gensim_available:
            base_types.extend(["Topic ID", "Topic Terms"])
        elif st.session_state.sentence_model_instance:
            # Modern topic discovery using embeddings
            base_types.extend(["Topic Discovery", "Topic Clustering"])
    
    return base_types

def render_search_history():
    """Render search history with enhanced UI."""
    if st.session_state.search_history:
        with st.expander(f"📜 Recent Searches ({len(st.session_state.search_history)})"):
            # Show last 10 searches
            recent = list(reversed(st.session_state.search_history[-10:]))
            
            for i, (search_type_hist, query_hist) in enumerate(recent):
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    if st.button(
                        f"🔍 {search_type_hist}: {query_hist[:50]}{'...' if len(query_hist) > 50 else ''}", 
                        key=f"hist_{i}",
                        use_container_width=True
                    ):
                        perform_enhanced_search(search_type_hist, query_hist, 25)
                
                with col2:
                    timestamp = datetime.datetime.now().strftime("%H:%M")
                    st.caption(f"⏰ {timestamp}")

def perform_enhanced_search(search_type: str, query: str, max_results: int = 25):
    """Enhanced search execution with better UX."""
    if not query.strip():
        st.warning("⚠️ Please enter a search query.")
        return
    
    # Update search stats
    st.session_state.search_stats['total_searches'] += 1
    
    # Add to search history
    search_entry = (search_type, query)
    if search_entry not in st.session_state.search_history:
        st.session_state.search_history.append(search_entry)
        
    # Keep history manageable
    max_history = st.session_state.get('user_preferences', {}).get('max_history', Config.MAX_SEARCH_HISTORY)
    if len(st.session_state.search_history) > max_history:
        st.session_state.search_history = st.session_state.search_history[-max_history:]

    # Initialize general stats if they somehow got removed (defensive)
    if 'total_searches' not in st.session_state.search_stats:
        st.session_state.search_stats['total_searches'] = 0
    # 'successful_searches' is updated in display_enhanced_results globally
    if 'searches_by_type' not in st.session_state.search_stats:
         st.session_state.search_stats['searches_by_type'] = {}
    
    st.session_state.search_stats['total_searches'] += 1
    
    # Initialize/update stats for this search_type
    current_type_stats = st.session_state.search_stats['searches_by_type'].setdefault(search_type, {
        "total": 0, 
        "successful": 0, 
        "zero_results": 0,
        "total_results_returned": 0,
        "queries": [] 
    })
    
    current_type_stats['total'] += 1
    current_type_stats['queries'].append(query) 
    if len(current_type_stats['queries']) > 50: # Limit stored queries history per type
        current_type_stats['queries'] = current_type_stats['queries'][-50:]
    
    spacy_model, sentence_model, qa_pipeline = ensure_models_loaded()
    results = []
    is_semantic = False
    
    try:
        with st.spinner(f"🔍 Searching using {search_type}..."):
            # Progress tracking for longer operations
            progress_placeholder = st.empty()
            
            if search_type == "Keyword":
                results = search_by_keyword(st.session_state.letter_index, query)
            elif search_type == "Sender":
                results = search_by_sender(st.session_state.letter_index, query)
            elif search_type == "Recipient":
                results = search_by_recipient(st.session_state.letter_index, query)
            elif search_type == "Year":
                results = search_by_year(st.session_state.letter_index, query)
            elif search_type == "Place":
                results = search_by_place(st.session_state.letter_index, query)
            elif search_type == "Smart Search (spaCy)":
                if spacy_model:
                    progress_placeholder.info("🧠 Processing natural language query...")
                    try:
                        filters = process_smart_query(query, spacy_model)
                        st.info(f"🎯 Extracted filters: {filters}")
                        results = execute_smart_search(st.session_state.letter_index, filters)
                    except Exception as e:
                        st.error(f"❌ Smart search error: {e}")
                        # Fallback to keyword search
                        st.info("🔄 Falling back to keyword search...")
                        results = search_by_keyword(st.session_state.letter_index, query)
                else:
                    st.error("❌ spaCy model not available")
                    return
            elif search_type == "Semantic Search":
                if sentence_model:
                    progress_placeholder.info("🎯 Computing semantic similarity...")
                    try:
                        results = execute_semantic_search(
                            st.session_state.letter_index, query, sentence_model, top_n=max_results
                        )
                        is_semantic = True
                    except Exception as e:
                        st.error(f"❌ Semantic search error: {e}")
                        # Fallback to keyword search
                        st.info("🔄 Falling back to keyword search...")
                        results = search_by_keyword(st.session_state.letter_index, query)
                else:
                    st.error("❌ Sentence model not available")
                    return
            elif search_type == "Question Answering":
                if qa_pipeline and sentence_model:
                    progress_placeholder.info("❓ Finding answers in documents...")
                    try:
                        answers = execute_extractive_qa(
                            st.session_state.letter_index, query, qa_pipeline, sentence_model, top_n=max_results
                        )
                        display_qa_results_enhanced(answers)
                        return
                    except Exception as e:
                        st.error(f"❌ QA search error: {e}")
                        # Fallback to semantic search
                        st.info("🔄 Falling back to semantic search...")
                        if sentence_model:
                            results = execute_semantic_search(
                                st.session_state.letter_index, query, sentence_model, top_n=max_results
                            )
                            is_semantic = True
                        else:
                            results = search_by_keyword(st.session_state.letter_index, query)
                else:
                    st.error("❌ QA pipeline not available")
                    return
            elif search_type == "Hybrid Search":
                if sentence_model:
                    progress_placeholder.info("🔄 Running hybrid search...")
                    try:
                        results = execute_hybrid_search(
                            st.session_state.letter_index, query, sentence_model, top_n=max_results
                        )
                        display_hybrid_results_enhanced(results)
                        return
                    except Exception as e:
                        st.error(f"❌ Hybrid search error: {e}")
                        # Fallback to keyword + semantic combination
                        st.info("🔄 Falling back to keyword search...")
                        results = search_by_keyword(st.session_state.letter_index, query)
                else:
                    st.error("❌ Sentence model not available")
                    return
            elif search_type == "Topic ID":
                if chat.global_lda_model:
                    try:
                        topic_id_query = int(query)
                        num_topics = chat.global_lda_model.num_topics
                        if not (0 <= topic_id_query < num_topics):
                            st.error(f"Invalid Topic ID. Please enter a number between 0 and {num_topics-1 if num_topics > 0 else 0}.")
                            # Clear results or do not proceed with search if error occurs
                            st.session_state.last_search_results = []
                            current_type_stats['zero_results'] += 1 # Count as zero result for this type
                            display_enhanced_results([], search_type, is_semantic) # Display no results
                            return 
                        results = search_by_topic_id(st.session_state.letter_index, query)
                    except ValueError:
                        st.error("Invalid Topic ID. Please enter a number.")
                        st.session_state.last_search_results = []
                        current_type_stats['zero_results'] += 1 
                        display_enhanced_results([], search_type, is_semantic)
                        return
                    except Exception as e:
                        st.error(f"❌ Topic ID search error: {e}")
                        results = []
                else:
                    st.error("❌ Topic model not available")
                    # Clear results or do not proceed
                    st.session_state.last_search_results = []
                    current_type_stats['zero_results'] += 1
                    display_enhanced_results([], search_type, is_semantic)
                    return
            elif search_type == "Topic Terms":
                if chat.global_lda_model:
                    try:
                        results = search_by_topic_terms(st.session_state.letter_index, query)
                    except Exception as e:
                        st.error(f"❌ Topic terms search error: {e}")
                        results = []
                else:
                    st.error("❌ Topic model not available")
                    return
            elif search_type == "Topic Discovery":
                if sentence_model:
                    try:
                        progress_placeholder.info("🔍 Discovering topics using modern AI clustering...")
                        # Use the modern topic discovery system
                        from modern_topics import fast_topic_modeling
                        topic_results = fast_topic_modeling(st.session_state.letter_index, n_topics=15)
                        
                        # Display topic discovery results
                        st.markdown("### 🏷️ Discovered Topics")
                        for topic_id, topic_info in topic_results['topics'].items():
                            name = topic_info.get('name', f'Topic {topic_id}')
                            keywords = topic_info.get('keywords', [])[:5]
                            doc_count = topic_info.get('doc_count', 0)
                            
                            with st.expander(f"📋 {name} ({doc_count} documents)"):
                                st.write(f"**Keywords:** {', '.join(keywords)}")
                                if 'representative_text' in topic_info:
                                    st.write(f"**Sample text:** {topic_info['representative_text']}")
                        
                        # Return documents from the first topic as results
                        if topic_results['topics']:
                            first_topic_id = list(topic_results['topics'].keys())[0]
                            results = [doc for doc in st.session_state.letter_index 
                                     if doc.get('topic_id') == first_topic_id]
                        
                    except Exception as e:
                        st.error(f"❌ Modern topic discovery error: {e}")
                        st.info("🔄 Falling back to semantic search...")
                        results = execute_semantic_search(
                            st.session_state.letter_index, query, sentence_model, top_n=max_results
                        )
                        is_semantic = True
                else:
                    st.error("❌ Sentence model not available for topic discovery")
                    return
            elif search_type == "Topic Clustering":
                if sentence_model:
                    try:
                        progress_placeholder.info("🎯 Clustering documents by topic similarity...")
                        # Find documents similar to the query topic
                        results = execute_semantic_search(
                            st.session_state.letter_index, f"topic about {query}", sentence_model, top_n=max_results
                        )
                        is_semantic = True
                        
                        # Group results by semantic similarity
                        st.info(f"🔍 Found {len(results)} documents related to '{query}' topic")
                        
                    except Exception as e:
                        st.error(f"❌ Topic clustering error: {e}")
                        results = search_by_keyword(st.session_state.letter_index, query)
                else:
                    st.error("❌ Sentence model not available for clustering")
                    return
            
            progress_placeholder.empty()
        
        # Store and display results
        st.session_state.last_search_results = results
        
        # Update per-type stats based on results
        if results:
            current_type_stats['successful'] += 1
            current_type_stats['total_results_returned'] += len(results)
        else:
            # This case is for when search types other than Topic ID (already handled) return no results
            # or if an error within a search type leads to empty results before exception.
            if not (search_type == "Topic ID" and ('error' in st.session_state or 'Invalid Topic ID' in [m.body for m in st.container()._parent.delta.new_elements[0].markdown])): # Avoid double counting for Topic ID errors
                 current_type_stats['zero_results'] += 1

        display_enhanced_results(results, search_type, is_semantic) # This handles global successful_searches
        
    except Exception as e:
        st.error(f"❌ Search failed: {e}")
        # Log as a failed/zero_results search for the type if an exception occurs
        # Ensure current_type_stats is available
        if 'searches_by_type' in st.session_state.search_stats and \
           search_type in st.session_state.search_stats['searches_by_type']:
            st.session_state.search_stats['searches_by_type'][search_type]['zero_results'] +=1 # or a specific 'error_count'

        with st.expander("🔍 Error Details"):
            st.code(traceback.format_exc())

# --- Enhanced QA and Hybrid Results Display ---
def display_qa_results_enhanced(answers):
    """Enhanced QA results display."""
    if not answers:
        st.markdown("""
        <div style="text-align: center; padding: 2rem; background: #fff3cd; border-radius: 10px; border-left: 4px solid #ffc107;">
            <h3 style="color: #856404;">🤔 No Direct Answers Found</h3>
            <p>Try rephrasing your question or using a different search method.</p>
        </div>
        """, unsafe_allow_html=True)
        return
    
    st.markdown(f"""
    <div style="background: linear-gradient(90deg, #007bff, #6610f2); color: white; padding: 1rem; border-radius: 8px; margin-bottom: 1rem;">
        <h3 style="margin: 0; color: white;">💬 Found {len(answers)} Answer(s)</h3>
        <p style="margin: 0; opacity: 0.9;">Direct answers extracted from historical documents</p>
    </div>
    """, unsafe_allow_html=True)
    
    for i, answer in enumerate(answers, 1):
        confidence = answer.get('confidence', 0)
        confidence_color = "#28a745" if confidence > 0.8 else "#ffc107" if confidence > 0.5 else "#dc3545"
        
        with st.expander(f"💬 Answer {i} - Confidence: {confidence:.1%}", expanded=(i <= 2)):
            st.markdown(f"""
            <div style="background: {confidence_color}; color: white; padding: 0.5rem; border-radius: 5px; margin-bottom: 1rem;">
                <strong>Answer:</strong> {answer.get('answer', 'N/A')}
            </div>
            """, unsafe_allow_html=True)
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                context = answer.get('context_snippet', '')
                if context:
                    st.markdown("**📄 Context:**")
                    st.text_area("", value=context, height=120, disabled=True, key=f"qa_context_{i}")
            
            with col2:
                st.markdown("**📋 Source:**")
                st.write(f"📄 {answer.get('document_title', 'Unknown')}")
                st.write(f"📅 {answer.get('document_year', 'Unknown')}")
                st.write(f"👤 {answer.get('document_sender', 'Unknown')}")

def display_hybrid_results_enhanced(results):
    """Enhanced hybrid search results display."""
    if not results:
        st.info("🔍 No matching documents found with hybrid search.")
        return
    
    st.markdown(f"""
    <div style="background: linear-gradient(90deg, #6f42c1, #e83e8c); color: white; padding: 1rem; border-radius: 8px; margin-bottom: 1rem;">
        <h3 style="margin: 0; color: white;">🔄 Found {len(results)} Documents (Hybrid Search)</h3>
        <p style="margin: 0; opacity: 0.9;">Combined keyword + semantic similarity results</p>
    </div>
    """, unsafe_allow_html=True)
    
    display_enhanced_results(results, "Hybrid Search", False)

# --- Navigation and Main Function ---
def render_navigation():
    """Render top navigation."""
    col1, col2, col3, col4, col5 = st.columns([1, 1, 1, 1, 1])
    
    with col1:
        if st.button("🏠 Home", use_container_width=True):
            st.session_state.current_page = 'home'
            st.rerun()
    
    with col2:
        if st.button("🔍 Search", use_container_width=True):
            st.session_state.current_page = 'search'
            st.rerun()
    
    with col3:
        if st.button("🌍 Map", use_container_width=True):
            st.session_state.current_page = 'map'
            st.rerun()
    
    with col4:
        if st.button("📊 Analytics", use_container_width=True):
            st.session_state.current_page = 'analytics'
            st.rerun()
    
    with col5:
        if st.button("ℹ️ Help", use_container_width=True):
            st.session_state.current_page = 'help'
            st.rerun()

def render_map_explorer_page():
    """Render the interactive map explorer page with geocoded locations."""
    st.title("🌍 Historical Letter Map Explorer")
    
    if not st.session_state.letter_index:
        st.markdown("""
        <div style="text-align: center; padding: 3rem; background: #f8f9fa; border-radius: 15px; border: 2px dashed #dee2e6;">
            <h2 style="color: #6c757d;">📁 No Documents Loaded</h2>
            <p style="color: #6c757d; font-size: 1.1em;">Please use the sidebar to load your historical documents first.</p>
            <p style="color: #6c757d;">👈 Click "Load Documents" in the Control Panel</p>
        </div>
        """, unsafe_allow_html=True)
        return
    
    # Collect all geocoded locations from the index
    all_locations = []
    location_doc_mapping = {}  # Map locations to documents that reference them
    
    for doc in st.session_state.letter_index:
        geocoded_places = doc.get('geocoded_places', [])
        
        for place in geocoded_places:
            if place.get('geocoded', False) and place.get('lat') and place.get('lon'):
                location_key = f"{place['lat']},{place['lon']}"
                
                if location_key not in location_doc_mapping:
                    location_doc_mapping[location_key] = {
                        'place_info': place,
                        'documents': []
                    }
                
                # Add document info
                doc_info = {
                    'title': doc.get('title', 'Untitled'),
                    'date': doc.get('main_date', 'Unknown'),
                    'year': doc.get('year', 'Unknown'),
                    'sender': doc.get('sender', 'Unknown'),
                    'recipient': doc.get('recipient', 'Unknown'),
                    'file_name': doc.get('file_name', 'Unknown')
                }
                location_doc_mapping[location_key]['documents'].append(doc_info)
    
    if not location_doc_mapping:
        st.warning("🗺️ No geocoded locations found in the loaded documents. The system may need to re-index with geocoding enabled.")
        return
    
    # Display statistics
    total_locations = len(location_doc_mapping)
    total_docs_with_locations = len([doc for doc in st.session_state.letter_index if doc.get('geocoded_places')])
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📍 Unique Locations", total_locations)
    with col2:
        st.metric("📄 Documents with Locations", total_docs_with_locations)
    with col3:
        st.metric("📚 Total Documents", len(st.session_state.letter_index))
    with col4:
        coverage_pct = (total_docs_with_locations / len(st.session_state.letter_index)) * 100
        st.metric("🎯 Geographic Coverage", f"{coverage_pct:.1f}%")
    
    # Filter options
    st.markdown("### 🎛️ Map Filters")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # Year filter
        years = [doc.get('year') for doc in st.session_state.letter_index if doc.get('year')]
        if years:
            min_year, max_year = min(years), max(years)
            year_range = st.slider(
                "📅 Filter by Year Range", 
                min_value=min_year, 
                max_value=max_year, 
                value=(min_year, max_year),
                help="Filter letters by year"
            )
        else:
            year_range = None
    
    with col2:
        # Marker style
        marker_style = st.selectbox(
            "📌 Marker Style",
            ["Default", "Clustered", "Heatmap"],
            help="Choose how to display locations on the map"
        )
    
    # Create the map
    st.markdown("### 🗺️ Interactive Map")
    
    # Calculate map center (average of all coordinates)
    lats = [float(loc['place_info']['lat']) for loc in location_doc_mapping.values()]
    lons = [float(loc['place_info']['lon']) for loc in location_doc_mapping.values()]
    
    if lats and lons:
        center_lat = sum(lats) / len(lats)
        center_lon = sum(lons) / len(lons)
    else:
        # Default to southeastern US (Civil War region)
        center_lat, center_lon = 33.0, -83.0
    
    # Create base map
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=6,
        tiles="OpenStreetMap"
    )
    
    # Add markers to map
    for location_key, location_data in location_doc_mapping.items():
        place_info = location_data['place_info']
        documents = location_data['documents']
        
        # Apply year filter if specified
        if year_range:
            filtered_docs = [
                doc for doc in documents 
                if isinstance(doc.get('year'), int) and year_range[0] <= doc['year'] <= year_range[1]
            ]
        else:
            filtered_docs = documents
        
        if not filtered_docs:
            continue  # Skip this location if no documents pass the filter
        
        lat, lon = float(place_info['lat']), float(place_info['lon'])
        
        # Create popup content
        popup_html = f"""
        <div style="width: 300px;">
            <h4>📍 {place_info['name']}</h4>
            <p><strong>Address:</strong> {place_info.get('raw_address', 'N/A')}</p>
            <p><strong>Letters from this location:</strong> {len(filtered_docs)}</p>
            <hr>
        """
        
        # Add document details (limit to first 3 to avoid too large popup)
        for i, doc in enumerate(filtered_docs[:3]):
            popup_html += f"""
            <div style="margin-bottom: 10px; padding: 5px; background: #f8f9fa; border-radius: 3px;">
                <strong>📄 {doc['title'][:50]}{'...' if len(doc['title']) > 50 else ''}</strong><br>
                <small>📅 {doc['date']} | 👤 {doc['sender']} → {doc['recipient']}</small>
            </div>
            """
        
        if len(filtered_docs) > 3:
            popup_html += f"<p><em>... and {len(filtered_docs) - 3} more letters</em></p>"
        
        popup_html += "</div>"
        
        # Determine marker color based on number of documents
        if len(filtered_docs) >= 5:
            color = 'red'  # Many letters
        elif len(filtered_docs) >= 2:
            color = 'orange'  # Several letters
        else:
            color = 'blue'  # Single letter
        
        # Add marker
        folium.Marker(
            location=[lat, lon],
            popup=folium.Popup(popup_html, max_width=400),
            tooltip=f"{place_info['name']} ({len(filtered_docs)} letters)",
            icon=folium.Icon(color=color, icon='envelope', prefix='fa')
        ).add_to(m)
    
    # Display the map
    map_data = st_folium(m, width=700, height=500, returned_objects=["last_object_clicked"])
    
    # Handle map interactions
    if map_data['last_object_clicked']:
        st.info("💡 Click on map markers to view letter details!")
    
    # Additional information
    with st.expander("📖 Map Legend & Information"):
        st.markdown("""
        **Marker Colors:**
        - 🔴 **Red**: 5+ letters from this location
        - 🟠 **Orange**: 2-4 letters from this location  
        - 🔵 **Blue**: 1 letter from this location
        
        **How to Use:**
        - Click on markers to see letter details
        - Use the year slider to filter letters by time period
        - Zoom and pan to explore different regions
        - The map shows all successfully geocoded locations from your document collection
        
        **Data Source:** Locations are geocoded using OpenStreetMap/Nominatim service.
        
        **Note:** This is a proof-of-concept map integration. Future enhancements could include:
        - Historical base maps overlay
        - Clustering for dense areas
        - Heatmap visualization
        - Search result filtering on map
        - Timeline animation
        """)
    
    # Show some sample geocoded locations for debugging/info
    if st.checkbox("🔍 Show Geocoding Details", help="Display technical details about geocoded locations"):
        st.markdown("#### Geocoded Locations Sample")
        sample_locations = list(location_doc_mapping.items())[:5]  # Show first 5
        
        for location_key, location_data in sample_locations:
            place_info = location_data['place_info']
            documents = location_data['documents']
            
            with st.expander(f"📍 {place_info['name']} ({len(documents)} documents)"):
                st.json({
                    'coordinates': {'lat': place_info['lat'], 'lon': place_info['lon']},
                    'raw_address': place_info.get('raw_address', 'N/A'),
                    'documents': [doc['file_name'] for doc in documents[:3]]
                })

def render_analytics_page():
    """Render analytics and statistics page."""
    st.title("📊 Search Analytics & Statistics")
    
    if not st.session_state.letter_index:
        st.warning("📁 Load documents first to see analytics")
        return
    
    # Document statistics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📚 Total Documents", len(st.session_state.letter_index))
    
    with col2:
        total_searches = st.session_state.search_stats.get('total_searches', 0)
        st.metric("🔍 Total Searches", total_searches)
    
    with col3:
        successful = st.session_state.search_stats.get('successful_searches', 0)
        st.metric("✅ Successful Searches", successful)
    
    with col4:
        if total_searches > 0:
            success_rate = (successful / total_searches) * 100
            st.metric("📈 Success Rate", f"{success_rate:.1f}%")
        else:
            st.metric("📈 Success Rate", "N/A")
    
    # Most searched terms
    if st.session_state.search_history:
        st.subheader("🔥 Popular Search Terms")
        search_terms = [query for _, query in st.session_state.search_history]
        
        # Simple frequency count
        from collections import Counter
        term_counts = Counter(search_terms)
        
        for term, count in term_counts.most_common(10):
            col1, col2 = st.columns([3, 1])
            with col1:
                st.write(f"'{term}'")
            with col2:
                st.write(f"{count} times")

    st.subheader("📊 Search Performance by Type")
    searches_by_type_data = st.session_state.search_stats.get('searches_by_type', {})
    if searches_by_type_data:
        data_for_df = []
        for search_type_key, stats_val in searches_by_type_data.items():
            total_type_searches = stats_val.get('total', 0)
            successful_type_searches = stats_val.get('successful', 0)
            zero_results_type_searches = stats_val.get('zero_results', 0)
            
            avg_results = (stats_val.get('total_results_returned', 0) / successful_type_searches) \
                if successful_type_searches > 0 else 0.0
            
            success_rate_type = (successful_type_searches / total_type_searches * 100) \
                if total_type_searches > 0 else 0.0

            data_for_df.append({
                "Search Type": search_type_key,
                "Total Searches": total_type_searches,
                "Successful": successful_type_searches,
                "Zero Results": zero_results_type_searches,
                "Success Rate (%)": f"{success_rate_type:.1f}",
                "Avg. Results": f"{avg_results:.1f}" # (when successful) implied
            })
        
        if data_for_df:
            df_stats = pd.DataFrame(data_for_df)
            st.dataframe(df_stats.set_index("Search Type"))
        else:
            st.info("No detailed search type statistics yet.")
    else:
        st.info("No search type statistics available yet.")

    st.subheader("🏷️ Topic Modeling Insights")
    if hasattr(chat, 'global_modern_topic_results') and chat.global_modern_topic_results and \
       hasattr(chat, 'global_lda_model') and chat.global_lda_model:
        
        num_discovered_topics = chat.global_lda_model.num_topics
        st.metric("Discovered Topics", num_discovered_topics)

        topic_details_dict = chat.global_modern_topic_results.get('topics', {})
        noise_docs_count = chat.global_modern_topic_results.get('noise_docs', 0)

        if topic_details_dict:
            topic_data_list = []
            for topic_id_key, info_val in topic_details_dict.items():
                topic_data_list.append({
                    "Topic ID": info_val.get('id', topic_id_key),
                    "Name": info_val.get('name', f"Topic {topic_id_key}"),
                    "Documents": info_val.get('doc_count', 0),
                    "Keywords": ", ".join(info_val.get('keywords', [])[:5]) # Show top 5
                })
            
            if topic_data_list:
                df_topics = pd.DataFrame(topic_data_list)
                st.dataframe(df_topics.set_index("Topic ID"))

                st.markdown("#### Document Distribution per Topic")
                chart_data_df = df_topics[df_topics["Documents"] > 0][["Name", "Documents"]].set_index("Name")

                if not chart_data_df.empty:
                    st.bar_chart(chart_data_df)
                else:
                    st.info("No documents assigned to topics for chart.")
            else:
                st.info("No topic details found in results.")

        if noise_docs_count > 0:
            st.info(f"{noise_docs_count} documents were considered noise and not assigned to any specific topic.")
        elif not topic_details_dict : # only show this if no topics were found AND no noise docs
             st.info("No topics were discovered, and no documents were marked as noise.")
    else:
        st.info("No topic modeling results available to display.")

def render_help_page():
    """Render help and documentation page."""
    st.title("ℹ️ Help & Documentation")
    
    tab1, tab2, tab3 = st.tabs(["🚀 Getting Started", "🔍 Search Types", "❓ FAQ"])
    
    with tab1:
        st.markdown("""
        ## Welcome to Historical Document Analyzer! 
        
        ### Quick Start:
        1. **Load Documents**: Use the sidebar to select and load your XML files
        2. **Choose Search Type**: Select from keyword, semantic, or AI-powered search
        3. **Enter Query**: Type your search terms naturally
        4. **Explore Results**: Click to expand, save favorites, or export findings
        
        ### Tips for Better Results:
        - Use **Smart Search** for natural language questions
        - Try **Semantic Search** to find similar concepts
        - Use **Question Answering** for specific information
        - Export results to CSV for further analysis
        """)
    
    with tab2:
        st.markdown("""
        ## Search Types Explained
        
        ### 🔍 **Keyword Search**
        Simple text matching across all document fields.
        *Example: "battle", "Richmond", "supplies"*
        
        ### 🧠 **Smart Search (spaCy)**
        AI understands entities and context in your questions.
        *Example: "Who wrote about battles in Virginia?"*
        
        ### 🎯 **Semantic Search**
        Finds documents with similar meaning, not just matching words.
        *Example: "military conflict" will find "battle", "war", "fighting"*
        
        ### ❓ **Question Answering**
        Get direct answers extracted from documents.
        *Example: "What supplies were needed?" → "ammunition and food"*
        
        ### 🔄 **Hybrid Search**
        Combines keyword matching with semantic similarity.
        *Best of both worlds for comprehensive results.*
        """)
    
    with tab3:
        st.markdown("""
        ## Frequently Asked Questions
        
        **Q: Why are some search types unavailable?**
        A: Some features require AI models to be loaded. Use the "Load Models" button in the sidebar.
        
        **Q: How do I enable topic modeling?**
        A: Topic modeling requires Gensim. Install it with: `pip install gensim`
        
        **Q: Can I search multiple terms?**
        A: Yes! Use phrases like "battle AND supplies" or "Richmond OR Virginia"
        
        **Q: How do I export my results?**
        A: Click the "Export Results" button after any search to download a CSV file.
        
        **Q: What file formats are supported?**
        A: Currently supports TEI XML format historical documents.
        """)

def main():
    """Enhanced main application function with navigation."""
    try:
        # Initialize session state
        initialize_session_state()
        
        # Show welcome screen if first visit
        if not st.session_state.tutorial_shown and st.session_state.current_page == 'search':
            st.session_state.current_page = 'home'
        
        # Render sidebar
        render_enhanced_sidebar()
        
        # Render navigation
        render_navigation()
        
        # Render current page
        if st.session_state.current_page == 'home':
            render_welcome_screen()
        elif st.session_state.current_page == 'search':
            render_enhanced_search_interface()
        elif st.session_state.current_page == 'map':
            render_map_explorer_page()
        elif st.session_state.current_page == 'analytics':
            render_analytics_page()
        elif st.session_state.current_page == 'help':
            render_help_page()
        
        # Footer
        st.markdown("---")
        st.markdown("""
        <div style="text-align: center; color: #666; padding: 1rem;">
            <p>📜 Historical Document Analyzer | Powered by AI & Advanced NLP</p>
            <p><small>Built with Streamlit, spaCy, Sentence Transformers & Gensim</small></p>
        </div>
        """, unsafe_allow_html=True)
        
    except Exception as e:
        st.error(f"🚨 **Application Error**: {e}")
        st.code(traceback.format_exc())

# --- Legacy function compatibility ---
def display_streamlit_results(results: List[Dict[str, Any]], is_semantic_search: bool = False):
    """Legacy compatibility function."""
    display_enhanced_results(results, "Legacy Search", is_semantic_search)

def export_results_to_csv(results: List[Dict[str, Any]]):
    """Legacy compatibility function."""
    export_enhanced_results(results, "Legacy Export")

def render_sidebar():
    """Legacy compatibility function."""
    render_enhanced_sidebar()

def render_search_interface():
    """Legacy compatibility function."""
    render_enhanced_search_interface()

def perform_search(search_type: str, query: str, max_results: int = 20):
    """Legacy compatibility function."""
    perform_enhanced_search(search_type, query, max_results)

def get_search_placeholder(search_type: str) -> str:
    """Get appropriate placeholder text for search input."""
    num_topics = 0
    if hasattr(chat, 'global_lda_model') and chat.global_lda_model and hasattr(chat.global_lda_model, 'num_topics'):
        num_topics = chat.global_lda_model.num_topics

    placeholders = {
        "Keyword": "Enter keywords to search for...",
        "Sender": "Enter sender name...",
        "Recipient": "Enter recipient name...",
        "Year": "Enter year (e.g., 1863)...",
        "Place": "Enter place name...",
        "Smart Search (spaCy)": "Ask a natural language question...",
        "Semantic Search": "Describe the content you're looking for...",
        "Question Answering": "Ask a direct question about the documents...",
        "Hybrid Search": "Enter query for combined keyword + semantic search...",
        "Topic ID": f"Enter topic number (0 to {num_topics-1 if num_topics > 0 else 0})",
        "Topic Terms": "Enter topic keywords..."
    }
    return placeholders.get(search_type, "Enter search query...")

def get_search_help(search_type: str) -> str:
    """Get help text for different search types."""
    help_texts = {
        "Keyword": "Simple text search across title, description, and content",
        "Sender": "Find letters from specific senders",
        "Recipient": "Find letters to specific recipients", 
        "Year": "Find letters from a specific year",
        "Place": "Find letters mentioning specific places",
        "Smart Search (spaCy)": "AI-powered search using natural language understanding",
        "Semantic Search": "Find semantically similar content using AI embeddings",
        "Question Answering": "Get direct answers extracted from the documents",
        "Hybrid Search": "Combines keyword matching with semantic similarity",
        "Topic ID": "Search by topic number from topic modeling",
        "Topic Terms": "Search letters containing specific topic keywords"
    }
    return help_texts.get(search_type, "")

def display_qa_results_streamlit(answers):
    """Legacy compatibility function."""
    display_qa_results_enhanced(answers)

def display_hybrid_results_streamlit(results):
    """Legacy compatibility function."""
    display_hybrid_results_enhanced(results)

if __name__ == "__main__":
    main()