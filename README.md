# AI-Powered Historical Document Analyzer

An advanced application for exploring Civil War era historical letters using AI and NLP technologies.

## 📚 Project Overview

This application analyzes historical documents (specifically Civil War era letters in XML/TEI format) using various AI and NLP techniques. It extracts metadata, processes content, and provides multiple search capabilities ranging from traditional keyword search to advanced AI-powered semantic search and question answering.

## 🌟 Key Features

- **Keyword Search**: Traditional text matching across documents
- **Smart Search**: AI-powered natural language query understanding with spaCy
- **Semantic Search**: Find documents by meaning using sentence transformers
- **Question Answering**: Get direct answers from historical documents
- **Hybrid Search**: Combines keyword and semantic search for best results
- **Topic Modeling**: Automatic discovery of themes across documents
- **Metadata Filtering**: Search by sender, recipient, year, or location
- **Interactive UI**: User-friendly Streamlit interface with expandable results

## 🛠️ Technical Architecture

- **Frontend**: Streamlit web interface with responsive design
- **Document Processing**: XML/TEI parser with metadata extraction
- **AI Models**: 
  - **spaCy**: For NLP tasks and smart query processing
  - **Sentence Transformers**: For document embeddings and semantic search
  - **Hugging Face Transformers**: For extractive question answering
  - **Gensim**: For topic modeling (LDA, HDP, Dynamic Topic Models)
- **Search Engine**: Custom-built with hybrid capabilities
- **Data Persistence**: Pickled index for fast startup

## 🚀 Installation & Setup

### Prerequisites
- **Python**: 3.8 or higher (recommended: Python 3.11)
- **Conda**: Miniconda or Anaconda for environment management
- **Git**: For cloning the repository
- **System Requirements**: 
  - RAM: Minimum 8GB (16GB recommended for large document collections)
  - Storage: At least 2GB free space for models and data
  - OS: Windows, macOS, or Linux

### Step-by-Step Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/YOUR_USERNAME/AI-Chatbot.git
   cd AI-Chatbot
   ```

2. **Create and Activate Virtual Environment**
   ```bash
   # Create new conda environment with Python 3.11
   conda create -n chatbot python=3.11
   
   # Activate the environment
   conda activate chatbot
   ```

3. **Install Dependencies**
   ```bash
   # Install all required Python packages
   pip install -r requirements_working.txt
   
   # This will install:
   # - streamlit (web interface)
   # - spacy (NLP processing)
   # - sentence-transformers (semantic search)
   # - transformers (AI models)
   # - gensim (topic modeling)
   # - scikit-learn (machine learning)
   # - pandas, numpy (data processing)
   # - And other dependencies
   ```

4. **Download Required AI Models**
   ```bash
   # Download spaCy English language model
   python -m spacy download en_core_web_sm
   
   # Note: Other AI models (sentence transformers, etc.) 
   # will be downloaded automatically on first use
   ```

5. **Prepare Your Data** (Optional)
   ```bash
   # Place your XML/TEI historical documents in the xmlfiles directory
   # The app comes with sample documents, but you can add your own
   ls xmlfiles/  # Check existing documents
   ```

## 🏃‍♂️ Running the Application

### Quick Start
```bash
# Make sure you're in the project directory
cd AI-Chatbot

# Make the run script executable (Unix/macOS/Linux)
chmod +x run_app.sh

# Run the application
bash run_app.sh
```

### Manual Start (Alternative Method)
If the run script doesn't work, you can start manually:
```bash
# Activate conda environment
conda activate chatbot

# Start Streamlit application
streamlit run app_gui.py
```

### What Happens When You Run
1. **Environment Activation**: The script activates the conda environment
2. **Streamlit Launch**: Starts the web server on port 8501
3. **Browser Opening**: Your default browser should open automatically
4. **Application Access**: Navigate to `http://localhost:8501` if it doesn't open automatically

### First-Time Setup in the App
1. **Access the Application**: Open `http://localhost:8501` in your browser
2. **Load Documents**: 
   - Use the sidebar to select "Load Documents"
   - Click "Process XML Files" to index your documents
   - Wait for indexing to complete (this may take a few minutes for large collections)
3. **Verify Setup**: You should see document count and available search options

## 💡 How to Use the Application

### Basic Workflow
1. **Start the App**: Run `bash run_app.sh`
2. **Load Documents**: Use sidebar to process XML files (first-time only)
3. **Choose Search Method**: Select from the dropdown menu
4. **Enter Query**: Type your search term or question
5. **Explore Results**: Click on results to see full document content
6. **Apply Filters**: Use metadata filters (sender, year, location) to refine results

### Search Options Explained

| Search Type | Best For | Example Query | Expected Results |
|-------------|----------|---------------|------------------|
| **Keyword Search** | Finding exact terms | "Richmond" | Documents containing "Richmond" |
| **Smart Search** | Natural language questions | "Who wrote about food shortages?" | AI interprets and finds relevant docs |
| **Semantic Search** | Concept-based searching | "military battles" | Documents about combat, warfare, conflicts |
| **Question Answering** | Getting direct answers | "What supplies were needed?" | Extracted answers from documents |
| **Topic Search** | Thematic exploration | Browse by automatically detected topics | Documents grouped by themes |
| **Hybrid Search** | Best of both worlds | "Confederate soldier supplies" | Combines keyword + semantic matching |

### Advanced Features
- **Metadata Filtering**: Filter by sender, recipient, year, or location
- **Result Expansion**: Click any result to see full document text
- **Topic Visualization**: Explore automatically discovered themes
- **Export Options**: Copy results for further analysis

## 🔧 Troubleshooting

### Common Issues and Solutions

**1. Application Won't Start**
```bash
# Check if conda is installed
conda --version

# Check if environment exists
conda env list

# Recreate environment if needed
conda env remove -n chatbot
conda create -n chatbot python=3.11
conda activate chatbot
pip install -r requirements_working.txt
```

**2. Port 8501 Already in Use**
```bash
# Kill existing Streamlit processes
pkill -f streamlit

# Or use a different port
streamlit run app_gui.py --server.port 8502
```

**3. Memory Issues with Large Document Collections**
- Reduce batch size in processing
- Process documents in smaller chunks
- Increase system RAM or use cloud instance

**4. Model Download Failures**
```bash
# Manually download spaCy model
python -c "import spacy; spacy.cli.download('en_core_web_sm')"

# Check internet connection for other models
pip install --upgrade transformers sentence-transformers
```

**5. Permission Errors (Unix/macOS)**
```bash
# Make script executable
chmod +x run_app.sh

# Or run with bash explicitly
bash run_app.sh
```

### Getting Help
- **Check Logs**: Look for error messages in the terminal
- **Restart Fresh**: Close browser, stop app (Ctrl+C), restart
- **Environment Issues**: Deactivate and reactivate conda environment
- **Clear Cache**: Delete `__pycache__` folders and `.pkl` files, restart app

## 🛑 Stopping the Application
- **Keyboard Shortcut**: Press `Ctrl+C` in the terminal
- **Close Browser**: Simply close the browser tab (app keeps running in background)
- **Full Stop**: Use `Ctrl+C` to completely stop the server

## 📊 Performance Notes
- **First Run**: Initial setup takes 5-10 minutes (model downloads)
- **Document Indexing**: Processing time varies by collection size
- **Search Speed**: Semantic search is slower than keyword search
- **Memory Usage**: Approximately 2-4GB RAM during operation

## 🚧 Improvement Roadmap

Based on code analysis, here are detailed improvement opportunities:

### 1. Performance Enhancements
- **Implement batch processing for large document sets** - Current indexing processes files sequentially, which becomes slow with large collections
- **Add caching layer for frequent queries** - Implement Redis or similar to cache common search results
- **Optimize embedding generation** - Use more efficient batching for sentence transformers
- **Implement progressive loading** - Show partial results while processing continues

### 2. Model Improvements
- **Implement fine-tuning for domain adaptation** - Train models on Civil War era language patterns
- **Add support for more specialized models**:
  - Historical NER model for better entity recognition
  - Domain-specific QA model
- **Implement model quantization** - Reduce model size and improve inference speed
- **Add model version management** - Support loading different model versions

### 3. Search Capabilities
- **Add fuzzy search capability** - For handling spelling variations in historical documents
- **Implement proximity search** - Find terms appearing near each other
- **Add time-series analysis** - Track concept evolution through time periods
- **Support complex boolean queries** - AND, OR, NOT operations with parentheses
- **Add cross-document reasoning** - Connect information across multiple documents

### 4. User Interface
- **Implement visualization dashboard** - Add network graphs, timelines, and geospatial views
- **Add user authentication** - Support multiple researchers with personalized views
- **Implement annotation system** - Allow adding notes to documents
- **Add mobile optimization** - Improve UI for tablet/phone access
- **Implement collaborative features** - Shared workspaces and results

### 5. Data Management
- **Add support for more document formats** - PDF, DOCX, plain text
- **Implement data versioning** - Track changes to document collections
- **Add automated data quality checks** - Validate and report on XML integrity
- **Support incremental indexing** - Only process new or changed files
- **Implement secure backup solution** - Automated backup of index and user data

### 6. Integration
- **Add API endpoints** - Enable programmatic access to search functionality
- **Implement plugin system** - Allow custom extensions
- **Add export to research tools** - Zotero, Omeka integration
- **Develop citation generation** - Format results for academic citation
- **Add social sharing capabilities** - Share interesting findings

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details. 