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

## 🚀 Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/AI-Chatbot.git
cd AI-Chatbot

# Create and activate virtual environment
conda create -n chatbot python=3.11
conda activate chatbot

# Install dependencies
pip install -r requirements_working.txt

# Download spaCy model
python -m spacy download en_core_web_sm

# Run the application
bash run_app.sh
```

## 💡 Usage

1. **Prepare your data**: Place XML/TEI files in the `xmlfiles/` directory
2. **Start the app**: Run `bash run_app.sh`
3. **Load documents**: Use the sidebar to index your documents
4. **Choose search type**: Select from various search options
5. **Enter query**: Type your search term or question
6. **Explore results**: Click to expand document details

## 🔍 Search Types Explained

| Search Type | Description | Example Query |
|-------------|-------------|--------------|
| **Keyword** | Exact text matching | "battle" or "Richmond" |
| **Smart** | AI interprets meaning | "Who wrote about shortages?" |
| **Semantic** | Meaning-based similarity | "military conflict" |
| **Question Answering** | Extracts direct answers | "What supplies were needed?" |
| **Topic** | Theme-based exploration | Select topic or enter keyword |
| **Hybrid** | Combined keyword & semantic | "Confederate soldiers supplies" |

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