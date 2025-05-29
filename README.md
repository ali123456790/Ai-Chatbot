# AI-Powered Historical Document Analyzer

An advanced application for exploring Civil War era historical letters using AI and NLP technologies.

## Features

- **Keyword Search**: Traditional text matching across documents
- **Smart Search**: AI-powered natural language query understanding with spaCy
- **Semantic Search**: Find documents by meaning using sentence transformers
- **Question Answering**: Get direct answers from historical documents
- **Hybrid Search**: Combines keyword and semantic search for best results

## Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/Historical-Document-Analyzer.git
cd Historical-Document-Analyzer

# Create and activate virtual environment
conda create -n chatbot python=3.11
conda activate chatbot

# Install dependencies
pip install -r requirements_working.txt

# Run the application
bash run_app.sh
```

## Technical Stack

- **Frontend**: Streamlit
- **AI Models**: 
  - Sentence Transformers
  - spaCy
  - Hugging Face Transformers (Question Answering)
- **Data Processing**: TEI XML parsing with metadata extraction

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details. 