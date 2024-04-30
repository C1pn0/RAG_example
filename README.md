# Query Response System
## Description
This Python application provides a web-based interface for users to input queries and receive responses generated by a language model. The application uses retrieval augmented generation to provide more accurate and relevant responses to user queries. The application uses the OpenAI API for language model inference and Pinecone as vector database for retrieval. 

## Usage
To start the web interface, run the following command from the terminal:
```bash
python app.py
```
This will launch a Gradio web interface accessible via a local URL, typically http://127.0.0.1:7860/. Navigate to this URL in a web browser to start interacting with the chatbot.

## Installation
1. Clone this repository: 
```bash
git clone https://github.com/C1pn0/RAG_example.git
cd RAG_example
```
2. Install the required packages from the `requirements.txt`:
```bash
pip install -r requirements.txt
```

### Configuration
Ensure that your API keys for external services like OpenAI and Pinecone are set in your environment variables (`OPENAI_API_KEY` and `PINECONE_API_KEY`) or specified in a configuration file.
Optionally e-mail should be added via `DEFAULT_EMAIL` environmental variable for large-scale parsing of NCBI databases to avoid server-side interruption.


