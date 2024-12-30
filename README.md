## Modelhaven Search Agent

Semantic search using advanced AI tools.
It combines various components to perform efficient web searches for AI models, process retrieved data into embeddings, and store the embeddings in a vector database for retrieval.

## Features

# Search Functionality
Retrieve data from web pages using Google search queries.
Text Processing
Clean and extract text from web pages.
Split the text into manageable chunks for processing.
Embeddings
Generate embeddings using the jina-embeddings-v2-base-en model from Ollama.
Vector Storage
Store and retrieve embeddings efficiently using ChromaDB.
Async Operations
Use Python's asyncio for concurrent fetching and processing.
Stack

1. Ollama Framework
A local AI playground for running models without cloud dependencies.
Used to handle the jina-embeddings-v2-base-en embedding model.
2. Llama 3.1
An instruction-tuned LLM for generating insightful and accurate responses.
Backbone for intelligent query handling and dialogue-based interactions.
3. Jina Embeddings
Embedding model powered by Ollama for text-to-vector conversion.
Enables semantic search for better query matching.
4. ChromaDB
Open-source vector database for storing and retrieving embeddings.
Installation

# Prerequisites
Python 3.10 or later
Ollama installed locally
LLMs 
Basic understanding of Python virtual environments
Steps
# Clone the Repository
``` git clone https://github.com/dom-inic/modelhavenagent.git ```
cd modelhaven-search-agent
# Install Dependencies
Use pip/uv to install the required dependencies:

``` pip install -r requirements.txt```
Run Ollama using the command
```ollama serve```
Pull LLama  
you can change llama version to your desired version.
```ollama pull llama3.1 ```
Pull Jina Embeddings Model Ensure that the jina-embeddings-v2-base-en model is pulled using Ollama:
``` ollama pull jina-embeddings-v2-base-en```
Set Environment Variables
Create a .env file in the root directory to store your OPENAI API keys or other environment variables:

Then load it in your Python code using dotenv.

Query the Agent
To perform a search, call the web_search function in the search.py module:

Contribution

Contributions are welcome! Please follow these steps:

Fork the repository.
Create a feature branch.
Commit your changes.
Open a pull request.
License

This project is licensed under the MIT License.
