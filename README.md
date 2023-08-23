# vector-embedding-api
`vector-embedding-api`provides a Flask API server and client to generate text embeddings using either [OpenAI's embedding model](https://platform.openai.com/docs/guides/embeddings) or the [SentenceTransformers](https://www.sbert.net/) library. The API server now supports in-memory LRU caching for faster retrievals, batch processing for handling multiple texts at once, and a health status endpoint for monitoring the server status.

SentenceTransformers supports over 500 models via [HuggingFace Hub](https://huggingface.co/sentence-transformers).

## Features 🎯
* POST endpoint to create text embeddings
  * sentence_transformers
  * OpenAI text-embedding-ada-002
* In-memory LRU cache for quick retrieval of embeddings
* Batch processing to handle multiple texts in a single request
* Easy setup with configuration file
* Health status endpoint
* Python client utility for submitting text or files

### Installation 💻
To run this server locally, follow the steps below:

**Clone the repository:** 📦
```bash
git clone https://github.com/deadbits/vector-embedding-api.git
cd vector-embedding-api
```

**Set up a virtual environment (optional but recommended):** 🐍
```bash
virtualenv -p /usr/bin/python3.10 venv
source venv/bin/activate
```

**Install the required dependencies:** 🛠️
```bash
pip install -r requirements.txt
```

### Usage

**Modify the [server.conf](/server.conf) configuration file:** ⚙️
```ini
[main]
openai_api_key = YOUR_OPENAI_API_KEY
sent_transformers_model = sentence-transformers/all-MiniLM-L6-v2
use_cache = true/false
```

**Start the server:** 🚀
```
python server.py
```

The server should now be running on http://127.0.0.1:5000/.

### API Endpoints 🌐
##### Client Usage
A small [Python client](/client.py) is provided to assist with submitting text strings or files. 

**Usage**
`python3 client.py -t "Your text here" -m local`

`python3 client.py -f /path/to/yourfile.txt -m openai`

#### POST /submit
Submits an individual text string or a list of text strings for embedding generation.

**Request Parameters**

* **text:** The text string or list of text strings to generate the embedding for. (Required)
* **model:** Type of model to be used, either local for SentenceTransformer models or openai for OpenAI's model. Default is local.

**Response**

* **embedding:** The generated embedding array.
* **status:** Status of the request, either success or error.
* **elapsed:** The elapsed time taken for generating the embedding (in milliseconds).
* **model:** The model used to generate the embedding.
* **cache:** Boolean indicating if the result was retrieved from cache. (Optional)
* **message:** Error message if the status is error. (Optional)

#### GET /health
Checks the server's health status.

**Response**

* **cache.enabled:** Boolean indicating status of the cache
* **cache.max_size:** Maximum cache size
* **cache.size:** Current cache size
* **models.openai:** Boolean indicating if OpenAI embeddings are enabled. (Optional)
* **models.sentence-transformers:** Name of sentence-transformers model in use.

```json
{
  "cache": {
    "enabled": true,
    "max_size": 500,
    "size": 0
  },
  "models": {
    "openai": true,
    "sentence-transformers": "sentence-transformers/all-MiniLM-L6-v2"
  }
}
```

#### Example Usage
Send a POST request to the /submit endpoint with JSON payload:

```json
{
    "text": "Your text here",
    "model": "local"
}

// multi text submission
{
    "text": ["Text1 goes here", "Text2 goes here"], 
    "model": "openai"
}
```

You'll receive a response containing the embedding and additional information:

```json
[
  {
    "embedding": [...],
    "status": "success",
    "elapsed": 123,
    "model": "sentence-transformers/all-MiniLM-L6-v2"
  }
]

[
  {
    "embedding": [...],
    "status": "success",
    "elapsed": 123,
    "model": "openai"
  }, 
  {
    "embedding": [...],
    "status": "success",
    "elapsed": 123,
    "model": "openai"
  }, 
]
```
