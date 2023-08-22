# vector-embedding-api
`vector-embedding-api` provides a Flask API server and client to generate text embeddings using either [OpenAI's embedding model](https://platform.openai.com/docs/guides/embeddings) or the [SentenceTransformers](https://www.sbert.net/) library. The API server also offers an in-memory cache for embeddings and returns results from the cache when available.

SentenceTransformers supports over 500 models via [HuggingFace Hub](https://huggingface.co/sentence-transformers).

## Features 🎯
* POST endpoint to create text embeddings
  * sentence_transformers
  * OpenAI text-embedding-ada-002
* In-memory cache for embeddings
* Easy setup with configuration file
* Simple integration with other applications
* Python client utility for submitting text

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
Modify the [server.conf](/server.conf) file to specify a SentenceTransformers model, your OpenAI API key, or both.

**Modify the server.conf configuration file:** ⚙️
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
#### Client Usage
A small [Python client](/client.py) is provided to assist with submitting text strings or text files. 

**Usage**
`python3 client.py -t "Your text here" -m local`

`python3 client.py -f /path/to/yourfile.txt -m openai`

#### POST /submit
Submits a text string for embedding generation.

**Request Parameters**

* **text:** The text string to generate the embedding for. (Required)
* **model:** Type of model to be used, either local for SentenceTransformer models or openai for OpenAI's model. Default is local.

**Response**

* **embedding:** The generated embedding array.
* **status:** Status of the request, either success or error.
* **elapsed:** The elapsed time taken for generating the embedding (in milliseconds).
* **model:** The model used to generate the embedding.
* **cache:** Boolean indicating if the result was retrieved from cache. (Optional)
* **message:** Error message if the status is error. (Optional)

#### Example Usage
Send a POST request to the /submit endpoint with JSON payload:

```json
{
    "text": "Your text here",
    "model": "local"
}
```

You'll receive a response containing the embedding and additional information:

```json
{
    "embedding": [...],
    "status": "success",
    "elapsed": 293.52,
    "model": "sentence-transformers/all-MiniLM-L6-v2"
}
```
