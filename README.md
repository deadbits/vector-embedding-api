# vector-embedding-api
Flask API server for generating text embeddings using [OpenAI's embedding model](https://platform.openai.com/docs/guides/embeddings) or the [SentenceTransformers](https://www.sbert.net/) library. SentenceTransformers supports over 500 models via [HuggingFace Hub](https://huggingface.co/sentence-transformers).

## Features 🎯
* POST endpoint access to text embedding models
  * sentence_transformers
  * OpenAI text-embedding-ada-002 
* Easy setup with configuration file
* Simple integration with other applications

### Installation 💻
To run this server locally, follow the steps below:

**Clone the repository:** 📦
```bash
git clone https://github.com/deadbits/vector-embedding-api.git
cd text-embeddings-server
```

**Set up a virtual environment (optional but recommended):** 🐍
```bash
python3 -m venv venv
source venv/bin/activate
```

**Install the required dependencies:** 🛠️
```bash
pip install -r requirements.txt
```

### Usage
Before running the server, make sure you have obtained an API key from OpenAI to use their model. You also need to set the SentenceTransformers model you want to use in the [server.conf](/server.conf) file.

**Modify the server.conf configuration file:** ⚙️
```ini
[main]
openai_api_key = YOUR_OPENAI_API_KEY
sent_transformers_model = sentence-transformers/all-MiniLM-L6-v2
```

**Start the server:** 🚀
```
python server.py
```

The server should now be running on http://127.0.0.1:5000/.


### API Endpoints 🌐
The server provides the following endpoint:

#### POST /submit
Submit text to be converted to embeddings.
The default is to use the SentenceTransformers model.
Setting the `ada` field to True will use the OpenAI model.

**Example Request:**
```bash
curl -X POST -H "Content-Type: application/json" -d '{'text': 'Put your text here', "ada": true}' http://127.0.0.1:5000/submit
```

**Example Response:**
```json
{
    "embedding": [0.123, 0.456, ..., 0.789],
    "status": "success"
}
```

### Error Handling ⚠️
The server responds with appropriate error messages if there are any issues with the request or generating embeddings.
