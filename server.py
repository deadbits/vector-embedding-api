#!/usr/bin/env python3
# github.com/deadbits/vector-embedding-api
# server.py
import os
import sys
import time
import argparse
import hashlib
import logging
import configparser

import openai

from flask import Flask, request, jsonify, abort
from sentence_transformers import SentenceTransformer


app = Flask(__name__)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Config:
    def __init__(self, config_file):
        self.config_file = config_file
        self.config = configparser.ConfigParser()
        if not os.path.exists(self.config_file):
            logging.error(f'Config file not found: {self.config_file}')
            sys.exit(1)

        logging.info(f'Loading config file: {self.config_file}')
        self.config.read(config_file)

    def get_val(self, section, key):
        answer = None

        try:
            answer = self.config.get(section, key)
        except Exception as err:
            logging.error(f'Config file missing section: {section} - {err}')

        return answer
    
    def get_bool(self, section, key, default=False):
        try:
            return self.config.getboolean(section, key)
        except Exception as err:
            logging.error(f'Failed to parse boolean - returning default "False": {section} - {err}')
            return default


class EmbeddingCache:
    def __init__(self):
        logger.info('Created in-memory cache')
        self.cache = {}

    def get_cache_key(self, text, model_type):
        return hashlib.sha256((text + model_type).encode()).hexdigest()

    def get(self, text, model_type):
        return self.cache.get(self.get_cache_key(text, model_type))

    def set(self, text, model_type, embedding):
        self.cache[self.get_cache_key(text, model_type)] = embedding


class EmbeddingGenerator:
    def __init__(self, sbert_model=None, openai_key=None):
        self.sbert_model = sbert_model
        if self.sbert_model is not None:
            try:
                self.model = SentenceTransformer(self.sbert_model)
                logger.info(f'enabled model: {self.sbert_model}')
            except Exception as err:
                logger.error(f'Failed to load SentenceTransformer model "{self.sbert_model}": {err}')
                sys.exit(1)

        if openai_key is not None:
            openai.api_key = openai_key
            logger.info('enabled model: text-embedding-ada-002')

    def get_openai_embeddings(self, text):
        start_time = time.time()

        try:
            response = openai.Embedding.create(input=text, model='text-embedding-ada-002')
            elapsed_time = (time.time() - start_time) * 1000
            data = {
                "embedding": response['data'][0]['embedding'],
                "status": "success",
                "elapsed": elapsed_time,
                "model": "text-embedding-ada-002"
            }
            return data
        except Exception as err:
            logger.error(f'Failed to get OpenAI embeddings: {err}')
            return {"status": "error", "message": str(err), "model": "text-embedding-ada-002"}

    def get_transformers_embeddings(self, text):
        start_time = time.time()

        try:
            embedding = self.model.encode(text).tolist()
            elapsed_time = (time.time() - start_time) * 1000
            data = {
                "embedding": embedding,
                "status": "success",
                "elapsed": elapsed_time,
                "model": self.sbert_model
            }
            return data
        except Exception as err:
            logger.error(f'Failed to get sentence-transformers embeddings: {err}')
            return {"status": "error", "message": str(err), "model": self.sbert_model}

    def generate(self, text, model_type):
        if model_type == 'openai':
            return self.get_openai_embeddings(text)
        else:
            return self.get_transformers_embeddings(text)


@app.route('/submit', methods=['POST'])
def submit_text():
    data = request.json

    text_data = data.get('text')
    model_type = data.get('model', 'local').lower()

    if text_data is None:
        abort(400, 'Missing text data to embed')

    if model_type not in ['local', 'openai']:
        abort(400, 'model field must be one of: local, openai')

    if embedding_cache:
        result = embedding_cache.get(text_data, model_type)
        if result:
            logger.info('found embedding in cache!')
            result = {'embedding': result, 'cache': True, "status": 'success'}
    else:
        result = None

    if result is None:
        result = embedding_generator.generate(text_data, model_type)

        if embedding_cache and result['status'] == 'success':
            embedding_cache.set(text_data, model_type, result['embedding'])
            logger.info('added to cache')

    return jsonify(result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '-c', '--config',
        help='config file',
        type=str,
        required=True
    )

    args = parser.parse_args()

    conf = Config(args.config)
    openai_key = conf.get_val('main', 'openai_api_key')
    sbert_model = conf.get_val('main', 'sent_transformers_model')
    use_cache = conf.get_bool('main', 'use_cache', default=False)

    if openai_key is None:
        logger.warn('No OpenAI API key set in configuration file: server.conf')

    if sbert_model is None:
        logger.warn('No transformer model set in configuration file: server.conf')

    if openai_key is None and sbert_model is None:
        logger.error('No sbert model set *and* no openAI key set; exiting')
        sys.exit(1)

    embedding_cache = EmbeddingCache() if use_cache else None
    embedding_generator = EmbeddingGenerator(sbert_model, openai_key)

    app.run(debug=True)
