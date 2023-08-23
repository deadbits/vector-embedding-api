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

from typing import Dict, Union, Optional
from collections import OrderedDict
from flask import Flask, request, jsonify, abort
from sentence_transformers import SentenceTransformer


app = Flask(__name__)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Config:
    def __init__(self, config_file: str):
        self.config_file = config_file
        self.config = configparser.ConfigParser()
        if not os.path.exists(self.config_file):
            logging.error(f'Config file not found: {self.config_file}')
            sys.exit(1)

        logging.info(f'Loading config file: {self.config_file}')
        self.config.read(config_file)

    def get_val(self, section: str, key: str) -> Optional[str]:
        answer = None

        try:
            answer = self.config.get(section, key)
        except Exception as err:
            logging.error(f'Config file missing section: {section} - {err}')

        return answer
    
    def get_bool(self, section: str, key: str, default: bool = False) -> bool:
        try:
            return self.config.getboolean(section, key)
        except Exception as err:
            logging.error(f'Failed to parse boolean - returning default "False": {section} - {err}')
            return default


class EmbeddingCache:
    def __init__(self, max_size: int = 500):
        logger.info(f'Created in-memory cache; max size={max_size}')
        self.cache = OrderedDict()
        self.max_size = max_size

    def get_cache_key(self, text: str, model_type: str) -> str:
        return hashlib.sha256((text + model_type).encode()).hexdigest()

    def get(self, text: str, model_type: str):
        return self.cache.get(self.get_cache_key(text, model_type))

    def set(self, text: str, model_type: str, embedding):
        key = self.get_cache_key(text, model_type)
        self.cache[key] = embedding
        if len(self.cache) > self.max_size:
            self.cache.popitem(last=False)


class EmbeddingGenerator:
    def __init__(self, sbert_model: Optional[str] = None, openai_key: Optional[str] = None):
        self.sbert_model = sbert_model
        self.openai_key = openai_key
        if self.sbert_model is not None:
            try:
                self.model = SentenceTransformer(self.sbert_model)
                logger.info(f'enabled model: {self.sbert_model}')
            except Exception as err:
                logger.error(f'Failed to load SentenceTransformer model "{self.sbert_model}": {err}')
                sys.exit(1)

        if openai_key is not None:
            openai.api_key = self.openai_key
            logger.info('enabled model: text-embedding-ada-002')

    def generate(self, text: str, model_type: str) -> Dict[str, Union[str, float, list]]:
        start_time = time.time()
        result = {'status': 'success'}

        if model_type == 'openai':
            try:
                response = openai.Embedding.create(input=text, model='text-embedding-ada-002')
                result['embedding'] = response['data'][0]['embedding']
                result['model'] = 'text-embedding-ada-002'
            except Exception as err:
                logger.error(f'Failed to get OpenAI embeddings: {err}')
                result['status'] = 'error'
                result['message'] = str(err)

        else:
            try:
                embedding = self.model.encode(text).tolist()
                result['embedding'] = embedding
                result['model'] = self.sbert_model
            except Exception as err:
                logger.error(f'Failed to get sentence-transformers embeddings: {err}')
                result['status'] = 'error'
                result['message'] = str(err)
        
        result['elapsed'] = (time.time() - start_time) * 1000
        return result


@app.route('/health', methods=['GET'])
def health_check():
    sbert_on = embedding_generator.sbert_model if embedding_generator.sbert_model else 'disabled'
    openai_on = True if embedding_generator.openai_key else 'disabled'

    health_status = {
        "models": {
            "openai": openai_on,
            'sentence-transformers': sbert_on
        },
        "cache": {
            "enabled": embedding_cache is not None,
            "size": len(embedding_cache.cache) if embedding_cache else None,
            "max_size": embedding_cache.max_size if embedding_cache else None
        }
    }

    return jsonify(health_status)


@app.route('/submit', methods=['POST'])
def submit_text():
    data = request.json

    text_data = data.get('text')
    model_type = data.get('model', 'local').lower()

    if text_data is None:
        abort(400, 'Missing text data to embed')

    if model_type not in ['local', 'openai']:
        abort(400, 'model field must be one of: local, openai')

    if isinstance(text_data, str):
        text_data = [text_data]
    
    if not all(isinstance(text, str) for text in text_data):
        abort(400, 'all data must be text strings')

    results = []
    for text in text_data:
        result = None

        if embedding_cache:
            result = embedding_cache.get(text, model_type)
            if result:
                logger.info('found embedding in cache!')
                result = {'embedding': result, 'cache': True, "status": 'success'}

        if result is None:
            result = embedding_generator.generate(text, model_type)

            if embedding_cache and result['status'] == 'success':
                embedding_cache.set(text, model_type, result['embedding'])
                logger.info('added to cache')

        results.append(result)

    return jsonify(results)


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
    if use_cache:
        max_cache_size = int(conf.get_val('main', 'cache_max'))

    if openai_key is None:
        logger.warn('No OpenAI API key set in configuration file: server.conf')

    if sbert_model is None:
        logger.warn('No transformer model set in configuration file: server.conf')

    if openai_key is None and sbert_model is None:
        logger.error('No sbert model set *and* no openAI key set; exiting')
        sys.exit(1)

    embedding_cache = EmbeddingCache(max_cache_size) if use_cache else None
    embedding_generator = EmbeddingGenerator(sbert_model, openai_key)

    app.run(debug=True)
