#!/usr/bin/env python3
# github.com/deadbits/vector-embedding-api
import os
import sys
import argparse
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

    def get(self, section, key):
        answer = None

        try:
            answer = self.config.get(section, key)
        except Exception as err:
            logging.error(f'Config file missing section: {section} - {err}')

        return answer


def get_openai_embeddings(text: str) -> list:
    try:
        response = openai.Embedding.create(input=text, model='text-embedding-ada-002')
        return response['data'][0]['embedding']
    except Exception as err:
        logger.error(f'Failed to get OpenAI embeddings: {err}')
        abort(500, 'Failed to get OpenAI embeddings')


def get_transformers_embeddings(text: str) -> list:
    try:
        return model.encode(text).tolist()
    except Exception as err:
        logger.error(f'Failed to get sentence-transformers embeddings: {err}')
        abort(500, 'Failed to get sentence-transformers embeddings')


@app.route('/submit', methods=['POST'])
def submit_text():
    data = request.json
    
    text_data = data.get('text')
    model_type = data.get('model', 'local').lower()

    if text_data is None:
        abort(400, 'Missing text data to embed')
    
    if model_type not in ['local', 'openai']:
        abort(400, 'model field must be one of: local, openai')

    if model_type == 'openai':
        embedding_data = get_openai_embeddings(text_data)
    else:
        embedding_data = get_transformers_embeddings(text_data)

    return jsonify({'embedding': embedding_data, 'status': 'success'})


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
    api_key = conf.get('main', 'openai_api_key')
    sent_model = conf.get('main', 'sent_transformers_model')

    if api_key is None:
        logger.warn('No OpenAI API key set in configuration file: server.conf')
    else:
        logger.info('Set OpenAI API key via openai.api_key')
        openai.api_key = api_key

    if sent_model is None:
        logger.warn('No transformer model set in configuration file: server.conf')

    try:
        model = SentenceTransformer(sent_model)
    except Exception as err:
        logger.error(f'Failed to load SentenceTransformer model "{sent_model}": {err}')
        sys.exit(1)

    app.run(debug=True)

