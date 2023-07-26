#!/usr/bin/env python3
# github.com/deadbits/vector-embedding-api
import os
import sys
import logging
import configparser

import openai

from flask import Flask, request, jsonify, abort
from sentence_transformers import SentenceTransformer


app = Flask(__name__)


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
        except:
            logging.error(f'Config file missing section: {section}')

        return answer


def get_openai_embeddings(text: str):
    try:
        response = openai.Embedding.create(input=text, model='text-embedding-ada-002')
        return response['data'][0]['embedding']
    except Exception as err:
        logger.error(f'Failed to get OpenAI embeddings: {err}')
        abort(500, 'Failed to get OpenAI embeddings')


def get_transformers_embeddings(text: str):
    try:
        return model.encode(text).tolist()
    except Exception as err:
        logger.error(f'Failed to get sentence-transformers embeddings: {err}')
        abort(500, 'Failed to get sentence-transformers embeddings')


@app.route('/submit', methods=['POST'])
def submit_text():
    data = request.json
    ada = data.get('ada', False)
    
    if not 'text' in data:
        abort(400, 'Text data is required')

    if ada:
        embedding_data = get_openai_embeddings(data['text'])
    else:
        embedding_data = get_transformers_embeddings(data['text'])

    return jsonify({'embedding': embedding_data, 'status': 'success'})


if __name__ == '__main__':
    conf = Config('server.conf')
    openai.api_key = conf.get('main', 'openai_api_key')
    sent_model = conf.get('main', 'sent_transformers_model')

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    try:
        model = SentenceTransformer(sent_model)
    except Exception as err:
        logger.error(f'Failed to load SentenceTransformer model "{sent_model}": {err}')
        sys.exit(1)

    app.run(debug=True)

