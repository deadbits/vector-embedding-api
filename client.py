#!/usr/bin/env python3
# github.com/deadbits/vector-embedding-api
# client.py
import os
import sys
import json
import argparse
import requests
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime


def timestamp_str():
    return datetime.isoformat(datetime.utcnow())


class Embedding(BaseModel):
    text: str = ''
    embedding: List[float] = []
    metadata: Optional[dict] = {}


def send_request(text_batch, model_type='local'):
    url = 'http://127.0.0.1:5000/submit'
    headers = {'Content-Type': 'application/json'}
    payload = {
        'text': text_batch,
        'model': model_type
    }

    try:
        response = requests.post(
            url,
            headers=headers,
            data=json.dumps(payload)
        )

        response.raise_for_status()
        return response.json()
    except requests.RequestException as err:
        print(f'[error] exception sending http request: {err}')
        return None


def process_batch(text_batch, model_type, embeddings_list, chunk_num, total_chunks):
    print(f'[status] {timestamp_str()} - Processing chunk {chunk_num} of {total_chunks}')
    result = send_request(text_batch, model_type)
    if result:
        if result[0]['status'] == 'error':
            print(f'[error] {timestamp_str()} - Received error: {result[0]["message"]}')
            return
        else:
            print(f'[status] {timestamp_str()} - Received embeddings: {len(result[0]["embeddings"])} ')
            for text, em in zip(text_batch, result[0]['embeddings']):
                metadata = {
                    'status': result[0]['status'],
                    'elapsed': result[0]['elapsed'],
                    'model': result[0]['model']
                }
                embedding = Embedding(text=text, embedding=em, metadata=metadata)
                embeddings_list.append(embedding.dict())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        '-t', '--text',
        help='text to embed'
    )

    group.add_argument(
        '-f', '--file',
        type=argparse.FileType('r'),
        help='text file to embed (one text per line)'
    )

    parser.add_argument(
        '-m', '--model',
        help='embedding model type',
        choices=['local', 'openai'],
        default='local'
    )

    parser.add_argument(
        '-o', '--output',
        help='output file',
        default='embeddings.json'
    )

    args = parser.parse_args()
    model_type = args.model
    output_file = args.output
    embeddings_list = []

    if os.path.exists(output_file):
        print(f'[error] {timestamp_str()} - Output file already exists')
        sys.exit(1)

    if args.file:
        if not os.path.exists(args.file.name):
            print(f'[error] {timestamp_str()} - File does not exist')
            sys.exit(1)

        print(f'[status] {timestamp_str()} - Processing file: {args.file.name}')

        text_batch = []
        chunk_size = 100
        total_lines = sum(1 for _ in args.file)
        args.file.seek(0)
        total_chunks = (total_lines + chunk_size - 1) // chunk_size

        print(f'[info] {timestamp_str()} - Total chunks: {total_chunks}')

        chunk_num = 1

        for line in args.file:
            text = line.strip()
            text_batch.append(text)
            if len(text_batch) == chunk_size:
                process_batch(text_batch, model_type, embeddings_list, chunk_num, total_chunks)
                text_batch = []
                chunk_num += 1

        if text_batch:
            process_batch(text_batch, model_type, embeddings_list, chunk_num, total_chunks)
    else:
        print(f'[status] {timestamp_str()} - Processing text input')
        text = args.text
        result = send_request([text], model_type)
        if result:
            for res in result:
                metadata = {'status': res['status'], 'elapsed': res['elapsed'], 'model': res['model']}
                embedding = Embedding(text=text, embedding=res['embedding'], metadata=metadata)
                embeddings_list.append(embedding.dict())

    try:
        print(f'[status] {timestamp_str()} - Saving embeddings to {output_file}')
        with open(output_file, 'w') as f:
            json.dump(embeddings_list, f)

        print(f'[status] {timestamp_str()} - Embeddings saved to embeddings.json')
    except Exception as err:
        print(f'[error] {timestamp_str()} - exception saving embeddings: {err}')
        sys.exit(1)

