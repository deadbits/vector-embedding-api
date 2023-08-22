import os
import sys
import json
import argparse
import requests


def send_request(text, model_type='local'):
    url = 'http://127.0.0.1:5000/submit'
    headers = {'Content-Type': 'application/json'}
    payload = {
        'text': text,
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
        print(f'[error] exception: {err}')
        return None


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
        help='text file to embed'
    )

    parser.add_argument(
        '-m', '--model',
        help='embedding model type',
        choices=['local', 'openai'],
        default='local'
    )

    args = parser.parse_args()

    if args.file:
        text = args.file.read()
    else:
        text = args.text

    model_type = args.model
    result = send_request(text, model_type)
    if result is not None:
        print(result)
