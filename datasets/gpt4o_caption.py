# -*- coding: utf-8 -*-
import json
from openai import OpenAI
from PIL import Image
import imghdr
import base64
import io
import httpx
import logging
import time
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Replace with your real OpenAI API keys
gpt_keys = [
    {"idx": 0, "key": "openai-key-1"},
    {"idx": 1, "key": "openai-key-2"},
]

MAX_API_RETRY = len(gpt_keys)
key_id = 0
proxy_url = 'http://127.0.0.1:10212'  # replace with your actual proxy if needed

def list_to_str(tmp):
    return '\n'.join(str(item) for item in tmp)

def one_ask(prompt, image_paths, image_size=(512, 512), detail='low'):
    global key_id
    for _ in range(MAX_API_RETRY):
        try:
            api_key = gpt_keys[key_id]['key']
            proxies = {"http://": proxy_url, "https://": proxy_url}
            http_client = httpx.Client(proxies=proxies)
            client = OpenAI(api_key=api_key, http_client=http_client)

            content = [{"type": "text", "text": prompt}]
            for image in image_paths:
                image_type = imghdr.what(image)
                with Image.open(image) as img:
                    if img.size[0] > image_size[0] or img.size[1] > image_size[1]:
                        img.thumbnail(image_size, Image.LANCZOS)
                    byte_stream = io.BytesIO()
                    img.save(byte_stream, format=image_type)
                    encoded_image = base64.b64encode(byte_stream.getvalue()).decode('utf-8')
                image_data_url = f'data:image/{image_type};base64,{encoded_image}'
                content.append({"type": "image_url", "image_url": {"url": image_data_url, "detail": detail}})

            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": content}],
                max_tokens=4096,
            )

            caption = response.choices[0].message.content
            logger.info(f"[Prompt] {prompt}")
            logger.info(f"[Caption] {caption}")

            key_id = (key_id + 1) % MAX_API_RETRY
            return caption

        except Exception as e:
            key_id = (key_id + 1) % MAX_API_RETRY
            logger.error('[Error in one_ask]: ' + repr(e))
            time.sleep(1.5)

    logger.error(f"[Error] Failed after {MAX_API_RETRY} retries.")
    return "error"

# Prompt for remote sensing image captioning
caption_prompt = (
    "Please describe the remote sensing image in detail, generating a description of at least 100 words. If you can, identify what objects are present in the image."
)

def get_captions_from_images(img_folder, source_path, dest_path, begin_ix):
    with open(source_path, 'r', encoding='utf-8') as f:
        source_data = json.load(f)

    for ix, data in enumerate(source_data):
        if ix < begin_ix:
            continue

        logger.info(f"Processing {ix + 1}/{len(source_data)}")

        try:
            image_file = os.path.join(img_folder, data['image_path'])
            prompt = caption_prompt
            caption = one_ask(prompt, [image_file])

            new_data = data.copy()
            new_data['generated_caption'] = caption
            new_data['used_prompt'] = prompt

            with open(dest_path, 'a', encoding='utf-8') as fout:
                fout.write(json.dumps(new_data, ensure_ascii=False) + '\n')

            time.sleep(1.5)

        except Exception as e:
            logger.error(f"[Exception] {repr(e)}")

    logger.info("Caption generation completed.")

if __name__ == '__main__':
    img_folder = 'poster'
    source_path = 'poster.jsonl'
    dest_path = 'poster_caption.jsonl'
    begin_ix = 0
    get_captions_from_images(img_folder, source_path, dest_path, begin_ix)
