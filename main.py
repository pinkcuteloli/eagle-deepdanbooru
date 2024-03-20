import deepdanbooru as dd
import tensorflow as tf
import huggingface_hub
import numpy as np
import pillow_avif
import PIL.Image
import requests
from urllib.parse import unquote

BASE_API_URL = 'http://localhost:41595'

def get_items_without_tags(max_iterations=None):
    offset = 0
    limit = 200
    items_without_tags = []

    while True:
        print(f'Fetching items with offset {offset}...')  # デバッグポイント1

        response = requests.get(f'{BASE_API_URL}/api/item/list?limit={limit}&offset={offset}')
        data = response.json()

        if response.status_code != 200:
            print('Error fetching data:', response.status_code, response.text)  # デバッグポイント2
            break

        if 'data' not in data:
            print('Error: "data" key not found in response.')  # デバッグポイント3
            break

        items = data['data']

        if not items:
            print('No more items to fetch.')  # デバッグポイント4
            break

        if max_iterations is not None and offset >= max_iterations:
            print(f'Max iterations reached: {max_iterations}')  # デバッグポイント5
            break

        for item in items:
            if 'deepdanbooru' not in [tag.lower() for tag in item['tags']] and item['ext'] in ['jpg', 'jpeg', 'png', 'bmp', 'gif', 'webp', "avif"]:
                items_without_tags.append(item)

        offset += 1
        print(f'Iteration {offset} completed. Total items without tags: {len(items_without_tags)}')  # デバッグポイント6

    return items_without_tags

def load_model() -> tf.keras.Model:
    path = huggingface_hub.hf_hub_download(
        'public-data/DeepDanbooru',
        'model-resnet_custom_v3.h5'
    )

    return tf.keras.models.load_model(path)


def load_labels() -> list[str]:

    path = huggingface_hub.hf_hub_download(
        'public-data/DeepDanbooru',
        'tags.txt'
    )

    with open(path) as f:
        labels = [line.strip() for line in f.readlines()]

    return labels

def get_thumbnail_path(item_id):
    response = requests.get(f'{BASE_API_URL}/api/item/thumbnail?id={item_id}')
    data = response.json()
    if response.status_code == 200 and 'data' in data:
        return data['data']
    else:
        print('Error fetching thumbnail:', response.text)
        return None

def update_item_tags(item_id, new_tags):
    response = requests.get(f'{BASE_API_URL}/api/item/info?id={item_id}')
    data = response.json()
    if response.status_code == 200 and 'data' in data and data['data']:
        all_tags = list(set(data["data"]['tags'] + new_tags))
        
        data = {
            'id': item_id,
            'tags': all_tags
        }
        response = requests.post(f'{BASE_API_URL}/api/item/update', json=data)
        if response.status_code == 200:
            print('Tags updated successfully for item', item_id)
        else:
            print('Error updating tags:', response.text)
    else:
        print('Error fetching item:', response.text)

model = load_model()
labels = load_labels()

def predict(
    image: PIL.Image.Image,
    score_threshold: float
) -> dict[str, float]:
    
    if image.mode != 'RGB':
        image = image.convert('RGB')

    _, height, width, _ = model.input_shape
    image = np.asarray(image)
    image = tf.image.resize(
        image,
        size=(height, width),
        method=tf.image.ResizeMethod.AREA,
        preserve_aspect_ratio=True
    )
    image = image.numpy()
    image = dd.image.transform_and_pad_image(image, width, height)
    image = image / 255.
    probs = model.predict(image[None, ...])[0]
    probs = probs.astype(float)
    res = dict()

    for prob, label in zip(probs.tolist(), labels):
        if prob < score_threshold:
            continue
        res[label] = prob

    return res

def main():
    items_without_tags = get_items_without_tags()

    for item in items_without_tags:
        thumbnail_path = get_thumbnail_path(item['id'])

        if thumbnail_path:
            thumbnail_path = unquote(thumbnail_path)
            try:
                with PIL.Image.open(thumbnail_path) as image:
                    predicted_tags = predict(image, 0.5)
                    tags = list(predicted_tags.keys())
                    tags.append('deepdanbooru')
                    update_item_tags(item['id'], tags)
                    print('Predicted tags for item', item['id'], tags)
            except FileNotFoundError:
                print(f'File not found: {thumbnail_path}')
            except PIL.UnidentifiedImageError:
                print(f'Cannot identify image file: {thumbnail_path}')

if __name__ == '__main__':
    main()