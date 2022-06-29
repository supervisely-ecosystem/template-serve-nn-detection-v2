import functools
import json
import os

import supervisely as sly
import requests


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def main():
    # response = requests.post('http://127.0.0.1:8000/inference_image_url/', json={'state': {'image_url': 'https://img.icons8.com/color/1000/000000/deciduous-tree.png'}}, timeout=5)  # for local testing
    # print(response.text)

    task_id = 18119

    api = sly.Api.from_env()

    post_serve_cb = functools.partial(api.task.send_request, task_id=task_id, timeout=10)

    functions_to_test = [
        functools.partial(post_serve_cb, method='/get_output_classes_and_tags/', data={}),
        functools.partial(post_serve_cb, method='/get_custom_inference_settings/', data={}),
        functools.partial(post_serve_cb, method='/get_session_info/', data={}),
        functools.partial(post_serve_cb, method='/inference_image_id/', data={"image_id": 2797154}),
        functools.partial(post_serve_cb, method='/inference_batch_ids/', data={"batch_ids": [2797154]}),
        functools.partial(post_serve_cb, method='/inference_image_url/', data={"image_url": "https://img.icons8.com/color/1000/000000/deciduous-tree.png"})
    ]

    for curr_func in functions_to_test:
        response = curr_func()
        print(f"{bcolors.OKBLUE}{curr_func.keywords['method']} returns:{bcolors.ENDC}")
        print(response, end='\n\n')


if __name__ == "__main__":
    main()
