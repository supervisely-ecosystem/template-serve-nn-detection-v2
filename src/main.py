import functools
import os
import supervisely as sly

import src.helpers as helpers
import src.sly_globals as g


def send_error_data(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        value = None
        try:
            value = func(*args, **kwargs)
        except Exception as e:
            request_id = kwargs["context"]["request_id"]
            g.sly_app.send_response(request_id, data={"error": repr(e)})
        return value

    return wrapper


@g.app.post("/get_output_classes_and_tags")
@sly.timeit
def get_output_classes_and_tags(api, task_id, context, state, app_logger):
    request_id = context["request_id"]
    g.sly_app.send_response(request_id, data=g.meta.to_json())


@g.app.post("/get_custom_inference_settings")
@sly.timeit
def get_custom_inference_settings(api, task_id, context, state, app_logger):
    request_id = context["request_id"]
    g.sly_app.send_response(request_id, data={"settings": g.default_settings})


@g.app.post("/get_session_info")
@sly.timeit
@send_error_data
def get_session_info(query, context):
    info = {
        "app": "Custom Detection Serve",
        # "model_name": "",
        "classes_count": len(g.meta.obj_classes),
        "tags_count": len(g.meta.tag_metas),
        # "supports_sliding_window": False
    }

    request_id = context["request_id"]
    g.sly_app.send_response(request_id, data=info)


@g.app.post("/inference_image_url")
@sly.timeit
def inference_image_url(api, task_id, context, state, app_logger):
    app_logger.debug("Input data", extra={"state": state})
    image_url = state["image_url"]
    ext = sly.fs.get_file_ext(image_url)
    if ext == "":
        ext = ".jpg"
    local_image_path = os.path.join(g.sly_app.data_dir, sly.rand_str(15) + ext)

    sly.fs.download(image_url, local_image_path)
    ann_json = helpers.inference(
        state=state, image_path=local_image_path, app_logger=app_logger
    )

    request_id = context["request_id"]
    g.sly_app.send_response(request_id, data=ann_json)


@g.app.post("/inference_image_id")
@sly.timeit
def inference_image_id(api, task_id, context, state, app_logger):
    app_logger.debug("Input data", extra={"state": state})
    image_id = state["image_id"]
    image_info = api.image.get_info_by_id(image_id)
    image_path = os.path.join(g.sly_app.data_dir, sly.rand_str(10) + image_info.name)
    ann_json = helpers.inference(
        state=state, image_path=image_path, app_logger=app_logger
    )
    sly.fs.silent_remove(image_path)
    request_id = context["request_id"]
    g.sly_app.send_response(request_id, data=ann_json)


@g.app.post("/inference_batch_ids")
@sly.timeit
def inference_batch_ids(api, task_id, context, state, app_logger):
    app_logger.debug("Input data", extra={"state": state})
    ids = state["batch_ids"]
    infos = api.image.get_info_by_id_batch(ids)

    paths = [
        os.path.join(g.sly_app.data_dir, sly.rand_str(10) + info.name) for info in infos
    ]
    api.image.download_paths(infos[0].dataset_id, ids, paths)

    results = []
    for image_path in paths:
        ann_json = helpers.inference(
            state=state, image_path=image_path, app_logger=app_logger
        )
        results.append(ann_json)

    request_id = context["request_id"]
    g.sly_app.send_response(request_id, data=results)

# from fastapi import Request
# @g.app.get("/")
# def read_index(request: Request):
#     return g.templates.TemplateResponse('index.html', {'request': request})
