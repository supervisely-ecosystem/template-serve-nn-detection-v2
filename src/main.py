import os
import functools

import supervisely as sly

import sly_globals as g
import helpers


def send_error_data(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        value = None
        try:
            value = func(*args, **kwargs)
        except Exception as e:
            request_id = kwargs["context"]["request_id"]
            send_response(request_id, data={"error": repr(e)})
        return value

    return wrapper


@g.sly_app.post("/get_output_classes_and_tags/")
@sly.timeit
def get_output_classes_and_tags(context):
    request_id = context["request_id"]
    sly_app.send_response(request_id, data=g.meta.to_json())


@g.sly_app.post("/get_custom_inference_settings/")
@sly.timeit
def get_custom_inference_settings(context):
    request_id = context["request_id"]
    send_response(request_id, data={"settings": default_settings})


@g.sly_app.post("/get_session_info/")
@sly.timeit
@send_error_data
def get_session_info(context):
    info = {
        "app": "Custom Detection Serve",
        "classes_count": len(g.meta.obj_classes),
        "tags_count": len(g.meta.tag_metas),
    }
    # add model name ^
    # supports_sliding_window: False ^

    request_id = context["request_id"]
    send_response(request_id, data=info)


@g.sly_app.post("/inference_image_url/")
@sly.timeit
def inference_image_url(api: sly.Api, task_id, context, state, app_logger):
    app_logger.debug("Input data", extra={"state": state})
    image_url = state["image_url"]
    ext = sly.fs.get_file_ext(image_url)
    if ext == "":
        ext = ".jpg"
    local_image_path = os.path.join(g.sly_app.data_dir, sly.rand_str(15) + ext)

    sly.fs.download(image_url, local_image_path)
    ann_json = helpers.inference(api=api, state=state, image_path=local_image_path, app_logger=app_logger)

    request_id = context["request_id"]
    # api.task.send_request()
    # g.sly_app.send_response(request_id, data=ann_json)


@g.sly_app.post("/inference_image_id/")
@sly.timeit
def inference_image_id(api: sly.Api, task_id, context, state, app_logger):
    app_logger.debug("Input data", extra={"state": state})
    image_id = state["image_id"]
    image_info = api.image.get_info_by_id(image_id)
    image_path = os.path.join(g.sly_app.data_dir, sly.rand_str(10) + image_info.name)
    ann_json = helpers.inference(api=api, state=state, image_path=image_path, app_logger=app_logger)
    sly.fs.silent_remove(image_path)
    request_id = context["request_id"]
    # api.task.send_request()
    # g.sly_app.send_response(request_id, data=ann_json)


@g.sly_app.callback("/inference_batch_ids/")
@sly.timeit
def inference_batch_ids(api: sly.Api, task_id, context, state, app_logger):
    app_logger.debug("Input data", extra={"state": state})
    ids = state["batch_ids"]
    infos = api.image.get_info_by_id_batch(ids)

    paths = [
        os.path.join(g.sly_app.data_dir, sly.rand_str(10) + info.name)
        for info in infos
    ]
    api.image.download_paths(infos[0].dataset_id, ids, paths)

    results = []
    for image_path in paths:
        ann_json = helpers.inference(
            api=api, state=state, image_path=image_path, app_logger=app_logger
        )
        results.append(ann_json)

    request_id = context["request_id"]
    # api.task.send_request()
    # g.sly_app.send_response(request_id, data=results)
