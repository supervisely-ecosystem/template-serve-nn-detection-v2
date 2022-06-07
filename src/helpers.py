from typing import Dict, List, Tuple, Union
import functools
import os
import sys
import supervisely as sly

import sly_globals as g


def serve_detection(get_classes_and_tags_fn,
                    get_session_info_fn,
                    inference_fn,
                    deploy_model_fn):
    g.get_classes_and_tags_fn = get_classes_and_tags_fn
    g.get_session_info_fn = get_session_info_fn
    g.inference_fn = inference_fn
    g.deploy_model_fn = deploy_model_fn

    sly.logger.info("Supervisely settings", extra={
        "context.teamId": g.team_id,
        "context.workspaceId": g.workspace_id
    })
    input_image_path, output_image_path = get_image_from_args()
    if input_image_path is not None: 
        result_annotation = inference(input_image_path)
        draw_demo_result(input_image_path, result_annotation, output_image_path)
    else:
        download_model()
        deploy()
        g.app.run()

def deploy():
    g.model = g.deploy_model_fn(g.local_weights_path)
    sly.logger.info("Model has been deployed successfully")

def inference(image_path: str) -> sly.Annotation:
    """This is a demo function to show how to inference your custom model
    on a selected image in supervsely.

    Parameters
    ----------
    state : dict
        Dict that stores application fields
    image_path : str
        Local path to image

    Returns
    -------
    sly.Annotation
        Supervisely annotation
    """

    predictions = g.inference_fn(image_path)
    
    image = sly.image.read(image_path)
    # This function converts model predictions into supervisely annotation format
    annotation = convert_preds_to_sly_annotation(
        predictions,
        img_size=image.shape[:2],
    )

    return annotation


def get_image_from_args() -> Tuple[str, str]:
    print(sys.argv)
    if len(sys.argv) == 1:
        return None, None
    if len(sys.argv) != 3: 
        raise AttributeError("Usage: main.py input_image_path output_image_path")
    if not os.path.exists(sys.argv[1]):
        raise FileExistsError(f"File {sys.argv[1]} not found.")
    return sys.argv[1], sys.argv[2]


def draw_demo_result(input_image_path, annotation, output_image_path) -> None:
    image = sly.image.read(path=input_image_path)
    for label in annotation.labels:
        label.draw_contour(image, thickness=5)
    
    sly.image.write(output_image_path, image)
    sly.logger.info(f"Labeled image saved to {output_image_path} successfully")


def send_error_data(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        value = None
        try:
            value = func(*args, **kwargs)
        except Exception as e:
            request_id = kwargs["context"]["request_id"]
            g.app.send_response(request_id, data={"error": repr(e)})
        return value

    return wrapper


@g.app.callback("get_output_classes_and_tags")
@sly.timeit
def get_output_classes_and_tags(api: sly.Api, task_id, context, state, app_logger):
    model_meta = g.get_classes_and_tags_fn()
    request_id = context["request_id"]
    g.app.send_response(request_id, data=model_meta.to_json())


@g.app.callback("get_custom_inference_settings")
@sly.timeit
def get_custom_inference_settings(api: sly.Api, task_id, context, state, app_logger):
    request_id = context["request_id"]
    g.app.send_response(request_id, data={})


@g.app.callback("get_session_info")
@sly.timeit
@send_error_data
def get_session_info(api: sly.Api, task_id, context, state, app_logger):

    info = g.get_session_info_fn()

    request_id = context["request_id"]
    g.app.send_response(request_id, data=info)


@g.app.callback("inference_image_url")
@sly.timeit
def inference_image_url(api: sly.Api, task_id, context, state, app_logger):
    app_logger.debug("Input data", extra={"state": state})
    image_url = state["image_url"]

    ext = sly.fs.get_file_ext(image_url)
    assert ext in ["png", "jpg", "jpeg"]
    local_image_path = os.path.join(g.app.data_dir, f"{sly.rand_str(15)}.{ext}")
    sly.fs.download(image_url, local_image_path)

    ann = g.inference_fn(image_path=local_image_path)

    request_id = context["request_id"]
    g.app.send_response(request_id, data=ann.to_json())


@g.app.callback("inference_image_id")
@sly.timeit
def inference_image_id(api: sly.Api, task_id, context, state, app_logger):
    app_logger.debug("Input data", extra={"state": state})
    image_id = state["image_id"]
    image_info = api.image.get_info_by_id(image_id)
    image_path = os.path.join(g.app.data_dir, sly.rand_str(10) + image_info.name)
    ann = g.inference_fn(image_path=image_path)
    sly.fs.silent_remove(image_path)
    
    request_id = context["request_id"]
    g.app.send_response(request_id, data=ann.to_json())


@g.app.callback("inference_batch_ids")
@sly.timeit
def inference_batch_ids(api: sly.Api, task_id, context, state, app_logger):
    app_logger.debug("Input data", extra={"state": state})
    ids = state["batch_ids"]
    infos = api.image.get_info_by_id_batch(ids)
    paths = [os.path.join(g.app.data_dir, sly.rand_str(10) + info.name) for info in infos]
    api.image.download_paths(infos[0].dataset_id, ids, paths)
    results = []
    for image_path in paths:
        ann = g.inference_fn(image_path=image_path)
        results.append(ann.to_json())

    request_id = context["request_id"]
    g.app.send_response(request_id, data=results)


def convert_preds_to_sly_annotation(
        predictions: dict,
        img_size: Tuple[int, int],
) -> sly.Annotation:
    """Convert model predictions to supervisely format annotation.

    Parameters
    ----------
    predictions : dict
        Prediction bounding boxes
    img_size : Tuple[int, int]
        height and width of the image

    Returns
    -------
    sly.Annotation
        Supervisely annotation in JSON format
    """
    labels = []
    for prediction in predictions:
        box = prediction["bbox"]
        sly_rect = sly.Rectangle(box[0], box[1], box[2], box[3])
        sly_obj_class = sly.ObjClass(
            prediction["class"], sly.Rectangle
        )
        if "confidence" in prediction.keys():
            sly_tag_meta = sly.TagMeta("confidence", sly.TagValueType.ANY_NUMBER)
            sly_tag = sly.Tag(sly_tag_meta, prediction["confidence"])
            sly_tag_collection = sly.TagCollection([sly_tag])
        else:
            sly_tag_collection = sly.TagCollection([])
        sly_label = sly.Label(sly_rect, sly_obj_class, tags=sly_tag_collection)
        labels.append(sly_label)

    ann = sly.Annotation(img_size=img_size, labels=labels)
    return ann


def download_model() -> None:
    if not g.remote_weights_path.endswith(".pth"):
        sly.logger.info("Model is not found. Template mode with example labels will be used.")
        return

    info = g.api.file.get_info_by_path(g.team_id, g.remote_weights_path)
    if info is None:
        raise FileNotFoundError(f"Weights file not found: {g.remote_weights_path}")

    sly.logger.info("Downloading model weights...")
    g.local_weights_path = os.path.join(g.app.data_dir, sly.fs.get_file_name_with_ext(g.remote_weights_path))
    g.api.file.download(
        g.team_id,
        g.remote_weights_path,
        g.local_weights_path,
        cache=g.app.cache
    )

    sly.logger.info("Model has been successfully downloaded")

