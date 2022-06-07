import random
from typing import Dict, List, Tuple, Union
import functools
import os
import sys
import numpy as np
import supervisely as sly

import src.sly_globals as g


def get_image_from_args() -> Tuple[str, str]:
    if len(sys.argv) == 1:
        return None
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
    request_id = context["request_id"]
    g.app.send_response(request_id, data=g.model_meta.to_json())


@g.app.callback("get_custom_inference_settings")
@sly.timeit
def get_custom_inference_settings(api: sly.Api, task_id, context, state, app_logger):
    request_id = context["request_id"]
    g.app.send_response(request_id, data={"settings": g.default_settings_str})


@g.app.callback("get_session_info")
@sly.timeit
@send_error_data
def get_session_info(api: sly.Api, task_id, context, state, app_logger):
    info = {
        "app": "Serve Custom Detection Model Template",
        "model_name": g.model_name,
        "device": g.device,
        "classes_count": len(g.model_meta.obj_classes),
        "tags_count": len(g.model_meta.tag_metas),
        "sliding_window_support": True
    }

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

    ann = g.inference_fn(state=state, image_path=local_image_path)

    request_id = context["request_id"]
    g.app.send_response(request_id, data=ann.to_json())


@g.app.callback("inference_image_id")
@sly.timeit
def inference_image_id(api: sly.Api, task_id, context, state, app_logger):
    app_logger.debug("Input data", extra={"state": state})
    image_id = state["image_id"]
    image_info = api.image.get_info_by_id(image_id)
    image_path = os.path.join(g.app.data_dir, sly.rand_str(10) + image_info.name)
    ann = g.inference_fn(state=state, image_path=image_path)
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
        ann = g.inference_fn(
            state=state, image_path=image_path
        )
        results.append(ann.to_json())

    request_id = context["request_id"]
    g.app.send_response(request_id, data=results)


def check_settings(
        settings: Dict[str, Union[str, int]]) -> None:
    """
    Check custom settings.

    In our example it returns only confidence threshold

    Parameters
    ----------
    settings : Dict[str, Union[str, int]]
        Settings from .yaml file

    Returns
    -------
    None
    """
    for key, value in g.default_settings.items():
        if key not in settings:
            sly.logger.warn(
                "Field {!r} not found in inference settings. Use default value {!r}".format(
                    key, value
                )
            )


def generate_predictions(
        image: np.ndarray
) -> Tuple[List[Tuple[int, int, int, int]], List[float], List[int]]:
    """
    Gets prediction bounding boxes, scores and classes.
    We use randomly generated labels but you should get predictions from your model here.

    Parameters
    ----------
    image : np.ndarray
        input image in numpy array format

    Returns
    -------
    Tuple[List[Tuple[int, int, int, int]], List[float], List[int]]
        Tuple with 3 arrays with predictions: bounding boxes, scores and classes
    """
    img_height, img_width = image.shape[:2]

    pred_bboxes = []
    pred_scores = []
    pred_classes = []

    min_boxes_on_image = 1
    max_boxes_on_image = 5
    for _ in range(random.randint(min_boxes_on_image, max_boxes_on_image)):
        x = np.random.randint(1, img_width - 1, size=(2,))
        if np.all(x == x[0]):
            x[0] += 1

        y = np.random.randint(1, img_height - 1, size=(2,))
        if np.all(y == y[0]):
            y[0] += 1

        left, top = x.min(), y.min()
        right, bottom = x.max(), y.max()
        pred_bboxes.append((top, left, bottom, right))

        pred_score = random.random()
        pred_scores.append(pred_score)

        pred_class_idx = random.choice(list(g.model_id_classes_map.keys()))
        pred_classes.append(pred_class_idx)
    return pred_bboxes, pred_scores, pred_classes


def convert_preds_to_sly_annotation(
        pred_bboxes: List[Tuple[int, int, int, int]],
        pred_scores: List[float],
        pred_classes: List[int],
        img_size: Tuple[int, int],
) -> Dict[str, Union[str, int]]:
    """Convert model predictions to supervisely format annotation.

    Parameters
    ----------
    pred_bboxes : List[Tuple[int, int, int, int]]
        Prediction bounding boxes
    pred_scores : List[float]
        Prediction scores
    pred_classes : List[int]
        Prediction classes ids
    img_size : Tuple[int, int]
        height and width of the image

    Returns
    -------
    Dict[str, Union[str, int]]
        Supervisely annotation in JSON format
    """
    assert (
            len(pred_bboxes) == len(pred_scores) == len(pred_classes)
    ), "Length of prediction arrays is not equal."

    labels = []
    for pred_box, pred_score, pred_class_idx in zip(
            pred_bboxes, pred_scores, pred_classes
    ):
        top, left, bottom, right = pred_box
        sly_rect = sly.Rectangle(top, left, bottom, right)
        sly_obj_class = sly.ObjClass(
            g.model_id_classes_map[pred_class_idx], sly.Rectangle
        )
        sly_tag_meta = sly.TagMeta(g.confidence_tag_name, sly.TagValueType.ANY_NUMBER)
        sly_tag = sly.Tag(sly_tag_meta, pred_score)
        sly_tag_collection = sly.TagCollection([sly_tag])
        sly_label = sly.Label(sly_rect, sly_obj_class, tags=sly_tag_collection)
        labels.append(sly_label)

    ann = sly.Annotation(img_size=img_size, labels=labels)
    return ann


def postprocess_predictions(
        pred_bboxes: List[Tuple[int, int, int, int]],
        pred_scores: List[float],
        pred_classes: List[int],
        conf_thres: float = 0.5,
) -> Tuple[List[Tuple[int, int, int, int]], List[float], List[int]]:
    """Process predictions with given confidence threshold.

    Parameters
    ----------
    pred_bboxes : List[Tuple[int, int, int, int]]
        Prediction bounding boxes
    pred_scores : List[float]
        Prediction scores
    pred_classes : List[int]
        Prediction classes ids
    conf_thres : int, optional
        Confidence threshold

    Returns
    -------
    Tuple[List[Tuple[int, int, int, int]], List[float], List[int]]
        Tuple with processed predictions
    """
    result_bboxes, result_scores, result_classes = [], [], []
    for pred_box, pred_score, pred_class_idx in zip(
            pred_bboxes, pred_scores, pred_classes
    ):
        if pred_score < conf_thres:
            continue
        result_bboxes.append(pred_box)
        result_scores.append(pred_score)
        result_classes.append(pred_class_idx)
    return result_bboxes, result_scores, result_classes


def construct_model_meta() -> None:
    """Generate project meta from model classes names."""

    colors = []
    for i in range(len(g.model_classes)):
        colors.append(sly.color.generate_rgb(exist_colors=colors))

    obj_classes = [
        sly.ObjClass(name, sly.Rectangle, color)
        for name, color in zip(g.model_classes, colors)
    ]
    tags = [sly.TagMeta(g.confidence_tag_name, sly.TagValueType.ANY_NUMBER)]
    meta = sly.ProjectMeta(
        obj_classes=sly.ObjClassCollection(obj_classes),
        tag_metas=sly.TagMetaCollection(tags),
    )
    g.model_meta = meta


def download_model() -> None:
    # TODO:
    """Add description"""
    if g.remote_weights_path is None:
        # Template case without model
        sly.logger.info("Model is not found. Template mode with random labels will be used.")
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

