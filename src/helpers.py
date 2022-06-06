import random
from typing import Dict, List, Tuple, Union

import numpy as np
import supervisely as sly

import sly_globals as g


def check_settings(
        settings: Dict[str, Union[str, int]], app_logger: sly.logger
) -> None:
    """
    Check custom settings.

    In our example it returns only confidence threshold

    Parameters
    ----------
    settings : Dict[str, Union[str, int]]
        Settings from .yaml file
    app_logger : sly.logger
        Supervisely logger

    Returns
    -------
    None
    """
    for key, value in g.default_settings.items():
        if key not in settings:
            app_logger.warn(
                "Field {!r} not found in inference settings. Use default value {!r}".format(
                    key, value
                )
            )


def generate_predictions(
        image: np.ndarray
) -> Tuple[List[Tuple[int, int, int, int]], List[float], List[int]]:
    """
    Generates prediction bounding boxes, scores and classes ids.

    Parameters
    ----------
    image : np.ndarray
        input image in numpy array format

    Returns
    -------
    Tuple[List[Tuple[int, int, int, int]], List[float], List[int]]
        Tuple with 3 arrays with predictions bounding boxes, scores and classes ids
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
    return ann.to_json()


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


def inference(
        state: Dict[str, Union[str, int]], image_path: str, app_logger: sly.logger
) -> Dict[str, Union[str, int]]:
    """Process inference from given image id.

    Parameters
    ----------
    state : Dict[str, Union[str, int]]
        Dict that stores application fields
    image_path : int
        Local path to image
    app_logger : sly.logger
        Supervisely logger

    Returns
    -------
    Dict[str, Union[str, int]]
        Supervisely annotation in JSON format
    """
    image = sly.image.read(path=image_path)

    """
    This is a demo function to show how to perform inference on a selected image in supervsely.

    Function generate random predictions in this template to demonstrate the functionality, but you will need to replace
    implementation of the generate_predictions() function on your own, using the inference of your own model
    """
    pred_bboxes, pred_scores, pred_classes = generate_predictions(image=image)

    """
    The file custom_settings.yaml contains settings for postprocessing and designed to store parameters.
    """
    check_settings(settings=state.get("settings", {}), app_logger=app_logger)
    conf_thres = state.get("settings").get(
        "confidence_threshold", g.default_settings["confidence_threshold"]
    )

    """
    The postprocess_predictions() function performs post-processing of model predictions (for example, NMS).
    """
    result_bboxes, result_scores, result_classes = postprocess_predictions(
        pred_bboxes=pred_bboxes,
        pred_scores=pred_scores,
        pred_classes=pred_classes,
        conf_thres=conf_thres,
    )

    """
    The convert_preds_to_sly_annotation() function converts model predictions into supervisely annotation format.
    """
    ann_json = convert_preds_to_sly_annotation(
        pred_bboxes=result_bboxes,
        pred_scores=result_scores,
        pred_classes=result_classes,
        img_size=image.shape[:2],
    )

    return ann_json


def construct_model_meta(model_classes: List[str]) -> sly.ProjectMeta:
    """Generate project meta from model classes list.

    Parameters
    ----------
    model_classes : List[str]
        List of model classes

    Returns
    -------
    sly.ProjectMeta
        Supervisely project meta
    """
    colors = []
    for i in range(len(model_classes)):
        colors.append(sly.color.generate_rgb(exist_colors=colors))

    obj_classes = [
        sly.ObjClass(name, sly.Rectangle, color)
        for name, color in zip(model_classes, colors)
    ]
    tags = [sly.TagMeta(g.confidence_tag_name, sly.TagValueType.ANY_NUMBER)]
    meta = sly.ProjectMeta(
        obj_classes=sly.ObjClassCollection(obj_classes),
        tag_metas=sly.TagMetaCollection(tags),
    )
    return meta
