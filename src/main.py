import supervisely as sly
import src.helpers as helpers
import src.sly_globals as g


def inference( state: dict, image_path: str) -> sly.Annotation:
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
    # Read image from local file
    image = sly.image.read(path=image_path)

    # Function generates random predictions in this template to demonstrate the functionality, 
    # but you will need to replace implementation of the generate_predictions() 
    # function to your own, using the inference of your own model
    pred_bboxes, pred_scores, pred_classes = helpers.generate_predictions(image=image)
    ##########################################
    # INSERT YOUR CODE INSTEAD OF LINE ABOVE #
    ##########################################

    # The file custom_settings.yaml contains settings for postprocessing and 
    # designed to store parameters
    settings = state.get("settings", {})
    helpers.check_settings(settings=settings)
    conf_thres = settings.get(
        "confidence_threshold", g.default_settings["confidence_threshold"]
    )

    
    # This function performs post-processing of model predictions (for example, NMS)
    result_bboxes, result_scores, result_classes = helpers.postprocess_predictions(
        pred_bboxes=pred_bboxes,
        pred_scores=pred_scores,
        pred_classes=pred_classes,
        conf_thres=conf_thres,
    )

    
    # This function converts model predictions into supervisely annotation format
    annotation = helpers.convert_preds_to_sly_annotation(
        pred_bboxes=result_bboxes,
        pred_scores=result_scores,
        pred_classes=result_classes,
        img_size=image.shape[:2],
    )

    return annotation


def deploy_model() -> None:
    """
    Add comment
    """
    #####################
    # CHANGE CODE BELOW #
    #####################
    pass
    # g.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # g.model_classes = get_classes_from_model_config()
    # g.model_id_classes_map = dict(enumerate(g.model_classes))
    # g.model_name = "Your Custom Model Name"

    # g.model = init_model(weigths=g.local_weights_path, device=g.device)
    # sly.logger.info("🟩 Model has been successfully deployed")



def main():
    sly.logger.info("Supervisely settings", extra={
        "context.teamId": g.team_id,
        "context.workspaceId": g.workspace_id
    })
    input_image_path, output_image_path = helpers.get_image_from_args()
    g.inference_fn = inference
    if input_image_path is not None: 
        state = {}
        result_annotation = g.inference_fn(state, input_image_path)
        helpers.draw_demo_result(input_image_path, result_annotation, output_image_path)
    else:
        helpers.download_model()
        deploy_model()
        helpers.construct_model_meta()
        g.app.run()


if __name__ == "__main__":
    sly.main_wrapper("main", main)
