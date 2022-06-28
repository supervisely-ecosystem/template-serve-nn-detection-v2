import os
from typing import Dict, List
import supervisely as sly
import helpers

from helpers import app

my_model = None

def get_classes_and_tags() -> sly.ProjectMeta:
    classes = sly.ObjClassCollection([
        # Example
        sly.ObjClass("person", sly.Rectangle),
        sly.ObjClass("car", sly.Rectangle),
        sly.ObjClass("bus", sly.Rectangle)

        # Put any needed classes here ....
    ])

    tags = sly.TagMetaCollection([
        # Example: confidence tag for bounding boxes with number value
        sly.TagMeta("confidence", sly.TagValueType.ANY_NUMBER)
    ])

    return sly.ProjectMeta(obj_classes=classes, tag_metas=tags)
    
    
def get_session_info() -> Dict:
    return {
        # Recommended info values
        "app": "Serve Custom Detection Model Template",
        "model_name": "Put your model name",
        "device": "cpu", 
        "classes_count": 3,
        "tags_count": 1,
        "sliding_window_support": False
        # Put any key-value that you want to ....
    }


def inference(image_path: str) -> List[Dict]:
    image = sly.image.read(path=image_path) # shape: [H, W, 3], RGB
    
    #########################
    # INSERT YOUR CODE HERE #
    #########################
    # predictions = my_model(image)

    # example (remove it when you'll use your own predictions)
    predictions = [
        {
            "bbox": [50, 100, 77, 145], # [top, left, bottom, right]
            "class": "person", # class name like in get_classes_and_tags() function
            "confidence": 0.88 # optional 
        }
    ]
    return predictions
    

def deploy_model(model_weights_path: str) -> None:
    global my_model
    #########################
    # INSERT YOUR CODE HERE #
    #########################
    # example:
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # my_model = init_model(weigths=model_weights_path, device=device)
    
    my_model = None
    
    
@app.on_event("startup")
async def startup_event():
    if "TASK_ID" not in os.environ:
        # Used for local debug
        model_weights_path = "./my_folder/my_weights.pth"
        input_image_path = "./my_folder/my_image.png"
        result_image_path = "./my_folder/result_image.png"
        deploy_model(model_weights_path)
        predictions = inference(input_image_path)
        helpers.draw_demo_result(predictions, input_image_path, result_image_path)
    else:
        # Used for production
        helpers.serve_detection(
            get_session_info, 
            get_classes_and_tags, 
            inference, 
            deploy_model
        )

        sly.logger.info("actual 1")



