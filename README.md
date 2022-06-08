<div align="center" markdown>

<img src="https://github.com/supervisely-ecosystem/template-serve-nn-detection/releases/download/v0.0.1/poster.png" style="width: 100%;"/>

# Serve Custom Detection Model Template

<p align="center">
  <a href="#overview">Overview</a> •
  <a href="#preparation">Preparation</a> •
  <a href="#how-to-use">How To Use</a> •
  <a href="#how-to-add-model-as-app-enterprise-edition-only">How To Add Model As App</a> •
  <a href="#how-to-run">How To Run</a> •
  <a href="#related-apps">Related Apps</a> •
  <a href="#result">Result</a>
</p>


[![](https://img.shields.io/badge/supervisely-ecosystem-brightgreen)](https://ecosystem.supervise.ly/apps/supervisely-ecosystem/template-serve-nn-detection)
[![](https://img.shields.io/badge/slack-chat-green.svg?logo=slack)](https://supervise.ly/slack)
![GitHub release (latest SemVer)](https://img.shields.io/github/v/release/supervisely-ecosystem/template-serve-nn-detection)
[![views](https://app.supervise.ly/public/api/v3/ecosystem.counters?repo=supervisely-ecosystem/template-serve-nn-detection&counter=views&label=views)](https://supervise.ly)
[![used by teams](https://app.supervise.ly/public/api/v3/ecosystem.counters?repo=supervisely-ecosystem/template-serve-nn-detection&counter=downloads&label=used%20by%20teams)](https://supervise.ly)
[![runs](https://app.supervise.ly/public/api/v3/ecosystem.counters?repo=supervisely-ecosystem/template-serve-nn-detection&counter=runs&label=runs&123)](https://supervise.ly)

</div>

# Overview

Template Serve NN Detection app is designed for **developers** and can be used as a starting point for creating an application for serving your own detection NN models on Supervisely.

# Preparation

**Step 1.** Make a fork from this repository

**Step 2.** Clone repository to your computer

**Step 3.** Open repo directory and create python virtual environment by running the following command from the application root directory in terminal:

```bash
python -m venv venv
```

**Step 4.** Activate virtual environment:

```bash
source venv/bin/activate
```

**Step 5.** Install requirements.txt:

```bash
pip install -r requirements.txt
```

**Note:** we provide a docker image with cuda runtime and it's dependencies, but if you need to use something specific, add it to the `requirements.txt`, or use your own docker image, please contact supervisely technical support for details

# How To Develop

**Note:** recommended Python version >= 3.8

**Details:**
By default template app generates demo predictions to demonstrate the functionality. In order to implement your custom model, you will need to edit `main.py` file only. 

`main.py` - contains 4 functions with commentaries to help you implement your custom nn model:

* `get_classes_and_tags()` - constructs ProjectMeta object with specified model classes and tags.
* `get_session_info()` - generates model info dict with any parameters (see recommended parameters in file). You will see this parameters when you will connect to your model from other apps.
* `inference(image_path)` - this functions gets input image path and return model predictions on this image. See predictions format in file. Inference results will be automatically converted to [supervisely annotation format](https://docs.supervise.ly/data-organization/00_ann_format_navi).
* `deploy_model(model_weights_path)` - function initializes model to be ready to get input data for inference.

**Step 1.** Make sure you've edited `main.py`, without edits it will generate demo predictions

**Step 2.** Run `main.py` from terminal or by using your IDE interface:

```bash
python main.py
```

**Step 3.** When your model is ready, add additional modules and packages that are required to run your served model to `requirements.txt`

**Step 4.** Add your model as private app to Supervisely Ecosystem

# How To Add Model As App [Enterprise Edition only]

**Step 1.** Go to Ecosystem page and click on `private apps`

<img src="https://github.com/supervisely-ecosystem/template-serve-nn-detection/releases/download/v0.0.1/how-to-add-app-1.png" style="width:80%;"/>

**Step 2.** Click `+ Add private app` button

<img src="https://github.com/supervisely-ecosystem/template-serve-nn-detection/releases/download/v0.0.1/how-to-add-app-2.png" style="width:80%;"/>

**Step 3.** Copy and paste repository url and generated [github/gitlab personal token](https://docs.supervise.ly/enterprise-edition/advanced-tuning/private-apps) to modal window

<img src="https://github.com/supervisely-ecosystem/template-serve-nn-detection/releases/download/v0.0.1/how-to-add-app-3.png" style="width:50%;"/>

# How To Run:

**Step 1.** Add app with implemented custom nn model to your team from Ecosystem

<img data-key="sly-module-link" data-module-slug="supervisely-ecosystem/template-serve-nn-detection" src="https://github.com/supervisely-ecosystem/template-serve-nn-detection/releases/download/v0.0.1/thumb.png" width="500px" style='padding-bottom: 20px'/>  

**Step 2.** Run the application from the context menu of `.pth` file. If you are running application from file with different than `.pth` extension, app will use demo model

<img src="https://github.com/supervisely-ecosystem/template-serve-nn-detection/releases/download/v0.0.1/how-to-run-2.png" style="width:80%;"/>

**Step 3.** Press the Run button in the modal window

<img src="https://github.com/supervisely-ecosystem/template-serve-nn-detection/releases/download/v0.0.1/modal.png" style="width:50%;"/>

**Step 4.** Add one of the [related apps](https://github.com/supervisely-ecosystem/template-serve-nn-detection/edit/dev-readme/README.md#related-apps) to your team from Ecosystem and run it.

**Step 5.** Open running applier app and connect to app session with served model

<div>
  <table>
    <tr style="width: 100%">
      <td>
        <b>NN Image Labeling</b>
        <img src="https://github.com/supervisely-ecosystem/template-serve-nn-detection/releases/download/v0.0.1/how-to-run-image-connect.png" style="width:100%;"/>
      </td>
      <td>
        <b>Apply NN to Images Project</b>
        <img src="https://github.com/supervisely-ecosystem/template-serve-nn-detection/releases/download/v0.0.1/how-to-run-project-connect.png" style="width:100%;"/>
      </td>
    </tr>
  </table>
</div>

**Step 6.** Your served model is ready to apply. 

Once you integrated serving app for your model, you can use any available inference interfaces in Ecosystem:


* [NN Image Labeling](https://ecosystem.supervise.ly/apps/supervisely-ecosystem%252Fnn-image-labeling%252Fannotation-tool) - **Apply served model to image** 

<img data-key="sly-module-link" data-module-slug="supervisely-ecosystem/nn-image-labeling/annotation-tool" src="https://github.com/supervisely-ecosystem/template-serve-nn-detection/releases/download/v0.0.1/related-apps-apply-to-image-thumb.png" width="450px" style='padding-bottom: 20px'/>  

* [Apply NN to Images Project](https://ecosystem.supervise.ly/apps/supervisely-ecosystem%252Fnn-image-labeling%252Fproject-dataset) - **Apply served model to whole project or dataset**

<img data-key="sly-module-link" data-module-slug="supervisely-ecosystem/nn-image-labeling/project-dataset" src="https://github.com/supervisely-ecosystem/template-serve-nn-detection/releases/download/v0.0.1/related-apps-apply-to-project-thumb.png" width="450px" style='padding-bottom: 20px'/>  

