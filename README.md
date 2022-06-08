<div align="center" markdown>

<img src="" style="width: 100%;"/>

# Template Serve NN Detection

<p align="center">
  <a href="#Overview">Overview</a> •
  <a href="#How-To-Run">How To Run</a> •
  <a href="#Related-Apps">Related Apps</a> •
  <a href="#Result">Result</a>
</p>


[![](https://img.shields.io/badge/supervisely-ecosystem-brightgreen)](https://ecosystem.supervise.ly/apps/supervisely-ecosystem/template-serve-nn-detection)
[![](https://img.shields.io/badge/slack-chat-green.svg?logo=slack)](https://supervise.ly/slack)
![GitHub release (latest SemVer)](https://img.shields.io/github/v/release/supervisely-ecosystem/template-serve-nn-detection)
[![views](https://app.supervise.ly/public/api/v3/ecosystem.counters?repo=supervisely-ecosystem/template-serve-nn-detection&counter=views&label=views)](https://supervise.ly)
[![used by teams](https://app.supervise.ly/public/api/v3/ecosystem.counters?repo=supervisely-ecosystem/template-serve-nn-detection&counter=downloads&label=used%20by%20teams)](https://supervise.ly)
[![runs](https://app.supervise.ly/public/api/v3/ecosystem.counters?repo=supervisely-ecosystem/template-serve-nn-detection&counter=runs&label=runs&123)](https://supervise.ly)

</div>

# Overview

Template Serve NN Detection app is designed for **developers** and can be used as a starting point for creating an application for serving your own custom NN models on Supervisely.

By default template app generates random predictions with random scores to demonstrate the functionality, in order to implement your custom model, you will need to edit **`src/main.py`**. Inference results will be automatically converted to [supervisely annotation format](https://docs.supervise.ly/data-organization/00_ann_format_navi).

# How To Run

**Note:** recommended Python version for supervisely is 3.8.x

## Common steps for demo and local run:

**Step 1.** In order to run the app you must configure environment variables. Learn more about configuring applciation environment in our [app creation guide](https://github.com/supervisely-ecosystem/how-to-create-app)
  * **a.** Add [While True Script](https://ecosystem.supervise.ly/apps/while-true-script) app from Ecosystem to your Team and Run it on your agent (watch step **e.** for more info).
  * **b.** [Copy `Task ID` of `While True Script App` to `debug.env` file](https://ecosystem.supervise.ly/apps/while-true-script)
  * **c.** Copy your `Team` and `Workspace` IDs to corresponding variables in `debug.env` : `context.teamId=XXX`,  `context.workspaceId=XXX`
  * **d.** Create `modal.state.slyFile="empty"` variable in debug.env
  * **e.** Add variables to `debug.env` to store app data and cache: `DEBUG_APP_DIR="/path/to/data_dir"`, `DEBUG_CACHE_DIR="/path/to/cache_dir"`
  * **f.** Create `secret_debug.env` file in app root dir, your personal API tokens will be stored here. Follow this [video guide](https://github.com/supervisely-ecosystem/how-to-create-app#2-hyper-quickstart-guide) to create `secret_debug.env`.

`.env` files should look like these:

`debug.env`
```
PYTHONUNBUFFERED=1

TASK_ID=XXX

context.teamId=XXX
context.workspaceId=XXX

modal.state.slyFile = "empty"

DEBUG_APP_DIR="/path/to/data_dir"
DEBUG_CACHE_DIR="/path/to/cache_dir"

SERVER_ADDRESS="put your value here"
API_TOKEN="put your value here"
AGENT_TOKEN="put your value here"
```

`secret_debug.env`
```
SERVER_ADDRESS="https://app.supervise.ly/" # or enterprise instance
API_TOKEN="xPxxxxxeOtqgjI5IHgVZgHdBgpYa0xxxxxxxxxxxxuuJXaoIq3Jit7TAbiQLxxxxxxxxxxxxtNpIQ2sDtsoxxxxxHjKlf1TNGDSexxxxiiAxSToxxbUxCxxxD50xxxxX"
AGENT_TOKEN="xxxkM8xxQxxxTxxxsNUEXxxxxMoxxxxx"
```

**Step 2.** Create python virtual environment by running the following command from the application root directory in terminal:

```bash
python -m venv venv
```

**Step 3.** Activate venv:

```bash
source venv/bin/activate
```

**Step 4.** Install requirements.txt:

```bash
pip install -r requirements.txt
```

**Step 5.** Make sure you've edited `main.py`, without edits it will generate random predictions

## Demo run:

**Step 1.** Use the following command in terminal to move to `src` directory:

```bash
cd src
```

**Step 2.** Run `main.py` with arguments from terminal:

```bash
python main.py /path/to/original/image /prediction/image/save/path
```

## Local run:

**Step 1.** Add one of the [related apps](https://github.com/supervisely-ecosystem/template-serve-nn-detection/edit/dev-readme/README.md#related-apps) to your team from Ecosystem and run it.

**Step 2. Optional** Locate your custom nn model in Team Files and copy it's path to `modal.state.slyFile="/path/to/nn_model.pth"` variable in `debug.env` file in the app root directory. If you skip this step, app will generate random predictions.


**Step 3.** Use the following command in terminal to move to `src` directory:

```bash
cd src
```

**Step 4.** Run `main.py` from terminal:

```bash
python main.py
```

**Step 5.** Open running applier app and connect to app session with served model

<div>
  <table>
    <tr style="width: 100%">
      <td>
        <b>NN Image Labeling</b>
        <img src="" style="width:100%;"/>
      </td>
      <td>
        <b>Apply NN to Images Project</b>
        <img src="" style="width:100%;"/>
      </td>
    </tr>
  </table>
</div>

**Step 6.** Apply NN model to image or whole project/dataset

<div>
  <table>
    <tr style="width: 100%">
      <td>
        <b>NN Image Labeling</b>
        <img src="" style="width:100%;"/>
      </td>
      <td>
        <b>Apply NN to Images Project</b>
        <img src="" style="width:100%;"/>
      </td>
    </tr>
  </table>
</div>

## Instance run:

**Step 1.** Add [Template Serve NN Detection](https://ecosystem.supervise.ly/apps/supervisely-ecosystem%252Ftemplate-serve-nn-detection) to your team from Ecosystem

<img src="" style="width:100%;"/>

**Step 2.** Run the application from the context menu of `pth` file.

<img src="" style="width:100%;"/>

**Step 3.** Press the Run button in the modal window

<img src="" style="width:100%;"/>

**Step 4.** Add one of the [related apps](https://github.com/supervisely-ecosystem/template-serve-nn-detection/edit/dev-readme/README.md#related-apps) to your team from Ecosystem and run it.

**Step 5.** Open running applier app and connect to app session with served model

<div>
  <table>
    <tr style="width: 100%">
      <td>
        <b>NN Image Labeling</b>
        <img src="" style="width:100%;"/>
      </td>
      <td>
        <b>Apply NN to Images Project</b>
        <img src="" style="width:100%;"/>
      </td>
    </tr>
  </table>
</div>

**Step 6.** Apply NN model to image or whole project/dataset

<div>
  <table>
    <tr style="width: 100%">
      <td>
        <b>NN Image Labeling</b>
        <img src="" style="width:100%;"/>
      </td>
      <td>
        <b>Apply NN to Images Project</b>
        <img src="" style="width:100%;"/>
      </td>
    </tr>
  </table>
</div>

# Related apps

Learn how to use served models in the corresponding apps:

* [NN Image Labeling](https://ecosystem.supervise.ly/apps/supervisely-ecosystem%252Fnn-image-labeling%252Fannotation-tool) - **Apply served model to image** 
* [Apply NN to Images Project](https://ecosystem.supervise.ly/apps/supervisely-ecosystem%252Fnn-image-labeling%252Fproject-dataset) - **Apply served model to whole project**

# Result

<img src="" style="width:80%;"/>



