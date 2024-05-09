#!/usr/bin/env python
# coding: utf-8

# Copyright (c) Microsoft Corporation. All rights reserved.
# 
# Licensed under the MIT License.

# ![Impressions](https://PixelServer20190423114238.azurewebsites.net/api/impressions/MachineLearningNotebooks/how-to-use-azureml/automated-machine-learning/classification-bank-marketing-all-features/auto-ml-classification-bank-marketing-all-features.png)

# # Automated Machine Learning
# _**Classification with Deployment using a Bank Marketing Dataset**_
# 
# ## Contents
# 1. [Introduction](#Introduction)
# 1. [Setup](#Setup)
# 1. [Train](#Train)
# 1. [Results](#Results)
# 1. [Deploy](#Deploy)
# 1. [Test](#Test)
# 1. [Use auto-generated code for retraining](#Using-the-auto-generated-model-training-code-for-retraining-on-new-data)
# 1. [Acknowledgements](#Acknowledgements)

# ## Introduction
# 
# In this example we use the UCI Bank Marketing dataset to showcase how you can use AutoML for a  classification problem and deploy it to an Azure Container Instance (ACI). The classification goal is to predict if the client will subscribe to a term deposit with the bank.
# 
# If you are using an Azure Machine Learning Compute Instance, you are all set.  Otherwise, go through the [configuration](../../../configuration.ipynb)  notebook first if you haven't already to establish your connection to the AzureML Workspace. 
# 
# Please find the ONNX related documentations [here](https://github.com/onnx/onnx).
# 
# In this notebook you will learn how to:
# 1. Create an experiment using an existing workspace.
# 2. Configure AutoML using `AutoMLConfig`.
# 3. Train the model using local compute with ONNX compatible config on.
# 4. Explore the results, featurization transparency options and save the ONNX model
# 5. Inference with the ONNX model.
# 6. Register the model.
# 7. Create a container image.
# 8. Create an Azure Container Instance (ACI) service.
# 9. Test the ACI service.
# 10. Leverage the auto generated training code and use it for retraining on an updated dataset
# 
# In addition this notebook showcases the following features
# - **Blocking** certain pipelines
# - Specifying **target metrics** to indicate stopping criteria
# - Handling **missing data** in the input

# ## Setup
# 
# As part of the setup you have already created an Azure ML `Workspace` object. For AutoML you will need to create an `Experiment` object, which is a named object in a `Workspace` used to run experiments.

# In[ ]:

print("Starting run!")
import json
import logging

from matplotlib import pyplot as plt
import pandas as pd
import os

import azureml.core
from azureml.core.experiment import Experiment
from azureml.core.workspace import Workspace
from azureml.core.dataset import Dataset
from azureml.train.automl import AutoMLConfig
from azureml.interpret import ExplanationClient

## Pull in a couple env variables to make testing with various sizes easier
## These can be changed by running the script from a terminal tab with:
## $ VM_SIZE=STANDARD_DS12_V2 MAX_NODES=2 python ./run.py
vm_size = os.getenv("VM_SIZE", "STANDARD_DS12_V2")
max_nodes = os.getenv("MAX_NODES", 6)
cpu_cluster_name = f"{vm_size}-{max_nodes}".replace("_", "-")

print(f"Going to run this on the VM {vm_size} with {max_nodes} max nodes")
# This sample notebook may use features that are not available in previous versions of the Azure ML SDK.

# Accessing the Azure ML workspace requires authentication with Azure.
# 
# The default authentication is interactive authentication using the default tenant.  Executing the `ws = Workspace.from_config()` line in the cell below will prompt for authentication the first time that it is run.
# 
# If you have multiple Azure tenants, you can specify the tenant by replacing the `ws = Workspace.from_config()` line in the cell below with the following:
# 
# ```
# from azureml.core.authentication import InteractiveLoginAuthentication
# auth = InteractiveLoginAuthentication(tenant_id = 'mytenantid')
# ws = Workspace.from_config(auth = auth)
# ```
# 
# If you need to run in an environment where interactive login is not possible, you can use Service Principal authentication by replacing the `ws = Workspace.from_config()` line in the cell below with the following:
# 
# ```
# from azureml.core.authentication import ServicePrincipalAuthentication
# auth = auth = ServicePrincipalAuthentication('mytenantid', 'myappid', 'mypassword')
# ws = Workspace.from_config(auth = auth)
# ```
# For more details, see [aka.ms/aml-notebook-auth](http://aka.ms/aml-notebook-auth)

# In[ ]:


ws = Workspace.from_config()

# choose a name for experiment
experiment_name = "automl-classification-bmarketing-all"

experiment = Experiment(ws, experiment_name)

output = {}
output["Subscription ID"] = ws.subscription_id
output["Workspace"] = ws.name
output["Resource Group"] = ws.resource_group
output["Location"] = ws.location
output["Experiment Name"] = experiment.name
output["SDK Version"] = azureml.core.VERSION
pd.set_option("display.max_colwidth", None)
outputDf = pd.DataFrame(data=output, index=[""])
outputDf.T


# ## Create or Attach existing AmlCompute
# You will need to create a compute target for your AutoML run. In this tutorial, you create AmlCompute as your training compute resource.
# 
# > Note that if you have an AzureML Data Scientist role, you will not have permission to create compute resources. Talk to your workspace or IT admin to create the compute targets described in this section, if they do not already exist.
# 
# #### Creation of AmlCompute takes approximately 5 minutes. 
# If the AmlCompute with that name is already in your workspace this code will skip the creation process.
# As with other Azure services, there are limits on certain resources (e.g. AmlCompute) associated with the Azure Machine Learning service. Please read [this article](https://docs.microsoft.com/en-us/azure/machine-learning/service/how-to-manage-quotas) on the default limits and how to request more quota.

# In[ ]:


from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core.compute_target import ComputeTargetException

# Verify that cluster does not exist already
try:
    compute_target = ComputeTarget(workspace=ws, name=cpu_cluster_name)
    print("Found existing cluster, use it.")
except ComputeTargetException:
    compute_config = AmlCompute.provisioning_configuration(
        vm_size=vm_size, max_nodes=int(max_nodes)
    )
    compute_target = ComputeTarget.create(ws, cpu_cluster_name, compute_config)
compute_target.wait_for_completion(show_output=True)


# # Data

# ### Load Data
# 
# Leverage azure compute to load the bank marketing dataset as a Tabular Dataset into the dataset variable. 

# ### Training Data

# In[ ]:


data = pd.read_csv(
    "https://automlsamplenotebookdata.blob.core.windows.net/automl-sample-notebook-data/bankmarketing_train.csv"
)
data.head()


# In[ ]:


# Add missing values in 75% of the lines.
import numpy as np

missing_rate = 0.75
n_missing_samples = int(np.floor(data.shape[0] * missing_rate))
missing_samples = np.hstack(
    (
        np.zeros(data.shape[0] - n_missing_samples, dtype=bool),
        np.ones(n_missing_samples, dtype=bool),
    )
)
rng = np.random.RandomState(0)
rng.shuffle(missing_samples)
missing_features = rng.randint(0, data.shape[1], n_missing_samples)
data.values[np.where(missing_samples)[0], missing_features] = np.nan


# In[ ]:


if not os.path.isdir("data"):
    os.mkdir("data")
# Save the train data to a csv to be uploaded to the datastore
pd.DataFrame(data).to_csv("data/train_data.csv", index=False)

ds = ws.get_default_datastore()
ds.upload(
    src_dir="./data", target_path="bankmarketing", overwrite=True, show_progress=True
)


# Upload the training data as a tabular dataset for access during training on remote compute
train_data = Dataset.Tabular.from_delimited_files(
    path=ds.path("bankmarketing/train_data.csv")
)
label = "y"


# ### Validation Data

# In[ ]:


validation_data = "https://automlsamplenotebookdata.blob.core.windows.net/automl-sample-notebook-data/bankmarketing_validate.csv"
validation_dataset = Dataset.Tabular.from_delimited_files(validation_data)


# ### Test Data

# In[ ]:


test_data = "https://automlsamplenotebookdata.blob.core.windows.net/automl-sample-notebook-data/bankmarketing_test.csv"
test_dataset = Dataset.Tabular.from_delimited_files(test_data)


# ## Train
# 
# Instantiate a AutoMLConfig object. This defines the settings and data used to run the experiment.
# 
# |Property|Description|
# |-|-|
# |**task**|classification or regression or forecasting|
# |**primary_metric**|This is the metric that you want to optimize. Classification supports the following primary metrics: <br><i>accuracy</i><br><i>AUC_weighted</i><br><i>average_precision_score_weighted</i><br><i>norm_macro_recall</i><br><i>precision_score_weighted</i>|
# |**iteration_timeout_minutes**|Time limit in minutes for each iteration.|
# |**blocked_models** | *List* of *strings* indicating machine learning algorithms for AutoML to avoid in this run. <br><br> Allowed values for **Classification**<br><i>LogisticRegression</i><br><i>SGD</i><br><i>MultinomialNaiveBayes</i><br><i>BernoulliNaiveBayes</i><br><i>SVM</i><br><i>LinearSVM</i><br><i>KNN</i><br><i>DecisionTree</i><br><i>RandomForest</i><br><i>ExtremeRandomTrees</i><br><i>LightGBM</i><br><i>GradientBoosting</i><br><i>TensorFlowDNN</i><br><i>TensorFlowLinearClassifier</i><br><br>Allowed values for **Regression**<br><i>ElasticNet</i><br><i>GradientBoosting</i><br><i>DecisionTree</i><br><i>KNN</i><br><i>LassoLars</i><br><i>SGD</i><br><i>RandomForest</i><br><i>ExtremeRandomTrees</i><br><i>LightGBM</i><br><i>TensorFlowLinearRegressor</i><br><i>TensorFlowDNN</i><br><br>Allowed values for **Forecasting**<br><i>ElasticNet</i><br><i>GradientBoosting</i><br><i>DecisionTree</i><br><i>KNN</i><br><i>LassoLars</i><br><i>SGD</i><br><i>RandomForest</i><br><i>ExtremeRandomTrees</i><br><i>LightGBM</i><br><i>TensorFlowLinearRegressor</i><br><i>TensorFlowDNN</i><br><i>Arima</i><br><i>Prophet</i>|
# |**allowed_models** |  *List* of *strings* indicating machine learning algorithms for AutoML to use in this run. Same values listed above for **blocked_models** allowed for **allowed_models**.|
# |**experiment_exit_score**| Value indicating the target for *primary_metric*. <br>Once the target is surpassed the run terminates.|
# |**experiment_timeout_hours**| Maximum amount of time in hours that all iterations combined can take before the experiment terminates.|
# |**enable_early_stopping**| Flag to enble early termination if the score is not improving in the short term.|
# |**featurization**| 'auto' / 'off'  Indicator for whether featurization step should be done automatically or not. Note: If the input data is sparse, featurization cannot be turned on.|
# |**n_cross_validations**|Number of cross validation splits.|
# |**training_data**|Input dataset, containing both features and label column.|
# |**label_column_name**|The name of the label column.|
# |**enable_code_generation**|Flag to enable generation of training code for each of the models that AutoML is creating.
# 
# **_You can find more information about primary metrics_** [here](https://docs.microsoft.com/en-us/azure/machine-learning/service/how-to-configure-auto-train#primary-metric)

# In[ ]:


automl_settings = {
    "experiment_timeout_hours": 1.00,
    "enable_early_stopping": True,
    "iteration_timeout_minutes": 5,
    "max_concurrent_iterations": int(max_nodes),
    "max_cores_per_iteration": -1,
    # "n_cross_validations": 2,
    "primary_metric": "AUC_weighted",
    "featurization": "auto",
    "verbosity": logging.INFO,
    "enable_code_generation": True,
}

automl_config = AutoMLConfig(
    task="classification",
    debug_log="automl_errors.log",
    compute_target=compute_target,
    experiment_exit_score=0.9984,
    blocked_models=["KNN", "LinearSVM"],
    enable_onnx_compatible_models=True,
    training_data=train_data,
    label_column_name=label,
    validation_data=validation_dataset,
    show_output=True,
    **automl_settings,
)


# Call the `submit` method on the experiment object and pass the run configuration. Execution of local runs is synchronous. Depending on the data and the number of iterations this can run for a while. Validation errors and current status will be shown when setting `show_output=True` and the execution will be synchronous.

# In[ ]:


remote_run = experiment.submit(automl_config, show_output=False)


# Run the following cell to access previous runs. Uncomment the cell below and update the run_id.

# In[ ]:


# from azureml.train.automl.run import AutoMLRun
# remote_run = AutoMLRun(experiment=experiment, run_id='<run_ID_goes_here')
# remote_run


# In[ ]:


# Wait for the remote run to complete
remote_run.wait_for_completion()


# In[ ]:


# Retrieve the best Run object
best_run = remote_run.get_best_child()


# ## Transparency
# 
# View featurization summary for the best model - to study how different features were transformed. This is stored as a JSON file in the outputs directory for the run.

# In[ ]:


# Download the featurization summary JSON file locally
best_run.download_file(
    "outputs/featurization_summary.json", "featurization_summary.json"
)

# Render the JSON as a pandas DataFrame
with open("featurization_summary.json", "r") as f:
    records = json.load(f)

pd.DataFrame.from_records(records)


# ## Results

# In[ ]:


# NOTE: This appears to just be relevant for running inside of a jupyterlab notebook
# from azureml.widgets import RunDetails
# RunDetails(remote_run).show()


# ### Retrieve the Best Model's explanation
# Retrieve the explanation from the best_run which includes explanations for engineered features and raw features. Make sure that the run for generating explanations for the best model is completed.

# In[ ]:


# Wait for the best model explanation run to complete
from azureml.core.run import Run

model_explainability_run_id = remote_run.id + "_" + "ModelExplain"
print(model_explainability_run_id)
model_explainability_run = Run(
    experiment=experiment, run_id=model_explainability_run_id
)
model_explainability_run.wait_for_completion()

# Get the best run object
best_run = remote_run.get_best_child()


# #### Download engineered feature importance from artifact store
# You can use ExplanationClient to download the engineered feature explanations from the artifact store of the best_run.

# In[ ]:


client = ExplanationClient.from_run(best_run)
engineered_explanations = client.download_model_explanation(raw=False)
exp_data = engineered_explanations.get_feature_importance_dict()
exp_data


# #### Download raw feature importance from artifact store
# You can use ExplanationClient to download the raw feature explanations from the artifact store of the best_run.

# In[ ]:


client = ExplanationClient.from_run(best_run)
engineered_explanations = client.download_model_explanation(raw=True)
exp_data = engineered_explanations.get_feature_importance_dict()
exp_data


# ### Retrieve the Best ONNX Model
# 
# Below we select the best pipeline from our iterations. The `get_output` method returns the best run and the fitted model. The Model includes the pipeline and any pre-processing.  Overloads on `get_output` allow you to retrieve the best run and fitted model for *any* logged metric or for a particular *iteration*.
# 
# Set the parameter return_onnx_model=True to retrieve the best ONNX model, instead of the Python model.

# In[ ]:


best_run, onnx_mdl = remote_run.get_output(return_onnx_model=True)


# ### Save the best ONNX model

# In[ ]:


from azureml.automl.runtime.onnx_convert import OnnxConverter

onnx_fl_path = "./best_model.onnx"
OnnxConverter.save_onnx_model(onnx_mdl, onnx_fl_path)


# ### Predict with the ONNX model, using onnxruntime package

# In[ ]:


import sys
import json
from azureml.automl.core.onnx_convert import OnnxConvertConstants
from azureml.train.automl import constants

from azureml.automl.runtime.onnx_convert import OnnxInferenceHelper


def get_onnx_res(run):
    res_path = "onnx_resource.json"
    run.download_file(
        name=constants.MODEL_RESOURCE_PATH_ONNX, output_file_path=res_path
    )
    with open(res_path) as f:
        result = json.load(f)
    return result


if sys.version_info < OnnxConvertConstants.OnnxIncompatiblePythonVersion:
    test_df = test_dataset.to_pandas_dataframe()
    mdl_bytes = onnx_mdl.SerializeToString()
    onnx_result = get_onnx_res(best_run)

    onnxrt_helper = OnnxInferenceHelper(mdl_bytes, onnx_result)
    pred_onnx, pred_prob_onnx = onnxrt_helper.predict(test_df)

    print(pred_onnx)
    print(pred_prob_onnx)
else:
    print("Please use Python version 3.6 or 3.7 to run the inference helper.")


# ## Deploy
# 
# ### Retrieve the Best Model
# 
# Below we select the best pipeline from our iterations.  The `get_best_child` method returns the Run object for the best model based on the default primary metric. There are additional flags that can be passed to the method if we want to retrieve the best Run based on any of the other supported metrics, or if we are just interested in the best run among the ONNX compatible runs. As always, you can execute `??remote_run.get_best_child` in a new cell to view the source or docs for the function.

# In[ ]:


# get_ipython().run_line_magic('pinfo2', 'remote_run.get_best_child')


# #### Widget for Monitoring Runs
# 
# The widget will first report a "loading" status while running the first iteration. After completing the first iteration, an auto-updating graph and table will be shown. The widget will refresh once per minute, so you should see the graph update as child runs complete.
# 
# **Note:** The widget displays a link at the bottom. Use this link to open a web interface to explore the individual run details

# In[ ]:


best_run = remote_run.get_best_child()


# In[ ]:


model_name = best_run.properties["model_name"]

script_file_name = "inference/score.py"

best_run.download_file("outputs/scoring_file_v_1_0_0.py", "inference/score.py")


# ### Register the Fitted Model for Deployment
# If neither `metric` nor `iteration` are specified in the `register_model` call, the iteration with the best primary metric is registered.

# In[ ]:


description = "AutoML Model trained on bank marketing data to predict if a client will subscribe to a term deposit"
tags = None
model = remote_run.register_model(
    model_name=model_name, description=description, tags=tags
)

print(
    remote_run.model_id
)  # This will be written to the script file later in the notebook.


# ### Deploy the model as a Web Service on Azure Container Instance

# In[ ]:


from azureml.core.model import InferenceConfig
from azureml.core.webservice import AciWebservice
from azureml.core.webservice import Webservice
from azureml.core.model import Model
from azureml.core.environment import Environment

inference_config = InferenceConfig(
    environment=best_run.get_environment(), entry_script=script_file_name
)

aciconfig = AciWebservice.deploy_configuration(
    cpu_cores=2,
    memory_gb=2,
    tags={"area": "bmData", "type": "automl_classification"},
    description="sample service for Automl Classification",
)

aci_service_name = model_name.lower()
print(aci_service_name)
aci_service = Model.deploy(ws, aci_service_name, [model], inference_config, aciconfig)
aci_service.wait_for_deployment(True)
print(aci_service.state)


# ### Get Logs from a Deployed Web Service
# 
# Gets logs from a deployed web service.

# In[ ]:


# aci_service.get_logs()


# ## Test
# 
# Now that the model is trained, run the test data through the trained model to get the predicted values.  This calls the ACI web service to do the prediction.
# 
# Note that the JSON passed to the ACI web service is an array of rows of data.  Each row should either be an array of values in the same order that was used for training or a dictionary where the keys are the same as the column names used for training.  The example below uses dictionary rows.

# In[ ]:


# Load the bank marketing datasets.
from numpy import array


# In[ ]:


X_test = test_dataset.drop_columns(columns=["y"])
y_test = test_dataset.keep_columns(columns=["y"], validate=True)
test_dataset.take(5).to_pandas_dataframe()


# In[ ]:


X_test = X_test.to_pandas_dataframe()
y_test = y_test.to_pandas_dataframe()


# In[ ]:


import requests

X_test_json = X_test.to_json(orient="records")
data = '{"data": ' + X_test_json + "}"
headers = {"Content-Type": "application/json"}

resp = requests.post(aci_service.scoring_uri, data, headers=headers)

y_pred = json.loads(json.loads(resp.text))["result"]


# In[ ]:


actual = array(y_test)
actual = actual[:, 0]
print(len(y_pred), " ", len(actual))


# ### Calculate metrics for the prediction
# 
# Now visualize the data as a confusion matrix that compared the predicted values against the actual values.
# 

# In[ ]:

# get_ipython().run_line_magic('matplotlib', 'notebook')
from sklearn.metrics import confusion_matrix
import itertools

cf = confusion_matrix(actual, y_pred)
plt.imshow(cf, cmap=plt.cm.Blues, interpolation="nearest")
plt.colorbar()
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
class_labels = ["no", "yes"]
tick_marks = np.arange(len(class_labels))
plt.xticks(tick_marks, class_labels)
plt.yticks([-0.5, 0, 1, 1.5], ["", "no", "yes", ""])
# plotting text value inside cells
thresh = cf.max() / 2.0
for i, j in itertools.product(range(cf.shape[0]), range(cf.shape[1])):
    plt.text(
        j,
        i,
        format(cf[i, j], "d"),
        horizontalalignment="center",
        color="white" if cf[i, j] > thresh else "black",
    )
plt.show()


# ### Delete a Web Service
# 
# Deletes the specified web service.

# In[ ]:


aci_service.delete()


# ### Using the auto generated model training code for retraining on new data
# 
# Because we enabled code generation when the original experiment was created, we now have access to the code that was used to generate any of the AutoML tried models. Below we'll be using the generated training script of the best model to retrain on a new dataset.
# 
# For this demo, we'll begin by creating new retraining dataset by combining the Train & Validation datasets that were used in the original experiment.

# In[ ]:


original_train_data = pd.read_csv(
    "https://automlsamplenotebookdata.blob.core.windows.net/automl-sample-notebook-data/bankmarketing_train.csv"
)

valid_data = pd.read_csv(
    "https://automlsamplenotebookdata.blob.core.windows.net/automl-sample-notebook-data/bankmarketing_validate.csv"
)

# we'll emulate an updated dataset for retraining by combining the Train & Validation datasets into a new one
retrain_pd = pd.concat([original_train_data, valid_data])
retrain_pd.to_csv("data/retrain_data.csv", index=False)
ds.upload_files(
    files=["data/retrain_data.csv"],
    target_path="bankmarketing/",
    overwrite=True,
    show_progress=True,
)
retrain_dataset = Dataset.Tabular.from_delimited_files(
    path=ds.path("bankmarketing/retrain_data.csv")
)

# after creating and uploading the retraining dataset, let's register it with the workspace for reuse
retrain_dataset = retrain_dataset.register(
    workspace=ws,
    name="Bankmarketing_retrain",
    description="Updated training dataset, includes validation data",
    create_new_version=True,
)


# Next, we'll download the generated script for the best run and use it for retraining. For more advanced scenarios, you can customize the training script as you need: change the featurization pipeline, change the learner algorithm or its hyperparameters, etc. 
# 
# For this exercise, we'll leave the script as it was generated.

# In[ ]:


# download the autogenerated training script into the generated_code folder
best_run.download_file(
    "outputs/generated_code/script.py", "generated_code/training_script.py"
)

# view the contents of the autogenerated training script
# get_ipython().system(' cat generated_code/training_script.py')


# In[ ]:


import uuid
from azureml.core import ScriptRunConfig
from azureml._restclient.models import RunTypeV2
from azureml._restclient.models.create_run_dto import CreateRunDto
from azureml._restclient.run_client import RunClient

codegen_runid = str(uuid.uuid4())
client = RunClient(
    experiment.workspace.service_context,
    experiment.name,
    codegen_runid,
    experiment_id=experiment.id,
)

# override the training_dataset_id to point to our new retraining dataset we just registered above
dataset_arguments = ["--training_dataset_id", retrain_dataset.id]

# create the retraining run as a child of the AutoML generated training run
create_run_dto = CreateRunDto(
    run_id=codegen_runid,
    parent_run_id=best_run.id,
    description="AutoML Codegen Script Run using an updated training dataset",
    target=cpu_cluster_name,
    run_type_v2=RunTypeV2(orchestrator="Execution", traits=["automl-codegen"]),
)

# the script for retraining run is pointing to the AutoML generated script
src = ScriptRunConfig(
    source_directory="generated_code",
    script="training_script.py",
    arguments=dataset_arguments,
    compute_target=cpu_cluster_name,
    environment=best_run.get_environment(),
)
run_dto = client.create_run(run_id=codegen_runid, create_run_dto=create_run_dto)

# submit the experiment
retraining_run = experiment.submit(config=src, run_id=codegen_runid)
retraining_run


# After the run completes, we can get download/test/deploy to the model it has built.

# In[ ]:


retraining_run.wait_for_completion()

retraining_run.download_file("outputs/model.pkl", "generated_code/model.pkl")


# ## Acknowledgements

# This Bank Marketing dataset is made available under the Creative Commons (CCO: Public Domain) License: https://creativecommons.org/publicdomain/zero/1.0/. Any rights in individual contents of the database are licensed under the Database Contents License: https://creativecommons.org/publicdomain/zero/1.0/ and is available at: https://www.kaggle.com/janiobachmann/bank-marketing-dataset .
# 
# _**Acknowledgements**_
# This data set is originally available within the UCI Machine Learning Database: https://archive.ics.uci.edu/ml/datasets/bank+marketing
# 
# [Moro et al., 2014] S. Moro, P. Cortez and P. Rita. A Data-Driven Approach to Predict the Success of Bank Telemarketing. Decision Support Systems, Elsevier, 62:22-31, June 2014
