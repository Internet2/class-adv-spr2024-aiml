# Azure AI/ML Deployment

## Architecture

### Provisioning
```mermaid
graph TD;
   tf[Terraform]--builds-->azml;
   tf --builds-->sa;
   tf --builds-->misc;
   subgraph azml[Azure Machine Learning Workspace]
     subclust[Job Submission\nCluster]
   end
   subgraph sa[Storage Account]
     results[Results]
     datasets[Data Sets]
   end
   subgraph misc[Misc. Azure Utils]
     kv[Key Vault]
     logs[Log Analytics]
     insights[Application\nInsights]
   end

```

### Data Science Workflow

```mermaid
graph LR;
  scientist[🙋‍♀️👩‍🔬🧑‍💻 Data Scientists] --Git Push--> repo[Pipeline Repository\non GitHub]
  scientist -->azml
  scientist --Copy In-->data
  subgraph azml[Azure Machine Learning Workspace]
    compute[Compute Node]
    subgraph computeC[Compute Cluster]
      computeP[Prebuilt]
      computeD[Dynamic]
    end
  end
  repo --> azml
  compute[Job Submission\nCluster] -. generate .->computeD
  compute[Job Submission\nCluster] -. utilize .->computeP

  computeC --Writes-->results
  computeC <--Reads--> data
  subgraph sa[Storage Account]
    results[Results]
    data[Data Sets]
  end
```

## Workflow

* Devops deploys infrastructure with Terraform
* Researchers interact with ML Workspace using either a GitHub pipeline, or interactively with a Jupyterlabs notebook

### Github Action Examples

* 2 scripts located in a [separate repository](https://github.com/jimwood/azureml-examples) forked from official AzureML examples
* Repository secrets contain Azure Service Principal credentials and pointers to the Azure ML Workspace
* Examples:
  * Pipeline job to process NYC Taxi Data: ([job definition](https://github.com/jimwood/azureml-examples/tree/main/cli/jobs/pipelines/nyc-taxi)) ([workflow](https://github.com/jimwood/azureml-examples/blob/main/.github/workflows/cli-jobs-pipelines-nyc-taxi-pipeline.yml))
    * Demonstrates deploying a multi-step pipeline to the workspace
    * Configured with the name of a known compute cluster within the workspace
    * Not Azure specific, could be any workload suitable for the specified cluster
  * Command job to analyze Bank Data: ([job definition](https://github.com/jimwood/azureml-examples/tree/main/cli/jobs/single-step/dynamic-compute/bank-data)) ([workflow](https://github.com/jimwood/azureml-examples/blob/main/.github/workflows/cli-jobs-single-step-bank-data-job.yml))
    * Taken from a Jupyter Notebook defined in the workspace and modified to run unattended
    * Runs a single step on a known compute cluster in the workspace designed to handle self-provisioning jobs
    * Uses azureml Python SDK to self provision an appropriately sized compute cluster within the workspace
    * Programmatically runs multi-step process passing inputs and outputs forward