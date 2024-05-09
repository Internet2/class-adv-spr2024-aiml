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
  subgraph azml[Azure Machine\nLearning Workspace]
    compute[Compute Node]
    subgraph computeC[Compute Cluster]
      computeP[Prebuilt]
      computeD[Dynamically Built]
    end
  end
  repo --> azml
  compute[Job Submission\nCluster] -. generates .->computeC[Short Lived\nCompute Cluster]
  computeC --Writes-->results
  computeC <--Reads--> data
  subgraph sa[Storage Account]
    results[Results]
    data[Data Sets]
  end
```

## TODO

Handle blob storage for holding data/models - Data asset points to this

Terraform Templates
- Spin up data science machine
- Get blob storage credentials
- Kubernetes Cluster with 10x machines, etc

JupyterLab instance Kubernetes orchestration

ML/AI Expert
- Write something that breaks the ML/AI thing out

Also maybe cost analysis?
