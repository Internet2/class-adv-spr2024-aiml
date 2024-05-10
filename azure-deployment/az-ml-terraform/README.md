# Azure Machine Learning Workspace

This Terraform deployment stamps out an [Azure Machine Learning Workspace](https://docs.microsoft.com/en-us/azure/machine-learning/concept-workspace), with it's dependent resources:
* Azure Application Insights
* Azure Storage 
* Azure Container Registry
* Azure Key Vault

Additionally, it stamps out 4 autoscaling compute clusters with varying CPU, Memory, and storage allocations, as well as varying min/max node scaling.
| Cluster | Cores | Memory | Storage | Min nodes | Max Nodes |
| - | - | - | - | - | - |
| `Lite` | 2 | 14GB | 28GB | 0 | 2 |
| `Small` | 4 | 14GB | 28GB | 0 | 2 |
| `Medium` | 4 | 32GB | 150GB | 0 | 5 |
| `Large` | 8 | 56GB | 400GB | 0 | 10 |


## Resources in this template
| Terraform Resource Type | Description |
| - | - |
| `azurerm_resource_group` | The resource group all resources get deployed into. |
| `azurerm_application_insights` | An Azure Application Insights instance associated to the Azure ML workspace. |
| `azurerm_key_vault` | An Azure Key Vault instance associated with the Azure ML Workspace. |
| `azurerm_storage_account` | An Azure Storage instance associated with the Azure ML Workspace. |
| `azurerm_container_registry` | An Azure Container registry instance associated with the Azure ML Workspace. |
| `azurerm_machine_learning_compute_cluster` | A managed compute cluster with autoscaling. |

## Variables
| Name | Description | Default |
| - | - | - |
| environment | The deployment environment name (used for prefixing resource names) | dev |
| location | The Azure region used for deployments | East US 2 (eastus2) |

## Usage
Ensure you are have a valid token in the Azure CLI (az login should get you there)
Ensure you have the correct Azure Subscription selected (az account show)

```bash
terraform init
terraform plan -out plan.tfplan
terraform apply plan.tfplan
```

## TODO
* Tie in a kubernetes cluster to the workspace