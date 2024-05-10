# Pulls in some useful data from your az cli instance login like client_id, tenant_id, etc
data "azurerm_client_config" "current" {}

# Uses the random provider to generate a unique resource group name.
resource "azurerm_resource_group" "default" {
  name = "${random_pet.prefix.id}-rg"
  location = var.location
}

# Generates a random pet name of length 2 (think ubuntu version names) with prefix from variables 
resource "random_pet" "prefix" {
  prefix = var.prefix
  length = 2
}

# Generates a random integer to serve as suffix for resources to avoid name collisions in globally constrained resources.
resource "random_integer" "suffix" {
  min = 10000000
  max = 99999999
}

module "az-ml-workspace" {
  source = "./modules/azure-ml-workspace"
  tenant_id = data.azurerm_client_config.current.tenant_id
  resource_group_name = azurerm_resource_group.default.name
  location = azurerm_resource_group.default.location
  azure_ml_workspace_name = "${random_pet.prefix.id}-mlw"
  application_insights_name = "${random_pet.prefix.id}-appi"
  key_vault_name = "${var.prefix}${var.environment}${random_integer.suffix.result}kv"
  storage_account_name = "${var.prefix}${var.environment}${random_integer.suffix.result}sa"
  container_registry_name = "${var.prefix}${var.environment}${random_integer.suffix.result}cr"
}

module "az-ml-compute-cluster-lite" {
  source = "./modules/azure-ml-compute-cluster"
  vm_size = "Standard_DS11_v2"
  min_node_count = 0
  max_node_count = 2
  compute_cluster_name = "${random_pet.prefix.id}-lite-cc"
  machine_learning_workspace_id = module.az-ml-workspace.machine_learning_workspace_id
  location = azurerm_resource_group.default.location
}

module "az-ml-compute-cluster-small" {
  source = "./modules/azure-ml-compute-cluster"
  vm_size = "Standard_DS3_v2"
  min_node_count = 0
  max_node_count = 2
  compute_cluster_name = "${random_pet.prefix.id}-small-cc"
  machine_learning_workspace_id = module.az-ml-workspace.machine_learning_workspace_id
  location = azurerm_resource_group.default.location
}

module "az-ml-compute-cluster-medium" {
  source = "./modules/azure-ml-compute-cluster"
  vm_size = "Standard_E4ds_v4"
  min_node_count = 0
  max_node_count = 5
  compute_cluster_name = "${random_pet.prefix.id}-medium-cc"
  machine_learning_workspace_id = module.az-ml-workspace.machine_learning_workspace_id
  location = azurerm_resource_group.default.location
}

module "az-ml-compute-cluster-large" {
  source = "./modules/azure-ml-compute-cluster"
  vm_size = "Standard_D13_v2"
  min_node_count = 0
  max_node_count = 10
  compute_cluster_name = "${random_pet.prefix.id}-large-cc"
  machine_learning_workspace_id = module.az-ml-workspace.machine_learning_workspace_id
  location = azurerm_resource_group.default.location
}