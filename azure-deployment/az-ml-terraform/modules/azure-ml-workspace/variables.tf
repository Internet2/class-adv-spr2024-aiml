variable "azure_ml_workspace_name" {
  description = "Name of the Azure ML Workspace."
  type = string
}
variable "application_insights_name" {
  description = "Name of the Application Insights instance."
  type = string
}
variable "key_vault_name" {
  description = "Name of the Azure Key Vault"
  type = string
}

variable "storage_account_name" {
  description = "Name of the Azure Blob Storage Account"
  type = string
}

variable "container_registry_name" {
  description = "Name of the Azure Container Registry"
  type = string
}

variable "resource_group_name" {
  description = "The Azure Resource Group to target for provisioning."
  type = string
}

variable "location" {
  description = "The Azure region to provision the resource."
  type = string
}

variable "tenant_id" {
  description = "The Azure tenant id"
  type = string
}