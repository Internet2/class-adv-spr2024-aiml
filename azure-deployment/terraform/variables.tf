# Noting existing workspace to test
data "azurerm_machine_learning_workspace" "existing_workspace" {
  name                = "ml-immortal-egret-mlw"
  resource_group_name = "ml-immortal-egret-rg"
}

variable "location" {
  type        = string
  description = "Location of the resources"
  default     = "eastus2"
}

variable "resourceGroupName" {
  type        = string
  default     = "ml-immortal-egret-rg"
        }

variable "prefix" {
  type        = string
  description = "Prefix of the resource name"
  default     = "i2-ml"
}