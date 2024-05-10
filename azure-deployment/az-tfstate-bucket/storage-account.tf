terraform {
  required_providers {
    azurerm = {
      source = "hashicorp/azurerm"
      version = ">= 3.0, < 4.0"
    }
  }
}

provider "azurerm" {
  features {}

  subscription_id = "1f0714f3-ad90-44a8-91cd-f58d8083895e"
  tenant_id = "cb72c54e-4a31-4d9e-b14a-1ea36dfac94c"
}

resource "azurerm_resource_group" "i2ml-tfstate" {
  name = "i2ml-tfstate"
  location = "eastus2"
}
resource "azurerm_storage_account" "i2ml-tfstate" {
  name = "i2mltfstate"
  resource_group_name = azurerm_resource_group.i2ml-tfstate.name
  location = azurerm_resource_group.i2ml-tfstate.location
  account_tier = "Standard"
  account_replication_type = "LRS"
  
}

resource "azurerm_storage_container" "i2ml-tfstate" {
  name = "tfstate"
  storage_account_name = azurerm_storage_account.i2ml-tfstate.name
  container_access_type = "blob"
}