terraform {
  required_version = ">= 1.0"

  required_providers {
    azurerm = {
      source = "hashicorp/azurerm"
      version = ">= 3.0, < 4.0"
    }
    random = {
      source = "hashicorp/random"
      version = ">= 3.0"
    }
  }

  backend "azurerm" {
    resource_group_name = "i2ml-tfstate"
    storage_account_name = "i2mltfstate"
    container_name = "tfstate"
    key = "terraform.tfstate"
  }
}

provider "azurerm" {
  features {
    key_vault {
      recover_soft_deleted_key_vaults = false
      purge_soft_delete_on_destroy = false
      purge_soft_deleted_keys_on_destroy = false
    }
    resource_group {
      prevent_deletion_if_contains_resources = false
    }
  }
}