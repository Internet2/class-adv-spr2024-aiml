# Dependent resources for Azure Machine Learning
resource "azurerm_application_insights" "default" {
  name = var.application_insights_name
  location = var.location
  resource_group_name = var.resource_group_name
  application_type = "web"
}

resource "azurerm_key_vault" "default" {
  name = var.key_vault_name
  location = var.location
  resource_group_name = var.resource_group_name
  tenant_id = var.tenant_id
  sku_name = "standard"
  purge_protection_enabled = false
}

resource "azurerm_storage_account" "default" {
  name = var.storage_account_name
  location = var.location
  resource_group_name = var.resource_group_name
  account_tier = "Standard"
  account_replication_type = "LRS"
  allow_nested_items_to_be_public = false
}

resource "azurerm_container_registry" "default" {
  name = var.container_registry_name
  location = var.location
  resource_group_name = var.resource_group_name
  sku = "Standard"
  admin_enabled = true
}

# Machine Learning Resources
resource "azurerm_machine_learning_workspace" "default" {
  name = var.azure_ml_workspace_name
  location = var.location
  resource_group_name = var.resource_group_name
  application_insights_id = azurerm_application_insights.default.id
  key_vault_id = azurerm_key_vault.default.id
  storage_account_id = azurerm_storage_account.default.id
  container_registry_id = azurerm_container_registry.default.id
  public_network_access_enabled = true

  identity {
    type = "SystemAssigned"
  }
}