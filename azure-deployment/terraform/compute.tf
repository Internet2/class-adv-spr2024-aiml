resource "random_pet" "my_random_name" {
  prefix = "i2"
}

# Compute instance
resource "azurerm_machine_learning_compute_instance" "main" {
  name                          = "${random_pet.my_random_name.id}-ci"
  machine_learning_workspace_id = data.azurerm_machine_learning_workspace.existing_workspace.id
  location                      = data.azurerm_machine_learning_workspace.existing_workspace.location
  virtual_machine_size          = "Standard_DS2_v2"
}

# Compute Cluster
resource "azurerm_machine_learning_compute_cluster" "compute" {
  name                          = "i2-cpu-cluster"
  machine_learning_workspace_id = data.azurerm_machine_learning_workspace.existing_workspace.id
  location                      = data.azurerm_machine_learning_workspace.existing_workspace.location
  vm_priority                   = "Dedicated"
  vm_size                       = "Standard_DS2_v2"

  identity {
    type = "SystemAssigned"
  }

  scale_settings {
    min_node_count                       = 0
    max_node_count                       = 3
    scale_down_nodes_after_idle_duration = "PT15M" # 15 minutes
  }

}