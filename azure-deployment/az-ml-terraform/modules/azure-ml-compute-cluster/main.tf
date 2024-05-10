# Compute Cluster
resource "azurerm_machine_learning_compute_cluster" "compute" {
  name                          = var.compute_cluster_name
  machine_learning_workspace_id = var.machine_learning_workspace_id
  
  location                      = var.location
  vm_priority                   = var.vm_priority
  vm_size                       = var.vm_size

  identity {
    type = "SystemAssigned"
  }

  scale_settings {
    min_node_count                       = var.min_node_count
    max_node_count                       = var.max_node_count
    scale_down_nodes_after_idle_duration = var.idle_scale_down_timeout
  }

}