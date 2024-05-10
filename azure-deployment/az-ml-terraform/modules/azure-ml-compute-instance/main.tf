# Compute instance
resource "azurerm_machine_learning_compute_instance" "main" {
  name                          = var.compute_instance_name
  machine_learning_workspace_id = var.machine_learning_workspace_id
  virtual_machine_size          = var.virtual_machine_size
  
}