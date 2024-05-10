variable "machine_learning_workspace_id" {
  description = "Object guid of the Azure ML workspace to attach compute intance to."
  type = string
}

variable "compute_instance_name" {
  description = "Name of the Azure ML compute instance."
  type = string
}

variable "virtual_machine_size" {
  description = "Azure Compute vm size sku"
  type = string
  default = "Standard_DS2_v2"
}