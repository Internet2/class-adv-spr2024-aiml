variable "compute_cluster_name" {
  description = "Name of the compute cluster"
  type = string
}

variable "machine_learning_workspace_id" {
  description = "ID of the Azure ML Workspace to attach to."
  type = string
}

variable "location" {
  description = "Azure region for the compute cluster."
  type = string
}

variable "vm_priority" {
  description = "value"
  type = string
  default = "Dedicated"
}

variable "vm_size" {
  description = "Azure Compute vm size sku"
  type = string
  default = "Standard_DS11_v2"

  validation {
    condition = contains(["Standard_DS11_v2", "Standard_DS3_v2", "Standard_E4ds_v4", "Standard_D13_v2"], var.vm_size)
    error_message = "Valid values for vm_size are (Standard_DS11_v2, Standard_DS3_v2, Standard_E4ds_v4, Standard_D13_v2)."
  }
}

variable "min_node_count" {
  description = "Azure Compute vm minimum nodes in cluster."
  type = number
  default = 0
}

variable "max_node_count" {
  description = "Azure Compute vm maximum nodes in cluster."
  type = number
  default = 3
}

variable "idle_scale_down_timeout" {
  description = "Idle time (in minutes) before scaling down nodes in cluster to minimum."
  type = string
  default = "PT15M" # 15 minutes
}