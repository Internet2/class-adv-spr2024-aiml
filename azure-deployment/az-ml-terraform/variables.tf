variable "location" {
  type        = string
  description = "Location of the resources"
  default     = "eastus2"
}

variable "prefix" {
  type        = string
  description = "Prefix of the resource name"
  default     = "i2ml"
}

variable "environment" {
  type = string
  description = "Development sandbox for Internet2 CLASS Advanced AI/ML project."
  default = "dev"
}