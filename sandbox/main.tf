provider "aws" {
  region                      = "us-east-1"
  access_key                  = "test"
  secret_key                  = "test"
  skip_credentials_validation = true
  skip_metadata_api_check     = true
  skip_requesting_account_id  = true

  endpoints {
    dynamodb = "http://localhost:4566"
  }
}

resource "aws_dynamodb_table" "rate_limits" {
  name           = "rate_limits"
  billing_mode   = "PAY_PER_REQUEST"
  hash_key       = "rate_limit_key"

  attribute {
    name = "rate_limit_key"
    type = "S"
  }

  ttl {
    attribute_name = "ttl"
    enabled        = true
  }

  tags = {
    Name        = "RateLimits"
    Environment = "development"
  }
}
