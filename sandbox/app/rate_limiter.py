import time
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Optional
import logging

import boto3
from botocore.exceptions import ClientError
from fastapi import Request

logger = logging.getLogger(__name__)


class DynamoDBClient(ABC):
    @abstractmethod
    def get_item(self, **kwargs) -> dict:
        pass

    @abstractmethod
    def put_item(self, **kwargs) -> dict:
        pass


class RateLimitExceededException(Exception):
    pass


class RateLimiter:
    def __init__(
        self,
        dynamodb_client: DynamoDBClient,
        table_name: str,
        requests_per_hour: int = 20,
    ):
        self.dynamodb_client = dynamodb_client
        self.table_name = table_name
        self.requests_per_hour = requests_per_hour

    def _get_current_hour_key(self) -> str:
        """Get the current hour as a string key for rate limiting.

        Returns:
            Hour key in format: YYYY-MM-DD-HH
        """
        now = datetime.now(timezone.utc)
        return now.strftime("%Y-%m-%d-%H")

    def _get_rate_limit_key(self, ip_address: str) -> str:
        """Generate the primary key for rate limiting storage.

        Args:
            ip_address: Client IP address

        Returns:
            Primary key combining IP and current hour
        """
        hour_key = self._get_current_hour_key()
        return f"{ip_address}#{hour_key}"

    async def check_and_update_rate_limit(self, ip_address: str) -> None:
        """Check if IP address is within rate limit and update counter.

        Args:
            ip_address: Client IP address to check

        Raises:
            RateLimitExceededException: If rate limit is exceeded
        """
        rate_limit_key = self._get_rate_limit_key(ip_address)

        try:
            # Try to get existing record
            response = self.dynamodb_client.get_item(
                TableName=self.table_name, Key={"rate_limit_key": {"S": rate_limit_key}}
            )

            current_count = 0
            if "Item" in response:
                current_count = int(response["Item"]["request_count"]["N"])

            # Check if limit exceeded
            if current_count >= self.requests_per_hour:
                logger.warning(
                    f"Rate limit exceeded for IP {ip_address}. "
                    f"Count: {current_count}/{self.requests_per_hour}"
                )
                raise RateLimitExceededException(
                    f"Rate limit exceeded. Maximum {self.requests_per_hour} requests per hour allowed."
                )

            # Update counter
            new_count = current_count + 1
            current_timestamp = int(time.time())

            self.dynamodb_client.put_item(
                TableName=self.table_name,
                Item={
                    "rate_limit_key": {"S": rate_limit_key},
                    "ip_address": {"S": ip_address},
                    "request_count": {"N": str(new_count)},
                    "hour_key": {"S": self._get_current_hour_key()},
                    "last_updated": {"N": str(current_timestamp)},
                    "ttl": {
                        "N": str(current_timestamp + 7200)
                    },  # 2 hours TTL for cleanup
                },
            )

            logger.info(
                f"Rate limit check passed for IP {ip_address}. "
                f"Count: {new_count}/{self.requests_per_hour}"
            )

        except ClientError as e:
            logger.error(f"DynamoDB error during rate limit check: {e}")
            logger.warning("Allowing request due to DynamoDB error (failing open)")

    async def get_remaining_requests(self, ip_address: str) -> int:
        """Get remaining requests for an IP address in current hour.

        Args:
            ip_address: Client IP address

        Returns:
            Number of remaining requests in current hour
        """
        rate_limit_key = self._get_rate_limit_key(ip_address)

        try:
            response = self.dynamodb_client.get_item(
                TableName=self.table_name, Key={"rate_limit_key": {"S": rate_limit_key}}
            )

            current_count = 0
            if "Item" in response:
                current_count = int(response["Item"]["request_count"]["N"])

            return max(0, self.requests_per_hour - current_count)

        except ClientError as e:
            logger.error(f"DynamoDB error getting remaining requests: {e}")
            return self.requests_per_hour  # Fail open


class BotoDynamoDBClient(DynamoDBClient):
    def __init__(
        self,
        endpoint_url: Optional[str] = None,
        region_name: str = "eu-west-1",
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
    ):
        self._client = boto3.client(
            "dynamodb",
            endpoint_url=endpoint_url,
            region_name=region_name,
            aws_access_key_id=aws_access_key_id
            or "test",  # LocalStack doesn't need real creds
            aws_secret_access_key=aws_secret_access_key or "test",
        )

    def get_item(self, **kwargs) -> dict:
        return self._client.get_item(**kwargs)

    def put_item(self, **kwargs) -> dict:
        return self._client.put_item(**kwargs)


class MockDynamoDBClient(DynamoDBClient):
    def __init__(self, fail_on_error: bool = False):
        self._storage = {}
        self._fail_on_error = fail_on_error

    def get_item(self, **kwargs) -> dict:
        if self._fail_on_error:
            raise ClientError(
                error_response={"Error": {"Code": "ServiceUnavailable"}},
                operation_name="GetItem",
            )

        table_name = kwargs.get("TableName")
        key = kwargs.get("Key", {})

        rate_limit_key = None
        if "rate_limit_key" in key and "S" in key["rate_limit_key"]:
            rate_limit_key = key["rate_limit_key"]["S"]

        if not rate_limit_key:
            return {}

        storage_key = f"{table_name}#{rate_limit_key}"
        if storage_key in self._storage:
            return {"Item": self._storage[storage_key]}

        return {}

    def put_item(self, **kwargs) -> dict:
        if self._fail_on_error:
            raise ClientError(
                error_response={"Error": {"Code": "ServiceUnavailable"}},
                operation_name="PutItem",
            )

        table_name = kwargs.get("TableName")
        item = kwargs.get("Item", {})

        rate_limit_key = None
        if "rate_limit_key" in item and "S" in item["rate_limit_key"]:
            rate_limit_key = item["rate_limit_key"]["S"]

        if rate_limit_key:
            storage_key = f"{table_name}#{rate_limit_key}"
            self._storage[storage_key] = item

        return {"ResponseMetadata": {"HTTPStatusCode": 200}}

    def clear_storage(self):
        self._storage.clear()

    def get_storage_contents(self) -> dict:
        return self._storage.copy()


def create_dynamodb_client(
    endpoint_url: Optional[str] = None,
    region_name: str = "eu-west-1",
    aws_access_key_id: Optional[str] = None,
    aws_secret_access_key: Optional[str] = None,
) -> DynamoDBClient:
    """Create DynamoDB client with configuration.

    Args:
        endpoint_url: DynamoDB endpoint (use for LocalStack)
        region_name: AWS region name
        aws_access_key_id: AWS access key (optional for LocalStack)
        aws_secret_access_key: AWS secret key (optional for LocalStack)

    Returns:
        Configured DynamoDB client
    """
    return BotoDynamoDBClient(
        endpoint_url=endpoint_url,
        region_name=region_name,
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
    )


def get_client_ip(request: Request) -> str:
    """Extract client IP address from FastAPI request.

    Handles various proxy scenarios by checking headers in order of preference.

    Args:
        request: FastAPI Request object

    Returns:
        Client IP address as string
    """
    # Check for forwarded headers (common in load balancers/proxies)
    forwarded_for = request.headers.get("X-Forwarded-For")
    if forwarded_for:
        # X-Forwarded-For can contain multiple IPs, take the first (original client)
        return forwarded_for.split(",")[0].strip()

    # Check for real IP header (common in some proxy setups)
    real_ip = request.headers.get("X-Real-IP")
    if real_ip:
        return real_ip.strip()

    # Fall back to direct connection IP
    if request.client and request.client.host:
        return request.client.host

    # Fallback for testing/unknown scenarios
    return "unknown"


def create_rate_limiter(
    table_name: str = "rate_limits",
    requests_per_hour: int = 20,
    endpoint_url: Optional[str] = None,
) -> RateLimiter:
    """Factory function to create configured rate limiter.

    Args:
        table_name: DynamoDB table name
        requests_per_hour: Rate limit threshold
        endpoint_url: DynamoDB endpoint (for LocalStack)

    Returns:
        Configured RateLimiter instance
    """
    dynamodb_client = create_dynamodb_client(endpoint_url=endpoint_url)
    return RateLimiter(
        dynamodb_client=dynamodb_client,
        table_name=table_name,
        requests_per_hour=requests_per_hour,
    )
