import time
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Optional
import logging

import boto3
from botocore.exceptions import ClientError
from fastapi import Request
import anyio
from botocore.config import Config as BotoConfig
import os

logger = logging.getLogger(__name__)


class DynamoDBClient(ABC):
    @abstractmethod
    def get_item(self, **kwargs) -> dict:
        pass

    @abstractmethod
    def put_item(self, **kwargs) -> dict:
        pass

    @abstractmethod
    def update_item(self, **kwargs) -> dict:
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

        Uses a single atomic UpdateItem with conditional expression to avoid
        race conditions and minimise round trips.

        Args:
            ip_address: Client IP address to check

        Raises:
            RateLimitExceededException: If rate limit is exceeded
        """
        rate_limit_key = self._get_rate_limit_key(ip_address)

        hour_key = self._get_current_hour_key()
        current_timestamp = int(time.time())
        ttl_seconds = 7200  # 2 hours TTL for cleanup

        def _update_item():
            return self.dynamodb_client.update_item(
                TableName=self.table_name,
                Key={"rate_limit_key": {"S": rate_limit_key}},
                UpdateExpression=(
                    "ADD request_count :inc "
                    "SET ip_address = if_not_exists(ip_address, :ip), "
                    "hour_key = :hour, last_updated = :now, ttl = :ttl"
                ),
                ExpressionAttributeValues={
                    ":inc": {"N": "1"},
                    ":ip": {"S": ip_address},
                    ":hour": {"S": hour_key},
                    ":now": {"N": str(current_timestamp)},
                    ":ttl": {"N": str(current_timestamp + ttl_seconds)},
                    ":limit": {"N": str(self.requests_per_hour)},
                },
                ConditionExpression=(
                    "attribute_not_exists(request_count) OR request_count < :limit"
                ),
                ReturnValues="UPDATED_NEW",
            )

        try:
            await anyio.to_thread.run_sync(_update_item)
            logger.info(
                f"Rate limit check passed for IP {ip_address}. "
                f"Key: {rate_limit_key}"
            )
        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code")
            if error_code == "ConditionalCheckFailedException":
                logger.warning(
                    f"Rate limit exceeded for IP {ip_address}. "
                    f"Key: {rate_limit_key}"
                )
                raise RateLimitExceededException(
                    f"Rate limit exceeded. Maximum {self.requests_per_hour} requests per hour allowed."
                )
            logger.error(f"DynamoDB error during rate limit check: {e}")
            logger.warning("Allowing request due to DynamoDB error (failing open)")
        except Exception as e:
            # Fail open for any unexpected exception (e.g., endpoint connection error)
            logger.error(f"Unexpected error during rate limit check: {e}")
            logger.warning("Allowing request due to unexpected error (failing open)")

    async def get_remaining_requests(self, ip_address: str) -> int:
        """Get remaining requests for an IP address in current hour.

        Args:
            ip_address: Client IP address

        Returns:
            Number of remaining requests in current hour
        """
        rate_limit_key = self._get_rate_limit_key(ip_address)

        def _get_item():
            return self.dynamodb_client.get_item(
                TableName=self.table_name, Key={"rate_limit_key": {"S": rate_limit_key}}
            )

        try:
            response = await anyio.to_thread.run_sync(_get_item)

            current_count = 0
            if "Item" in response:
                current_count = int(response["Item"].get("request_count", {}).get("N", "0"))

            return max(0, self.requests_per_hour - current_count)

        except ClientError as e:
            logger.error(f"DynamoDB error getting remaining requests: {e}")
            return self.requests_per_hour  # Fail open
        except Exception as e:
            logger.error(f"Unexpected error getting remaining requests: {e}")
            return self.requests_per_hour  # Fail open


class BotoDynamoDBClient(DynamoDBClient):
    def __init__(
        self,
        endpoint_url: Optional[str] = None,
        region_name: str = "eu-west-1",
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
    ):
        # Aggressive timeouts and minimal retries so tests don’t hang if endpoint is down
        client_config = BotoConfig(
            connect_timeout=float(os.environ.get("DDB_CONNECT_TIMEOUT", "0.3")),
            read_timeout=float(os.environ.get("DDB_READ_TIMEOUT", "0.3")),
            retries={"max_attempts": int(os.environ.get("DDB_MAX_ATTEMPTS", "1")), "mode": "standard"},
        )
        self._client = boto3.client(
            "dynamodb",
            endpoint_url=endpoint_url,
            region_name=region_name,
            aws_access_key_id=aws_access_key_id
            or "test",  # LocalStack doesn't need real creds
            aws_secret_access_key=aws_secret_access_key or "test",
            config=client_config,
        )

    def get_item(self, **kwargs) -> dict:
        return self._client.get_item(**kwargs)

    def put_item(self, **kwargs) -> dict:
        return self._client.put_item(**kwargs)

    def update_item(self, **kwargs) -> dict:
        return self._client.update_item(**kwargs)


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

    def update_item(self, **kwargs) -> dict:
        if self._fail_on_error:
            raise ClientError(
                error_response={"Error": {"Code": "ServiceUnavailable"}},
                operation_name="UpdateItem",
            )

        table_name = kwargs.get("TableName")
        key = kwargs.get("Key", {})
        expr_values = kwargs.get("ExpressionAttributeValues", {})

        rate_limit_key = None
        if "rate_limit_key" in key and "S" in key["rate_limit_key"]:
            rate_limit_key = key["rate_limit_key"]["S"]

        if not rate_limit_key:
            raise ClientError(
                error_response={"Error": {"Code": "ValidationException"}},
                operation_name="UpdateItem",
            )

        storage_key = f"{table_name}#{rate_limit_key}"
        item = self._storage.get(storage_key, {})

        current_count = int(item.get("request_count", {}).get("N", "0"))
        limit = int(expr_values.get(":limit", {}).get("N", "0"))
        inc = int(expr_values.get(":inc", {}).get("N", "1"))

        if current_count >= limit and limit > 0:
            # Simulate conditional check failure
            raise ClientError(
                error_response={
                    "Error": {"Code": "ConditionalCheckFailedException"}
                },
                operation_name="UpdateItem",
            )

        # Apply update
        new_count = current_count + inc
        updated_item = {
            "rate_limit_key": {"S": rate_limit_key},
            "ip_address": item.get("ip_address")
            or {"S": expr_values.get(":ip", {}).get("S", "")},
            "request_count": {"N": str(new_count)},
            "hour_key": {"S": expr_values.get(":hour", {}).get("S", "")},
            "last_updated": {"N": expr_values.get(":now", {}).get("N", "0")},
            "ttl": {"N": expr_values.get(":ttl", {}).get("N", "0")},
        }
        self._storage[storage_key] = updated_item

        return {"Attributes": {"request_count": {"N": str(new_count)}}}

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
