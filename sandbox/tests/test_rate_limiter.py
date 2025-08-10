import pytest
import anyio
from app.rate_limiter import RateLimiter, MockDynamoDBClient, RateLimitExceededException


@pytest.fixture
def mock_client():
    return MockDynamoDBClient()


@pytest.fixture
def rate_limiter(mock_client):
    return RateLimiter(
        dynamodb_client=mock_client, table_name="test_table", requests_per_hour=5
    )


def test_allows_request_within_limit(rate_limiter):
    async def _test():
        await rate_limiter.check_and_update_rate_limit("192.168.1.1")

    anyio.run(_test)


def test_blocks_request_over_limit(rate_limiter):
    async def _test():
        ip = "192.168.1.2"

        # Make 5 requests (the limit)
        for _ in range(5):
            await rate_limiter.check_and_update_rate_limit(ip)

        # 6th request should fail
        with pytest.raises(RateLimitExceededException):
            await rate_limiter.check_and_update_rate_limit(ip)

    anyio.run(_test)


def test_remaining_requests_decreases(rate_limiter):
    async def _test():
        ip = "192.168.1.3"

        # Initially should have 5 requests available
        remaining = await rate_limiter.get_remaining_requests(ip)
        assert remaining == 5

        # After one request, should have 4 remaining
        await rate_limiter.check_and_update_rate_limit(ip)
        remaining = await rate_limiter.get_remaining_requests(ip)
        assert remaining == 4

    anyio.run(_test)


def test_different_ips_separate_limits(rate_limiter):
    async def _test():
        ip1 = "192.168.1.4"
        ip2 = "192.168.1.5"

        # Use up limit for ip1
        for _ in range(5):
            await rate_limiter.check_and_update_rate_limit(ip1)

        # ip2 should still work
        await rate_limiter.check_and_update_rate_limit(ip2)
        remaining = await rate_limiter.get_remaining_requests(ip2)
        assert remaining == 4

    anyio.run(_test)


def test_handles_dynamodb_errors_gracefully(rate_limiter):
    async def _test():
        # Set up client to fail
        rate_limiter.dynamodb_client._fail_on_error = True

        # Should not raise exception (fails open)
        await rate_limiter.check_and_update_rate_limit("192.168.1.6")

    anyio.run(_test)
