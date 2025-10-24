# Take Home Assignment Updates

- Converted to UV Python dependency & virtual environment manager
- AI generated rate_limiter instead of library such as SlowAPI or limits.py due to complications with custom backend
- Various async performance fixes in main.py 
- added tests for rate limiter and server endpoint
- added FastAPI middleware
- ran out of time so did not test deployment!

# Recommendations

- Do not use DynamodB for rate limiting - use a battle-tested library with Redis or MemoryDB backend