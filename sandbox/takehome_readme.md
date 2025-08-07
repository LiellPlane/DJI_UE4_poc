# Take Home Assignment Updates

- Converted to UV Python dependency & virtual environment manager
- AI generated rate_limiter instead of library such as SlowAPI or limits.py due to complications with non-Redis backend
- Various async performance fixes in main.py 
- added tests for rate limiter and server endpoint