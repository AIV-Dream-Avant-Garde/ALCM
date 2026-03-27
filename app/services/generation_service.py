"""Generation service — Score-to-Generation pipeline.

Phase 1: Logic is in core/prompt_assembly.py + routers/generate.py.
Phase 2: Will add full 7-layer caching, token budget management, and
output validation via Safeguard Gateway.
"""
# Phase 2: Extract from routers/generate.py with caching layer
