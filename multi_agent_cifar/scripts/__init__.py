"""Operational scripts for the multi-agent supervisor.

Turn-key utilities that read the blackboard / events log / knowledge base
but never participate in the supervisor hot path. Import as a subpackage
(e.g. ``python -m multi_agent.scripts.tool_usage``) so relative imports
into ``..harness`` resolve correctly.
"""
