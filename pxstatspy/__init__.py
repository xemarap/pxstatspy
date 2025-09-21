"""
PxAPI-2 Python Wrapper
A Python client for the Statistics Sweden PxAPI-2 REST API.
"""

from .pxapi import PxAPI, PxAPIConfig, OutputFormat, OutputFormatParam, PxAPIError

__version__ = "0.2.0"
__all__ = ['PxAPI', 'PxAPIConfig', 'OutputFormat', 'OutputFormatParam', 'PxAPIError']