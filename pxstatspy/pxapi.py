"""
PxAPI-2 Python Wrapper
A Python client for the Statistics Sweden PxAPI-2 REST API.

Author: Emanuel Raptis
License: MIT
"""

import requests
from typing import Dict, List, Optional, Union, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import json
import pandas as pd
from io import StringIO
from datetime import datetime
import time
from collections import deque
from threading import Lock
import math

class RateLimiter:
    """Implements a sliding window rate limiter"""
    
    def __init__(self, max_calls: int, time_window: int):
        """
        Initialize rate limiter
        
        Args:
            max_calls: Maximum number of calls allowed in the time window
            time_window: Time window in seconds
        """
        self.max_calls = max_calls
        self.time_window = time_window
        self.calls = deque()
        self.lock = Lock()
        
    def wait_if_needed(self):
        """
        Check if we need to wait before making another call
        and wait if necessary
        """
        with self.lock:
            now = time.time()
            
            # Remove calls outside the time window
            while self.calls and now - self.calls[0] >= self.time_window:
                self.calls.popleft()
            
            if len(self.calls) >= self.max_calls:
                # Calculate sleep time based on oldest call
                wait_time = self.calls[0] + self.time_window - now
                if wait_time > 0:
                    time.sleep(wait_time)
                    # Update time and clean expired calls after waiting
                    now = time.time()
                    while self.calls and now - self.calls[0] >= self.time_window:
                        self.calls.popleft()
            
            # Add the current call timestamp
            self.calls.append(now)

class OutputFormat(str, Enum):
    PX = "px"
    JSON_STAT2 = "json-stat2" 
    CSV = "csv"
    XLSX = "xlsx"
    HTML = "html"
    JSON_PX = "json-px"
    PARQUET = "parquet"

class OutputFormatParam(str, Enum):
    USE_CODES = "UseCodes"
    USE_TEXTS = "UseTexts"
    USE_CODES_AND_TEXTS = "UseCodesAndTexts"
    INCLUDE_TITLE = "IncludeTitle"
    SEPARATOR_TAB = "SeparatorTab"
    SEPARATOR_SPACE = "SeparatorSpace"
    SEPARATOR_SEMICOLON = "SeparatorSemicolon"

@dataclass
class PxAPIConfig:
    """Configuration for PxAPI client"""
    base_url: str
    api_key: Optional[str] = None
    language: str = "en"

class PxAPIError(Exception):
    """Base exception for PxAPI errors"""
    pass

@dataclass
class NavigationItem:
    """Represents an item in the navigation structure"""
    id: str
    label: str
    type: str
    description: Optional[str] = None
    updated: Optional[datetime] = None
    tags: Optional[List[str]] = None
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'NavigationItem':
        """Create NavigationItem from API response dictionary"""
        updated = None
        if 'updated' in data:
            try:
                updated = datetime.fromisoformat(data['updated'].rstrip('Z'))
            except (ValueError, AttributeError):
                pass
                
        return cls(
            id=data['id'],
            label=data['label'],
            type=data['type'],
            description=data.get('description'),
            updated=updated,
            tags=data.get('tags', [])
        )

class NavigationExplorer:
    """Helper class for exploring the database structure"""
    
    def __init__(self, api: 'PxAPI'):
        self.api = api
        self._current_folder = None
        self.history = []
        
    def get_root(self) -> Dict[str, List[NavigationItem]]:
        """Get root folder contents categorized by type"""
        response = self.api.get_navigation_root()
        self._current_folder = response
        self.history = []
        return self._categorize_contents(response['folderContents'])
    
    def navigate_to(self, folder_id: str) -> Dict[str, List[NavigationItem]]:
        """Navigate to specific folder by ID"""
        response = self.api.get_navigation_by_id(folder_id)
        self._current_folder = response
        self.history.append(folder_id)
        return self._categorize_contents(response['folderContents'])
    
    def go_back(self) -> Optional[Dict[str, List[NavigationItem]]]:
        """Go back to previous folder"""
        if not self.history:
            return None
            
        self.history.pop()  # Remove current
        if not self.history:
            return self.get_root()
            
        return self.navigate_to(self.history[-1])
    
    def _categorize_contents(self, contents: List[Dict]) -> Dict[str, List[NavigationItem]]:
        """Organize folder contents by type"""
        categorized = {
            'folders': [],
            'tables': [],
            'headings': []
        }
        
        for item in contents:
            nav_item = NavigationItem.from_dict(item)
            if nav_item.type == 'FolderInformation':
                categorized['folders'].append(nav_item)
            elif nav_item.type == 'Table':
                categorized['tables'].append(nav_item)
            elif nav_item.type == 'Heading':
                categorized['headings'].append(nav_item)
                
        return categorized
    
    def print_current_location(self):
        """Print information about current folder"""
        if not self._current_folder:
            print("Not currently in any folder (call get_root() first)")
            return
            
        print(f"\nCurrent folder: {self._current_folder['label']}")
        if self._current_folder.get('description'):
            print(f"Description: {self._current_folder['description']}")
            
        contents = self._categorize_contents(self._current_folder['folderContents'])
        
        if contents['headings']:
            print("\nHeadings:")
            for item in contents['headings']:
                print(f"  - {item.label}")
                
        if contents['folders']:
            print("\nFolders:")
            for item in contents['folders']:
                print(f"  📁 {item.id} - {item.label}")
                if item.description:
                    print(f"    {item.description}")
                    
        if contents['tables']:
            print("\nTables:")
            for item in contents['tables']:
                print(f"  📊 {item.id} - {item.label}")
                if item.updated:
                    print(f"    Last updated: {item.updated.strftime('%Y-%m-%d')}")
                if item.tags:
                    print(f"    Tags: {', '.join(item.tags)}")

class PxVariables:
    """Constants for standard PxAPI-2 variable names and types"""
    
    # Common variable codes
    TIME = "Tid"                 # Time dimension variable
    REGION = "Region"            # Geographic region variable 
    CONTENTS = "ContentsCode"    # Content/measure variable
    
    # Variable type identifiers
    TYPE_TIME = "TimeVariable"
    TYPE_GEOGRAPHICAL = "GeographicalVariable" 
    TYPE_CONTENTS = "ContentsVariable"
    TYPE_REGULAR = "RegularVariable"
    
    # Swedish and English alternatives for detection
    TIME_ALTERNATIVES = ["tid", "år", "månad", "kvartal", "period", 
                         "time", "year", "quarter", "month"]
    
    REGION_ALTERNATIVES = ["region", "land", "riket", "län", "kommun", "deso", "regso", 
                           "nuts2", "nuts3", "fa-region", "la-region", "kommunkod", "länskod",
                           "geo", "area", "location", "county", "municipality", "country"]
    
    CONTENTS_ALTERNATIVES = ["innehåll", "tabellinnehåll", "mått", "värde", "variabel", 
                            "contents", "measure", "metric", "value", "variable"]

class SelectionMatcher:
    """Handles pattern matching and expansion for PxAPI selection expressions"""
    
    @staticmethod
    def count_matching_values(pattern: str, all_codes: List[str]) -> int:
        """
        Count how many values match a pattern or selection expression
        
        Args:
            pattern: Selection pattern (wildcard, expression, etc.)
            all_codes: List of all value codes to match against
            
        Returns:
            Number of matching codes
        """
        # Handle wildcard "*" directly for better performance
        if pattern == "*":
            return len(all_codes)
            
        # Handle other wildcards and question marks
        if "*" in pattern or "?" in pattern:
            import fnmatch
            matches = [code for code in all_codes if fnmatch.fnmatch(code, pattern)]
            return len(matches)
            
        # Handle selection expressions (TOP, RANGE, etc.)
        if "(" in pattern and ")" in pattern:
            try:
                matches = SelectionMatcher.expand_expression(pattern, all_codes)
                return len(matches)
            except ValueError:
                # If expression parsing fails, treat as direct match
                return 1 if pattern in all_codes else 0
                
        # Direct match
        return 1 if pattern in all_codes else 0
    
    @staticmethod
    def expand_expression(expr: str, all_codes: List[str]) -> List[str]:
        """
        Expand a selection expression into actual value codes
        
        Args:
            expr: Selection expression (e.g., 'TOP(5)', 'RANGE(2020,2023)')
            all_codes: List of all possible value codes
            
        Returns:
            List of value codes matching the expression
        """
        # Extract function name and arguments
        func = expr[:expr.find('(')].upper()
        args = expr[expr.find('(')+1:expr.find(')')].split(',')
        args = [arg.strip() for arg in args]
        
        if func == 'TOP':
            n = int(args[0])
            offset = int(args[1]) if len(args) > 1 else 0
            return all_codes[offset:offset+n]
            
        elif func == 'BOTTOM':
            n = int(args[0])
            offset = int(args[1]) if len(args) > 1 else 0
            return all_codes[-(n+offset):-offset if offset else None]
            
        elif func == 'RANGE':
            if len(args) != 2:
                raise ValueError(f"RANGE expression requires 2 arguments, got {len(args)}")
            start, end = args
            start_idx = all_codes.index(start)
            end_idx = all_codes.index(end)
            return all_codes[start_idx:end_idx+1]
            
        elif func == 'FROM':
            if len(args) != 1:
                raise ValueError(f"FROM expression requires 1 argument, got {len(args)}")
            start_idx = all_codes.index(args[0])
            return all_codes[start_idx:]
            
        elif func == 'TO':
            if len(args) != 1:
                raise ValueError(f"TO expression requires 1 argument, got {len(args)}")
            end_idx = all_codes.index(args[0])
            return all_codes[:end_idx+1]
            
        else:
            raise ValueError(f"Unknown selection expression: {func}")
    
    @staticmethod
    def get_matching_codes(patterns: List[str], all_codes: List[str]) -> List[str]:
        """
        Get all codes that match any of the given patterns
        
        Args:
            patterns: List of selection patterns
            all_codes: List of all possible value codes
            
        Returns:
            List of matching value codes (deduplicated)
        """
        matched_codes = []
        
        for pattern in patterns:
            # Handle wildcard "*" directly
            if pattern == "*":
                return all_codes
                
            # Handle other wildcards and question marks
            if "*" in pattern or "?" in pattern:
                import fnmatch
                matches = [code for code in all_codes if fnmatch.fnmatch(code, pattern)]
                matched_codes.extend(matches)
                continue
                
            # Handle selection expressions
            if "(" in pattern and ")" in pattern:
                try:
                    matches = SelectionMatcher.expand_expression(pattern, all_codes)
                    matched_codes.extend(matches)
                    continue
                except ValueError:
                    pass
                    
            # Direct match
            if pattern in all_codes:
                matched_codes.append(pattern)
                
        # Remove duplicates while preserving order
        return list(dict.fromkeys(matched_codes))

class PxAPI:
    """
    Python client for the Statistics Sweden PxAPI-2
    
    Args:
        config (PxAPIConfig): Configuration object with base URL and optional API key
        fetch_config (bool): Whether to fetch configuration from API (default: True)
    """
    
    def __init__(self, config: PxAPIConfig, fetch_config: bool = True):
        self.config = config
        self.session = requests.Session()
        self.debug = False
        
        # Set up authorization if API key provided
        if config.api_key:
            self.session.headers.update({"Authorization": f"Bearer {config.api_key}"})
        
        # Default values in case API is unreachable or fetch_config is False
        self._default_max_calls = 30
        self._default_time_window = 10
        self._default_max_data_cells = 150000
        
        # Initialize with default values
        self.max_data_cells = self._default_max_data_cells
        self.rate_limiter = RateLimiter(
            max_calls=self._default_max_calls,
            time_window=self._default_time_window
        )
        
        # Create navigator
        self.navigator = NavigationExplorer(self)
        
        # Fetch actual configuration if requested
        if fetch_config:
            try:
                api_config = self.get_config()
                
                # Update rate limiter with actual values
                self.rate_limiter = RateLimiter(
                    max_calls=api_config.get('maxCallsPerTimeWindow', self._default_max_calls),
                    time_window=api_config.get('timeWindow', self._default_time_window)
                )
                
                # Update max data cells
                self.max_data_cells = api_config.get('maxDataCells', self._default_max_data_cells)
                
                if self.debug:
                    print(f"API Configuration loaded:")
                    print(f"Max calls per window: {self.rate_limiter.max_calls}")
                    print(f"Time window: {self.rate_limiter.time_window} seconds")
                    print(f"Max data cells: {self.max_data_cells:,}")
                    
            except Exception as e:
                if self.debug:
                    print(f"Warning: Could not fetch API configuration: {str(e)}")
                    print("Using default values instead")

    def get_config(self) -> Dict:
        """Get API configuration settings"""
        response = self._make_request('GET', '/config')
        return response.json()

    def get_navigation_root(self) -> Dict:
        """Get root folder navigation"""
        response = self._make_request('GET', '/navigation')
        return response.json()

    def get_navigation_by_id(self, folder_id: str) -> Dict:
        """Get folder navigation by ID"""
        response = self._make_request('GET', f'/navigation/{folder_id}')
        return response.json()
    
    def find_tables(self,
                query: Optional[str] = None,
                past_days: Optional[int] = None,
                include_discontinued: bool = False,
                page_number: int = 1,
                page_size: Optional[int] = None,
                display: bool = True) -> Dict:
        """
        List and optionally display tables from the API with filtering options.

        This method serves two purposes:
        1. Data retrieval: When display=False, returns raw API response with table data
        2. Display functionality: When display=True, prints formatted table information
       and returns None

        Args:
            query: Search query string to filter tables by name or content
            past_days: Filter for tables updated in the last N days
            include_discontinued: If True, includes discontinued tables in results
            page_number: Page number for pagination (starts at 1)
            page_size: Number of items per page
            display: If True, prints formatted output and returns None
                If False, returns raw API response dictionary

        Returns:
            If display=False: Dict containing API response with table data
            If display=True: None (output is printed to console)
        """
        params = {
            'query': query,
            'pastDays': past_days,
            'includeDiscontinued': include_discontinued,
            'pageNumber': page_number,
            'pageSize': page_size
        }
        
        # Remove None values
        params = {k: v for k, v in params.items() if v is not None}
    
        response = self._make_request('GET', '/tables', params=params)
        data = response.json()
    
        if not display:
            return data
        
        # Print search information
        total_tables = data['page']['totalElements']
        current_page = data['page']['pageNumber']
        total_pages = data['page']['totalPages']
    
        print(f"\nFound {total_tables} tables", end="")
        if query:
            print(f" matching '{query}'", end="")
        if past_days:
            print(f" updated in the last {past_days} days", end="")
        print(f" (Page {current_page} of {total_pages})")
    
        # Print header
        print("\nID         Updated      First    Last     Title")
        print("-" * 70)
    
        # Print each table's information
        for table in data['tables']:
            # Format the date
            updated = "N/A        "
            if table.get('updated'):
                try:
                    date = datetime.fromisoformat(table['updated'].rstrip('Z'))
                    updated = date.strftime("%Y-%m-%d  ")
                except (ValueError, AttributeError):
                    updated = table['updated'][:10] + "  "
        
            # Format periods
            first_period = table.get('firstPeriod', 'N/A').ljust(8)
            last_period = table.get('lastPeriod', 'N/A').ljust(8)
        
            # Print the row
            print(f"{table['id']:<10} {updated}{first_period}{last_period}{table['label']}")
        
            # Print variables on the next line with indentation
            if table.get('variableNames'):
                vars_str = ", ".join(table['variableNames'])
                print(f"           Variables: {vars_str}")
            print()
    
        # Print pagination info
        if total_pages > 1:
            print(f"Page {current_page} of {total_pages}. Use page_number parameter to view other pages.")
    
        return None  # Return None when display=True to prevent automatic output
    
    def find_tables_as_dataframe(self,
                    query: Optional[str] = None,
                    past_days: Optional[int] = None,
                    include_discontinued: bool = False,
                    all_pages: bool = False) -> pd.DataFrame:
        """
        List tables from the API with filtering options and return results as a pandas DataFrame.
    
        Args:
            query: Search query string to filter tables by name or content
            past_days: Filter for tables updated in the last N days
            include_discontinued: If True, includes discontinued tables in results
            all_pages: If True, automatically fetches all pages and combines them
    
        Returns:
            pandas.DataFrame: DataFrame containing table information with columns:
                - id: Table identifier
                - label: Table title/description
                - updated: Last update date
                - first_period: First time period in the data
                - last_period: Last time period in the data
                - variables: List of variable names
                - discontinued: Whether the table is discontinued
                - category: Table category (public, private, etc.)
        """
        # Get first page to determine pagination info
        first_page = self.find_tables(
            query=query,
            past_days=past_days,
            include_discontinued=include_discontinued,
            page_number=1,
            page_size=None,  # Use API default
            display=False  # Get raw data instead of display
        )
    
        # Extract table data
        tables = first_page['tables']
    
        # Get pagination info
        total_pages = first_page['page']['totalPages']
        current_page = first_page['page']['pageNumber']
        page_size = first_page['page']['pageSize']
        total_elements = first_page['page']['totalElements']
    
        # If all_pages=True and there are more pages, fetch them
        if all_pages and total_pages > 1:
            print(f"\nFetching all {total_pages} pages containing {total_elements} tables...")
        
            # Fetch remaining pages and append to tables list
            for page_num in range(2, total_pages + 1):
                print(f"Fetching page {page_num} of {total_pages}...", end='\r')
            
                page_data = self.find_tables(
                    query=query,
                    past_days=past_days,
                    include_discontinued=include_discontinued,
                    page_number=page_num,
                    page_size=page_size,
                    display=False
                )
            
                # Append tables from this page
                tables.extend(page_data['tables'])
        
            print(f"\nSuccessfully retrieved all {len(tables)} tables.")
    
        # Convert to DataFrame
        df = pd.DataFrame(tables)
    
        # Reorder and select relevant columns
        columns = [
            'id', 'label', 'updated', 'firstPeriod', 'lastPeriod', 
            'variableNames', 'discontinued', 'category'
        ]
    
        # Only include columns that exist in the data
        columns = [col for col in columns if col in df.columns]
        df = df[columns]
    
        # Rename columns to more readable names
        column_mapping = {
            'firstPeriod': 'first_period',
            'lastPeriod': 'last_period',
            'variableNames': 'variables'
        }
        df = df.rename(columns=column_mapping)
    
        # Convert updated column to datetime
        if 'updated' in df.columns:
            df['updated'] = pd.to_datetime(df['updated'])
    
        # Add pagination information as DataFrame attributes
        df.attrs['total_elements'] = total_elements
        df.attrs['total_pages'] = total_pages
        df.attrs['current_page'] = current_page
        df.attrs['page_size'] = page_size
    
        # Print summary
        if all_pages and total_pages > 1:
            print(f"\nReturned {len(df)} tables", end="")
        else:
            print(f"\nFound {total_elements} tables", end="")
        
        if query:
            print(f" matching '{query}'", end="")
        if past_days:
            print(f" updated in the last {past_days} days", end="")
        
        if not all_pages and total_pages > 1:
            print(f" (Page {current_page} of {total_pages})")
            print(f"Use all_pages=True to retrieve all {total_elements} tables across all pages")
        else:
            print()
    
        return df

    def get_table_by_id(self, table_id: str) -> Dict:
        """Get table metadata by ID"""
        response = self._make_request('GET', f'/tables/{table_id}')
        return response.json()

    def get_table_metadata(self, 
                          table_id: str,
                          output_format: Optional[str] = None,
                          default_selection: bool = False) -> Dict:
        """
        Get detailed metadata about a specific table
        
        Args:
            table_id: Table identifier
            output_format: Output format (json-px or json-stat2)
            default_selection: Include default selection metadata
        """
        params = {
            'outputFormat': output_format,
            'defaultSelection': default_selection
        }
        response = self._make_request('GET', f'/tables/{table_id}/metadata', params=params)
        return response.json()
    
    def print_table_variables(self, 
                      table_id: str, 
                      max_values: Union[int, str] = 10,
                      variable_id: Optional[str] = None):
        """
        Print available variables and their values for a specific table

        Args:
            table_id: Table identifier
            max_values: Maximum number of values to show for each variable.
                    Use "*" to show all values. (default: 10)
            variable_id: Optional variable ID to show values for only that variable.
                    If None, shows all variables. (default: None)
        """
        metadata = self.get_table_metadata(table_id, output_format="json-stat2")
        print(f"\nTable: {metadata['label']}")

        show_all = max_values == "*"
        limit = None if show_all else int(max_values)

        # Get variables from JSON-stat2 format
        variables = []
        for var_id in metadata['id']:
            dimension = metadata['dimension'][var_id]
        
            # Create a variable entry with the same structure expected by the function
            var = {
                'id': var_id,
                'label': dimension.get('label', var_id),
                'values': []
            }
        
            # Extract values from category
            if 'category' in dimension and 'index' in dimension['category']:
                index = dimension['category']['index']
                labels = dimension['category'].get('label', {})
            
                for code in index.keys():
                    var['values'].append({
                        'code': code,
                        'label': labels.get(code, code)
                    })
        
            variables.append(var)

        if variable_id:
            # Find the specified variable
            var = next((v for v in variables if v['id'] == variable_id), None)
            if not var:
                raise ValueError(f"Variable '{variable_id}' not found in table")
        
            print(f"\nValues for {var['label']} ({var['id']}):")
            if show_all:
                print("All values:")
            else:
                print(f"First {limit} values:")
        
            values_to_show = var['values'] if show_all else var['values'][:limit]
            for val in values_to_show:
                print(f"  - {val['label']} (Code: {val['code']})")
        
            total_values = len(var['values'])
            if not show_all and total_values > limit:
                print(f"\n... and {total_values - limit} more values")
        else:
            # Show all variables
            print("\nAvailable variables:")
            for var in variables:
                print(f"\n{var['label']} ({var['id']}):")
        
                if show_all:
                    print("All values:")
                else:
                    print(f"First {limit} values:")
            
                values_to_show = var['values'] if show_all else var['values'][:limit]
                for val in values_to_show:
                    print(f"  - {val['label']} (Code: {val['code']})")
            
                total_values = len(var['values'])
                if not show_all and total_values > limit:
                    print(f"... and {total_values - limit} more values")
    
    def get_table_data(self,
                  table_id: str,
                  value_codes: Optional[Dict[str, List[str]]] = None,
                  code_lists: Optional[Dict[str, str]] = None,
                  output_format: Optional[OutputFormat] = None,
                  output_format_params: Optional[List[OutputFormatParam]] = None,
                  heading: Optional[List[str]] = None,
                  stub: Optional[List[str]] = None) -> Union[Dict, str, bytes, List[Union[Dict, str, bytes]]]:
        """
        Get table data with automatic chunking for large requests.

        Args:
            table_id: Table identifier
            value_codes: Dictionary mapping variable IDs to lists of value codes or selection expressions
            code_lists: Dictionary mapping variable IDs to codelist IDs
            output_format: Desired output format
            output_format_params: List of output format parameters
            heading: Variables to place in heading
            stub: Variables to place in stub
    
        Returns:
            Data in requested format. If chunking is needed, returns a list of chunks.
        """
        # Validate output format
        if output_format is not None and not isinstance(output_format, OutputFormat):
            raise ValueError(f"Invalid output format: {output_format}. Must be one of {[f.value for f in OutputFormat]}")

        if value_codes is None:
            value_codes = {}
    
        # Get JSON-stat metadata for calculations - this has all the information we need
        metadata_stat = self.get_table_metadata(table_id, output_format="json-stat2")

        # Use the extraction method
        var_meta_dict = self._extract_variable_metadata(metadata_stat)

        # Format for compatibility with _get_chunk_variable
        metadata_px = {
            'variables': [
                {
                    'id': var_id,
                    'type': meta['type'],
                    'elimination': meta['elimination']
                }
                for var_id, meta in var_meta_dict.items()
            ]
        }

        # Calculate cells without max validation since we'll handle chunking
        total_cells, cells_per_var = self._calculate_cells(
            table_id, 
            value_codes,
            validate_max_cells=False
        )

        print(f"\nRetrieving data from table {table_id}")
        print(f"Calculated data cells: {self._format_number(total_cells)}")

        # If total cells are within API limit, make single request
        if total_cells <= self.max_data_cells:
            return self._make_single_request(
                table_id=table_id,
                value_codes=value_codes,
                code_lists=code_lists,
                output_format=output_format,
                output_format_params=output_format_params,
                heading=heading,
                stub=stub
            )

        # Need to chunk - find best variable to chunk on using helper method
        chunk_var, _ = self._get_chunk_variable(
            cells_per_var=cells_per_var,
            metadata_px=metadata_px,
            value_codes=value_codes
        )

        # Prepare chunked requests using helper method
        chunks = self._prepare_chunks(
            table_id=table_id,
            chunk_var=chunk_var,
            value_codes=value_codes,
            metadata_stat=metadata_stat
        )

        # Process chunks
        num_chunks = len(chunks)
        print(f"Request will be split into {num_chunks} parts using variable '{chunk_var}'")
        print(f"Processing data in batches... ", end='', flush=True)

        results = []
        for i, chunk_codes in enumerate(chunks, 1):
            if i > 1:
                print(".", end='', flush=True)
            chunk_result = self._make_single_request(
                table_id=table_id,
                value_codes=chunk_codes,
                code_lists=code_lists,
                output_format=output_format,
                output_format_params=output_format_params,
                heading=heading,
                stub=stub
            )
            results.append(chunk_result)

        print(" Done!")
        return results
    
    def get_data_as_dataframe(self,
                     table_id: str,
                     value_codes: Optional[Dict[str, List[str]]] = None,
                     output_format_param: OutputFormatParam = OutputFormatParam.USE_TEXTS,
                     region_type: Optional[str] = None,
                     clean_colnames: bool = False) -> pd.DataFrame:
        """
        Get table data as a pandas DataFrame with smart formatting options.

        This method handles automatic chunking for large requests based on the API's data cell limit.

        Args:
            table_id: Table identifier
            value_codes: Dictionary mapping variable IDs to lists of value codes or selection expressions
                (wildcards, TOP/BOTTOM, RANGE, FROM/TO are supported)
            output_format_param: Format for values - use OutputFormatParam enum values:
                USE_TEXTS (default), USE_CODES, or USE_CODES_AND_TEXTS
            region_type: Filter for specific region types: "deso" or "regso" (or None for no filtering)
            clean_colnames: If True, standardizes column names (lowercase, underscores, ASCII)

        Returns:
            pandas.DataFrame: DataFrame with requested statistical data

        Raises:
            PxAPIError: On API errors (invalid table, too many cells, etc.)
            ValueError: On invalid parameters (e.g., invalid region_type)
        """
        try:
            data = self.get_table_data(
                table_id=table_id,
                value_codes=value_codes,
                output_format=OutputFormat.JSON_STAT2
            )
    
            if isinstance(data, list):
                frames = []
                for chunk in data:
                    chunk_df = self._process_jsonstat_to_df(
                        chunk,
                        output_format_param,
                        clean_colnames
                    )
                    frames.append(chunk_df)
                df = pd.concat(frames, ignore_index=True)
            else:
                df = self._process_jsonstat_to_df(
                    data,
                    output_format_param,
                    clean_colnames
                )
    
            # Apply region filtering if requested
            if region_type and 'region_code' in df.columns:
                region_type_lower = region_type.lower()
                if region_type_lower == 'deso':
                    # DeSO - Demographic Statistical Areas
                    # Format: XXXXAXXXX where A is a letter (A, B, C) indicating type
                    deso_mask = (df['region_code'].str.len() == 9) & \
                                (df['region_code'].str[4].isin(['A', 'B', 'C'])) & \
                                (df['region_code'].str[5:].str.match(r'\d{4}'))
                    df = df[deso_mask]
                elif region_type_lower == 'regso':
                    # RegSO - Regional Statistical Areas
                    # Format: XXXXRXXX where R indicates RegSO type
                    regso_mask = (df['region_code'].str[4] == 'R') & \
                                (df['region_code'].str[5:].str.match(r'\d{3}'))
                    df = df[regso_mask]
                else:
                    raise ValueError("Invalid region_type. Use 'deso', 'regso', or None")
    
            print(f"\nSuccessfully retrieved {len(df):,} rows of data")
            return df
    
        except Exception as e:
            print(f"\nError retrieving data: {str(e)}")
            raise
    
    def get_table_data_post(self,
                           table_id: str,
                           selection: Dict,
                           output_format: Optional[OutputFormat] = None,
                           output_format_params: Optional[List[OutputFormatParam]] = None) -> Union[Dict, str, bytes]:
        """
        Get table data using POST method
        
        Args:
            table_id: Table identifier
            selection: Selection criteria as dictionary
            output_format: Desired output format
            output_format_params: List of output format parameters
        """
        params = {}
        if output_format:
            params['outputFormat'] = output_format.value
        if output_format_params:
            params['outputFormatParams'] = ','.join(p.value for p in output_format_params)
            
        response = self._make_request('POST', f'/tables/{table_id}/data',
                                    params=params,
                                    json=selection)
        
        content_type = response.headers.get('content-type', '')
        if 'json' in content_type:
            return response.json()
        elif 'text' in content_type:
            return response.text
        else:
            return response.content
    
    def _format_number(self, num: int) -> str:
        """Format large numbers with commas and abbreviate if very large"""
        if num >= 1_000_000:
            return f"{num/1_000_000:.1f}M"
        elif num >= 1_000:
            return f"{num/1_000:.1f}K"
        return f"{num:,}"
    
    def _make_request(self, method: str, endpoint: str, **kwargs) -> requests.Response:
        """Make HTTP request to API with rate limiting"""
        url = f"{self.config.base_url.rstrip('/')}/{endpoint.lstrip('/')}"
        
        # Apply rate limiting before making request
        self.rate_limiter.wait_if_needed()
        
        if self.debug:
            print("\nDEBUG: Making API request")
            print(f"Method: {method}")
            print(f"URL: {url}")
            print("Parameters:", kwargs.get('params'))
        
        # Add language parameter if not already present
        if 'params' not in kwargs:
            kwargs['params'] = {}
        if 'lang' not in kwargs['params']:
            kwargs['params']['lang'] = self.config.language
            
        response = self.session.request(method, url, **kwargs)
    
        try:
            response.raise_for_status()
        except requests.exceptions.HTTPError as e:
            if self.debug:
                print("\nError Response:")
                print(f"Status code: {response.status_code}")
                try:
                    error_content = response.json()
                    print("Error details:", error_content.get('detail', str(e)))
                except:
                    print("Error content:", response.text)
            
            if response.headers.get('content-type') == 'application/problem+json':
                error_detail = response.json().get('detail', str(e))
                raise PxAPIError(f"API error: {error_detail}")
            raise PxAPIError(f"HTTP error: {str(e)}")
            
        return response

    def _parse_selection_expression(self, expr: str, values: List[Dict[str, str]]) -> List[str]:
        """
        Parse a selection expression and return matching value codes
    
        Args:
            expr: Selection expression (e.g., 'RANGE(2020,2023)', 'TOP(5)', etc.)
            values: List of value dictionaries with 'code' and 'label' keys
    
        Returns:
            List of matching value codes
        """
        # Extract just the codes from the values list
        all_codes = [v['code'] for v in values]
    
        # Delegate to the centralized implementation
        return SelectionMatcher.expand_expression(expr, all_codes)
    
    def _expand_selection_expressions(self, var_meta: Dict, codes: List[str]) -> List[str]:
        """
        Expand selection expressions into actual value codes
    
        Args:
            var_meta: Variable metadata dictionary containing values
            codes: List of code patterns or selection expressions
    
        Returns:
            List of expanded value codes
        """
        expanded_codes = []
        for code in codes:
            try:
                expanded_codes.extend(self._parse_selection_expression(code, var_meta['values']))
            except ValueError as e:
                raise PxAPIError(f"Error expanding selection for variable '{var_meta['id']}': {str(e)}")
        return list(dict.fromkeys(expanded_codes))  # Remove duplicates while preserving order
    
    def _count_values_for_code(self, var_meta: Dict, code: str) -> int:
        """
        Count how many values match a code pattern or selection expression
    
        Args:
            var_meta: Variable metadata dictionary containing values
            code: Code pattern or selection expression
    
        Returns:
            int: Number of matching values
        """
        try:
            matching_codes = self._parse_selection_expression(code, var_meta['values'])
            return len(matching_codes)
        except ValueError as e:
            raise PxAPIError(f"Error processing selection for variable '{var_meta['id']}': {str(e)}")
    
    def _matches_pattern(self, code: str, pattern: str) -> bool:
        """
        Check if a code matches a selection pattern
    
        Args:
            code: The value code to check
            pattern: The selection pattern
        
        Returns:
            bool: True if the code matches the pattern
        """
        # Create a list with just the one code and see if it's in the matches
        return code in SelectionMatcher.get_matching_codes([pattern], [code])
    
    def _extract_variable_metadata(self, metadata_stat: Dict) -> Dict[str, Dict]:
        """
        Extract variable metadata from JSON-stat2 response
    
        Args:
            metadata_stat: JSON-stat2 metadata dictionary
    
        Returns:
            Dictionary mapping variable IDs to their metadata
        """
        var_meta = {}
        for var_id in metadata_stat['id']:
            # Create a basic variable metadata entry
            dimension = metadata_stat['dimension'].get(var_id, {})
            var_meta[var_id] = {
                'id': var_id,
                'elimination': False,  # Default to False
            }
    
            # Try to determine variable type from role or naming conventions
            var_type = PxVariables.TYPE_REGULAR  # Default type
    
            if 'role' in metadata_stat:
                # Use the role attribute if available (standard JSON-stat2 approach)
                if var_id in metadata_stat['role'].get('time', []):
                    var_type = PxVariables.TYPE_TIME
                elif var_id in metadata_stat['role'].get('metric', []):
                    var_type = PxVariables.TYPE_CONTENTS
                elif var_id in metadata_stat['role'].get('geo', []):
                    var_type = PxVariables.TYPE_GEOGRAPHICAL
            else:
                # Fall back to variable name patterns if role isn't defined
                var_id_lower = var_id.lower()
            
                # Check if matches exact variable names
                if var_id == PxVariables.TIME:
                    var_type = PxVariables.TYPE_TIME
                elif var_id == PxVariables.CONTENTS:
                    var_type = PxVariables.TYPE_CONTENTS
                elif var_id == PxVariables.REGION:
                    var_type = PxVariables.TYPE_GEOGRAPHICAL
                # Check if contains any alternative names
                elif any(alt in var_id_lower for alt in PxVariables.TIME_ALTERNATIVES):
                    var_type = PxVariables.TYPE_TIME
                elif any(alt in var_id_lower for alt in PxVariables.CONTENTS_ALTERNATIVES):
                    var_type = PxVariables.TYPE_CONTENTS
                elif any(alt in var_id_lower for alt in PxVariables.REGION_ALTERNATIVES):
                    var_type = PxVariables.TYPE_GEOGRAPHICAL
    
            var_meta[var_id]['type'] = var_type
    
            # Try to extract elimination info from extension if available
            if dimension.get('extension') and 'elimination' in dimension['extension']:
                var_meta[var_id]['elimination'] = dimension['extension']['elimination']
        
        return var_meta
    
    def _calculate_cells(self, 
                   table_id: str,
                   value_codes: Optional[Dict[str, List[str]]] = None,
                   validate_max_cells: bool = True) -> Tuple[int, Dict[str, int]]:
        """Calculate the number of cells for a request"""
        if value_codes is None:
            value_codes = {}

        # Validate value_codes input format
        for var_id, codes in value_codes.items():
            if not isinstance(codes, list):
                raise PxAPIError(f"Values for variable '{var_id}' must be a list, got {type(codes)}")

        # Get JSON-stat metadata for size information
        metadata_stat = self.get_table_metadata(
            table_id,
            output_format="json-stat2"
        )

        # Extract variable metadata and size info
        var_meta = self._extract_variable_metadata(metadata_stat)
        var_sizes = dict(zip(metadata_stat['id'], metadata_stat['size']))
    
        # Check for invalid variables
        for var_id in value_codes:
            if var_id not in var_meta:
                raise PxAPIError(f"Invalid variable '{var_id}' not found in table")

        # Calculate cells for each variable
        values_per_var = {}
        total_cells = 1

        for var_id, full_size in var_sizes.items():
            if var_id in value_codes:
                # Get all codes for this variable
                dimension = metadata_stat['dimension'][var_id]
                all_codes = list(dimension['category']['index'].keys())
            
                # Count matching values using centralized matcher
                selected_size = 0
                for pattern in value_codes[var_id]:
                    selected_size += SelectionMatcher.count_matching_values(pattern, all_codes)
                
                    # Short-circuit for "*" wildcard
                    if pattern == "*":
                        selected_size = len(all_codes)
                        break
            
                if selected_size == 0:
                    raise PxAPIError(f"No matching values found for patterns in variable '{var_id}'")
            
                values_per_var[var_id] = selected_size
        
            else:
                # If no selection made, check elimination rules
                can_eliminate = var_meta[var_id].get('elimination', False)
                if can_eliminate:
                    values_per_var[var_id] = 1
                else:
                    values_per_var[var_id] = full_size
    
            total_cells *= values_per_var[var_id]

        # Validate against max cells if requested
        if validate_max_cells and total_cells > self.max_data_cells:
            formatted_total = self._format_number(total_cells)
            formatted_max = self._format_number(self.max_data_cells)
            raise PxAPIError(
                f"Request would return {formatted_total} cells which exceeds "
                f"maximum of {formatted_max}. Consider using filters or chunking."
            )

        return total_cells, values_per_var
    
    def _get_chunk_variable(self, 
                      cells_per_var: Dict[str, int], 
                      metadata_px: Dict,
                      value_codes: Dict[str, List[str]]) -> Tuple[str, int]:
        """
        Determine which variable to chunk on using smart prioritization rules.

        Args:
            cells_per_var: Dictionary of variable IDs to their cell counts
            metadata_px: Regular PX metadata containing elimination info
            value_codes: Dictionary of variable selections/filters
    
        Returns:
            Tuple[str, int]: Variable ID to chunk on and its cell count
        """
        # Get variable metadata for quick lookup
        var_meta = {v['id']: v for v in metadata_px['variables']}

        # Variables that should not be chunked:
        # 1. Time variables (type: TimeVariable)
        # 2. Content variables (type: ContentsVariable)
        # 3. Variables with TOP/BOTTOM filters
        # 4. Variables that were eliminated (not selected and can be eliminated)
        non_chunkable = set()

        for var_id, meta in var_meta.items():
            # Check variable type
            var_type = meta['type']
            if var_type in (PxVariables.TYPE_TIME, PxVariables.TYPE_CONTENTS):
                non_chunkable.add(var_id)
    
            # Check if variable was eliminated
            if var_id not in value_codes and meta.get('elimination', False):
                non_chunkable.add(var_id)
    
            # Check for TOP/BOTTOM filters if variable was selected
            if var_id in value_codes:
                if any(code.upper().startswith(('TOP', 'BOTTOM')) 
                    for code in value_codes[var_id]):
                    non_chunkable.add(var_id)

        # Remove non-chunkable variables from consideration
        chunk_candidates = {
            var_id: cells 
            for var_id, cells in cells_per_var.items()
            if var_id not in non_chunkable
        }

        if not chunk_candidates:
            raise PxAPIError(
                "Cannot chunk query - no suitable variables found for chunking. "
                "All variables are either time/content variables, using TOP/BOTTOM filters, "
                "or were eliminated."
            )

        # Prioritize variables for chunking:
        # 1. Prefer geographical variables (type: GeographicalVariable)
        # 2. Then by number of cells (highest first)
        # 3. Finally by variable ID as tiebreaker
        def chunk_priority(item):
            var_id, cells = item
            is_geo = var_meta[var_id]['type'] == PxVariables.TYPE_GEOGRAPHICAL
            return (is_geo, cells, var_id)

        best_var, cells = max(chunk_candidates.items(), key=chunk_priority)
        return best_var, cells

    def _prepare_chunks(self, 
                table_id: str,
                chunk_var: str,
                value_codes: Dict[str, List[str]],
                metadata_stat: Dict) -> List[Dict[str, List[str]]]:
        """
        Prepare chunked value codes for requests using JSON-stat2 metadata
    
        Args:
            table_id: Table identifier
            chunk_var: Variable ID to chunk on
            value_codes: Dictionary of variable selections
            metadata_stat: JSON-stat2 metadata
        
        Returns:
            List of value_codes dictionaries, one for each chunk
        """
        dimension = metadata_stat['dimension'][chunk_var]
        all_codes = list(dimension['category']['index'].keys())

        # Get values to chunk based on selection patterns
        if chunk_var in value_codes:
            # Use the centralized matcher
            all_values = SelectionMatcher.get_matching_codes(value_codes[chunk_var], all_codes)
        else:
            all_values = all_codes

        # Calculate chunk size based on total cells
        total_cells, _ = self._calculate_cells(
            table_id, 
            value_codes,
            validate_max_cells=False
        )
        cells_per_value = total_cells / len(all_values)
        values_per_chunk = max(1, int(self.max_data_cells / cells_per_value))

        # Create chunks
        chunks = []
        for i in range(0, len(all_values), values_per_chunk):
            chunk_values = all_values[i:i + values_per_chunk]
            chunk_codes = value_codes.copy()
            chunk_codes[chunk_var] = chunk_values
            chunks.append(chunk_codes)

        return chunks
    
    def _make_single_request(self,
                     table_id: str,
                     value_codes: Dict[str, List[str]],
                     code_lists: Optional[Dict[str, str]] = None,
                     output_format: Optional[OutputFormat] = None,
                     output_format_params: Optional[List[OutputFormatParam]] = None,
                     heading: Optional[List[str]] = None,
                     stub: Optional[List[str]] = None) -> Union[Dict, str, bytes]:
        """Make a single data request to the API with expanded selection expressions"""

        # Get table metadata in JSON-stat2 format
        metadata = self.get_table_metadata(table_id, output_format="json-stat2")
    
        # Create a variable map from JSON-stat2 format
        var_map = {}
        for var_id in metadata['id']:
            dimension = metadata['dimension'][var_id]
        
            # Create a variable entry with structure similar to original
            var = {
                'id': var_id,
                'values': []
            }
        
            # Extract values from category
            if 'category' in dimension and 'index' in dimension['category']:
                index = dimension['category']['index']
                labels = dimension['category'].get('label', {})
            
                for code in index.keys():
                    var['values'].append({
                        'code': code,
                        'label': labels.get(code, code)
                    })
        
            var_map[var_id] = var

        # Expand selection expressions in value_codes
        expanded_value_codes = {}
        if value_codes:
            for var, codes in value_codes.items():
                if var not in var_map:
                    raise PxAPIError(f"Invalid variable '{var}' not found in table")
                # For wildcard "*", just pass it directly
                if codes == ["*"]:
                    expanded_value_codes[var] = ["*"]
                else:
                    try:
                        expanded_value_codes[var] = self._expand_selection_expressions(var_map[var], codes)
                    except Exception as e:
                        # If expansion fails, just use the codes directly
                        if self.debug:
                            print(f"Warning: Could not expand selection for {var}: {e}")
                        expanded_value_codes[var] = codes

        # Build parameters with expanded selections
        params = {}
        if expanded_value_codes:
            for var, values in expanded_value_codes.items():
                params[f'valuecodes[{var}]'] = ','.join(values)

        # Add other parameters
        if code_lists:
            for var, codelist in code_lists.items():
                params[f'codelist[{var}]'] = codelist

        if output_format:
            params['outputFormat'] = output_format.value

        if output_format_params:
            params['outputFormatParams'] = ','.join(p.value for p in output_format_params)

        if heading:
            params['heading'] = ','.join(heading)

        if stub:
            params['stub'] = ','.join(stub)

        # Make the request
        response = self._make_request('GET', f'/tables/{table_id}/data', params=params)

        # Return appropriate type based on content-type
        content_type = response.headers.get('content-type', '')
        if 'json' in content_type:
            return response.json()
        elif 'text' in content_type:
            return response.text
        else:
            return response.content

    def _process_jsonstat_to_df(self, 
                           data: Dict,
                           output_format_param: OutputFormatParam,
                           clean_colnames: bool) -> pd.DataFrame:
        """Process JSON-stat data into DataFrame"""
    
        dimensions = data['dimension']
        values = data['value']

        # Find the ContentsVariable to get its values
        contents_var_id = next(dim_id for dim_id in data['id'] if PxVariables.CONTENTS in dim_id)
        contents_dim = dimensions[contents_var_id]
        contents_mapping = {
            code: label for code, label in contents_dim['category']['label'].items()
        }

        # Create a list to store rows
        rows = []

        # Get dimension sizes and names
        sizes = data['size']
        dim_names = data['id']

        # Create a helper function to convert flat index to dimensional indices
        def flat_to_indices(flat_idx, sizes):
            indices = []
            for size in reversed(sizes):
                indices.append(flat_idx % size)
                flat_idx //= size
            return list(reversed(indices))

        # Find the content dimension index
        content_dim_id = next(i for i, dim in enumerate(dim_names) if PxVariables.CONTENTS in dim)

        # Create a dictionary to temporarily store values
        temp_data = {}

        # Process each value
        for i, value in enumerate(values):
            # Get indices for each dimension
            dim_indices = flat_to_indices(i, sizes)

            # Create key for the row (excluding content dimension)
            row_key = []
            content_code = None

            for dim_name, idx in zip(dim_names, dim_indices):
                dim = dimensions[dim_name]
                code = list(dim['category']['index'].keys())[idx]

                if PxVariables.CONTENTS in dim_name:
                    content_code = code
                    continue

                if dim_name == PxVariables.REGION:
                    label = dim['category']['label'][code]
                    row_key.extend([code, label])
                else:
                    if dim_name == PxVariables.TIME:
                        # For time dimension, just use the code/label without combining
                        label = dim['category']['label'][code] if output_format_param == OutputFormatParam.USE_TEXTS else code
                        row_key.append(label)
                    else:
                        # For other dimensions
                        if output_format_param == OutputFormatParam.USE_CODES_AND_TEXTS:
                            label = dim['category']['label'][code]
                            row_key.append(f"{code} - {label}")
                        else:
                            label = dim['category']['label'][code] if output_format_param == OutputFormatParam.USE_TEXTS else code
                            row_key.append(label)

            row_key = tuple(row_key)

            # Initialize row if not exists
            if row_key not in temp_data:
                temp_data[row_key] = {}

            # Add value to appropriate content column
            temp_data[row_key][content_code] = value

        # Convert to DataFrame
        rows = []
        for row_key, row_values in temp_data.items():
            if PxVariables.REGION in dim_names:
                # Assuming region is always first when present
                region_code, region_label = row_key[:2]
                other_dims = row_key[2:]
                row_dict = {
                    'region_code': region_code,
                    'region': region_label
                }
            else:
                other_dims = row_key
                row_dict = {}

            # Add other dimensions
            other_dim_names = [dim for dim in dim_names if dim != PxVariables.REGION and PxVariables.CONTENTS not in dim]
            for dim_name, value in zip(other_dim_names, other_dims):
                clean_name = dimensions[dim_name]['label']
                row_dict[clean_name] = value

            # Add content values
            row_dict.update(row_values)
            rows.append(row_dict)

        df = pd.DataFrame(rows)

        # Rename content columns to their labels if using texts
        if output_format_param in [OutputFormatParam.USE_TEXTS, OutputFormatParam.USE_CODES_AND_TEXTS]:
            df = df.rename(columns=contents_mapping)

        # Clean column names if requested
        if clean_colnames:
            def clean_column_name(col):
                # Convert to lowercase
                col = col.lower()
                # Replace spaces with underscores
                col = col.replace(' ', '_')
                # Replace Swedish characters
                col = col.replace('å', 'a').replace('ä', 'a').replace('ö', 'o')
                # Remove special characters (keeping underscores)
                col = ''.join(c for c in col if c.isalnum() or c == '_')
                # Remove multiple consecutive underscores
                col = '_'.join(filter(None, col.split('_')))
                return col

            df.columns = [clean_column_name(col) for col in df.columns]

        return df
    
                                                    