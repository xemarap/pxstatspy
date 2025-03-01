import requests
import time
import pytest
from unittest.mock import Mock, patch
import json
import pandas as pd
from datetime import datetime
from pxstatspy import PxAPI, PxAPIConfig, OutputFormat, OutputFormatParam, PxAPIError

# Fixtures for common test setup
@pytest.fixture
def api_client():
    """Create a PxAPI client with test configuration"""
    config = PxAPIConfig(
        base_url="https://api.test.com/v2",
        language="en"
    )
    return PxAPI(config)

@pytest.fixture
def mock_response():
    """Create a mock response object"""
    mock = Mock()
    mock.status_code = 200
    mock.headers = {"content-type": "application/json"}
    return mock

@pytest.fixture
def mock_metadata():
    """Sample metadata in PX format"""
    return {
        "id": "TAB123",
        "label": "Test Table",
        "variables": [
            {
                "id": "Tid",
                "label": "Year",
                "type": "TimeVariable",
                "values": [
                    {"code": "2022", "label": "2022"},
                    {"code": "2023", "label": "2023"}
                ]
            },
            {
                "id": "Region",
                "label": "Region",
                "type": "RegularVariable",
                "values": [
                    {"code": "00", "label": "All regions"},
                    {"code": "01", "label": "Stockholm"}
                ]
            }
        ]
    }

@pytest.fixture
def mock_metadata_jsonstat():
    """Sample metadata in JSON-stat2 format"""
    return {
        "version": "2.0",
        "class": "dataset",
        "id": ["Tid", "Region"],
        "size": [2, 2],
        "dimension": {
            "Tid": {
                "label": "Year",
                "category": {
                    "index": {"2022": 0, "2023": 1},
                    "label": {"2022": "2022", "2023": "2023"}
                }
            },
            "Region": {
                "label": "Region",
                "category": {
                    "index": {"00": 0, "01": 1},
                    "label": {"00": "All regions", "01": "Stockholm"}
                }
            }
        }
    }

@pytest.fixture
def large_metadata():
    """Sample metadata for a large table in PX format"""
    return {
        "id": "TAB999",
        "label": "Large Test Table",
        "variables": [
            {
                "id": "Region",
                "label": "Region",
                "type": "GeographicalVariable",
                "values": [{"code": f"{i:02d}", "label": f"Region {i}"} for i in range(30)]
            },
            {
                "id": "Tid",
                "label": "Year",
                "type": "TimeVariable",
                "values": [{"code": str(year), "label": str(year)} for year in range(2020, 2025)]
            },
            {
                "id": "ContentsCode",
                "label": "Contents",
                "type": "ContentsVariable",
                "values": [{"code": f"CODE{i}", "label": f"Metric {i}"} for i in range(5)]
            }
        ]
    }

@pytest.fixture
def large_metadata_jsonstat():
    """Sample metadata for a large table in JSON-stat2 format"""
    return {
        "version": "2.0",
        "class": "dataset",
        "id": ["Region", "Tid", "ContentsCode"],
        "size": [30, 5, 5],
        "dimension": {
            "Region": {
                "label": "Region",
                "category": {
                    "index": {f"{i:02d}": i for i in range(30)},
                    "label": {f"{i:02d}": f"Region {i}" for i in range(30)}
                }
            },
            "Tid": {
                "label": "Year",
                "category": {
                    "index": {str(year): i for i, year in enumerate(range(2020, 2025))},
                    "label": {str(year): str(year) for year in range(2020, 2025)}
                }
            },
            "ContentsCode": {
                "label": "Contents",
                "category": {
                    "index": {f"CODE{i}": i for i in range(5)},
                    "label": {f"CODE{i}": f"Metric {i}" for i in range(5)}
                }
            }
        }
    }

@pytest.fixture
def sample_jsonstat_data():
    """Sample JSON-stat data for testing"""
    return {
        "version": "2.0",
        "class": "dataset",
        "id": ["Region", "Tid", "ContentsCode"],
        "size": [2, 2, 1],
        "dimension": {
            "Region": {
                "label": "Region",
                "category": {
                    "index": {"00": 0, "01": 1},
                    "label": {"00": "All regions", "01": "Stockholm"}
                }
            },
            "Tid": {
                "label": "Year",
                "category": {
                    "index": {"2022": 0, "2023": 1},
                    "label": {"2022": "2022", "2023": "2023"}
                }
            },
            "ContentsCode": {
                "label": "Contents",
                "category": {
                    "index": {"BE0101N1": 0},
                    "label": {"BE0101N1": "Population"}
                }
            }
        },
        "value": [100, 200, 300, 400]
    }

# Test initialization and configuration
def test_init_api_client(api_client):
    """Test API client initialization"""
    assert api_client.config.base_url == "https://api.test.com/v2"
    assert api_client.config.language == "en"
    assert api_client.max_data_cells == 150000

def test_init_with_api_key():
    """Test API client initialization with API key"""
    config = PxAPIConfig(
        base_url="https://api.test.com/v2",
        api_key="test-key"
    )
    client = PxAPI(config)
    assert "Authorization" in client.session.headers
    assert client.session.headers["Authorization"] == "Bearer test-key"

# Test data cell calculations
def test_calculate_data_cells(api_client, mock_metadata, mock_metadata_jsonstat):
    """Test calculation of data cells"""
    def mock_get_metadata(table_id, output_format=None):
        return mock_metadata_jsonstat if output_format == "json-stat2" else mock_metadata
    
    with patch.object(api_client, 'get_table_metadata', side_effect=mock_get_metadata):
        # Test with all values
        total_cells, cells_per_var = api_client._calculate_cells("TAB123", None)
        assert total_cells == 4  # 2 years * 2 regions
        assert cells_per_var == {'Tid': 2, 'Region': 2}

        # Test with specific values
        total_cells, cells_per_var = api_client._calculate_cells(
            "TAB123",
            {"Tid": ["2022"], "Region": ["00"]}
        )
        assert total_cells == 1
        assert cells_per_var == {'Tid': 1, 'Region': 1}

        # Test with wildcards
        total_cells, cells_per_var = api_client._calculate_cells(
            "TAB123",
            {"Tid": ["*"], "Region": ["01"]}
        )
        assert total_cells == 2
        assert cells_per_var == {'Tid': 2, 'Region': 1}

# Test API request handling
def test_make_request_success(api_client, mock_response):
    """Test successful API request"""
    mock_response.json.return_value = {"data": "test"}
    
    with patch('requests.Session.request', return_value=mock_response):
        response = api_client._make_request('GET', '/test')
        assert response.json() == {"data": "test"}

def test_make_request_error(api_client):
    """Test API request error handling"""
    error_response = Mock()
    error_response.status_code = 404
    error_response.json.return_value = {
        "title": "Not Found",
        "detail": "Resource not found"
    }
    http_error = requests.exceptions.HTTPError("404 Client Error: Not Found for url: /test")
    error_response.raise_for_status.side_effect = http_error
    
    with patch('requests.Session.request', return_value=error_response):
        with pytest.raises(PxAPIError) as exc_info:
            api_client._make_request('GET', '/test')
        # We're just checking that we get a PxAPIError with the HTTP error info
        assert "HTTP error: 404" in str(exc_info.value)

# Test DataFrame conversion
def test_get_data_as_dataframe(api_client, sample_jsonstat_data):
    """Test conversion of JSON-stat data to DataFrame"""
    with patch.object(api_client, 'get_table_data', return_value=sample_jsonstat_data):
        df = api_client.get_data_as_dataframe(
            table_id="TAB123",
            output_format_param=OutputFormatParam.USE_TEXTS
        )
        
        # Check DataFrame structure
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 4
        assert "region" in df.columns
        assert "Year" in df.columns
        assert "Population" in df.columns

def test_get_data_as_dataframe_region_filter(api_client, sample_jsonstat_data):
    """Test DataFrame conversion with region filtering"""
    # Modify sample data to include DeSO regions
    sample_jsonstat_data["dimension"]["Region"]["category"]["index"] = {
        "0114A0001": 0,
        "0114B0002": 1
    }
    sample_jsonstat_data["dimension"]["Region"]["category"]["label"] = {
        "0114A0001": "DeSO 1",
        "0114B0002": "DeSO 2"
    }
    
    with patch.object(api_client, 'get_table_data', return_value=sample_jsonstat_data):
        df = api_client.get_data_as_dataframe(
            table_id="TAB123",
            region_type="deso"
        )
        assert len(df) == 4
        assert all(df["region_code"].str.len() == 9)
        assert all(df["region_code"].str[4].isin(["A", "B"]))


# Test navigation functionality
def test_navigation_explorer(api_client, mock_response):
    """Test navigation explorer functionality"""
    mock_response.json.return_value = {
        "id": "BE",
        "label": "Population",
        "folderContents": [
            {
                "id": "BE01",
                "label": "Population Statistics",
                "type": "FolderInformation"
            }
        ]
    }
    
    with patch('requests.Session.request', return_value=mock_response):
        contents = api_client.navigator.navigate_to("BE")
        assert len(contents["folders"]) == 1
        assert contents["folders"][0].id == "BE01"

@pytest.fixture
def large_table_metadata():
    """Sample metadata for a large table"""
    return {
        "id": "TAB999",
        "label": "Large Test Table",
        "variables": [
            {
                "id": "Tid",
                "label": "Year",
                "type": "TimeVariable",
                "values": [{"code": str(year), "label": str(year)} for year in range(2000, 2024)]
            },
            {
                "id": "Region",
                "label": "Region",
                "type": "RegularVariable",
                "values": [{"code": f"{i:02d}", "label": f"Region {i}"} for i in range(30)]
            },
            {
                "id": "ContentsCode",
                "label": "Contents",
                "type": "ContentsVariable",
                "values": [{"code": f"CODE{i}", "label": f"Metric {i}"} for i in range(5)]
            }
        ]
    }

# Rate Limiting Tests
def test_rate_limiter_basic_functionality(api_client):
    """Test that rate limiter enforces call limits"""
    # Configure rate limiter for testing (3 calls per 1 second window)
    api_client.rate_limiter.max_calls = 3
    api_client.rate_limiter.time_window = 1
    
    # Mock successful response
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"data": "test"}
    
    with patch('requests.Session.request', return_value=mock_response):
        start_time = time.time()
        
        # Make calls up to the limit
        for _ in range(3):
            api_client._make_request('GET', '/test')
            
        # Next call should be delayed
        api_client._make_request('GET', '/test')
        elapsed_time = time.time() - start_time
        
        # Should have waited close to 1 second
        assert elapsed_time >= 0.9, "Rate limiter didn't enforce waiting period"

def test_rate_limiter_sliding_window(api_client):
    """Test that rate limiter properly implements sliding window"""
    api_client.rate_limiter.max_calls = 2
    api_client.rate_limiter.time_window = 0.5
    
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"data": "test"}
    
    with patch('requests.Session.request', return_value=mock_response):
        # Make initial calls
        api_client._make_request('GET', '/test')
        api_client._make_request('GET', '/test')
        
        # Wait for half the window
        time.sleep(0.25)
        
        start_time = time.time()
        # This call should be allowed without full delay
        api_client._make_request('GET', '/test')
        elapsed_time = time.time() - start_time
        
        assert elapsed_time < 0.5, "Sliding window not properly implemented"

# Error Handling Tests
def test_malformed_value_codes(api_client, mock_metadata, mock_metadata_jsonstat):
    """Test handling of malformed value codes"""
    def mock_get_metadata(table_id, output_format=None):
        if output_format == "json-stat2":
            # Return metadata without the nonexistent variable
            stat_metadata = mock_metadata_jsonstat.copy()
            stat_metadata['id'] = ['Tid', 'Region']  # Only existing variables
            return stat_metadata
        return mock_metadata

    with patch.object(api_client, 'get_table_metadata', side_effect=mock_get_metadata):
        # Test with invalid variable
        with pytest.raises(PxAPIError, match="Invalid variable"):
            api_client._calculate_cells(
                "TAB123",
                {"NonexistentVar": ["2022"]}
            )
        
        # Test with invalid value format
        with pytest.raises(PxAPIError, match="must be a list"):
            api_client._calculate_cells(
                "TAB123",
                {"Tid": "2022"}  # Not in list format
            )

def test_invalid_output_format(api_client):
    """Test handling of invalid output format"""
    # Mock the metadata request to avoid network calls
    mock_metadata = {
        "variables": [
            {"id": "Tid", "values": [{"code": "2023"}]}
        ]
    }
    
    with patch.object(api_client, 'get_table_metadata', return_value=mock_metadata):
        with pytest.raises(ValueError) as exc_info:
            api_client.get_table_data(
                "TAB123",
                output_format="invalid_format"
            )
        assert "invalid output format" in str(exc_info.value).lower()

def test_excessive_data_request(api_client, large_metadata, large_metadata_jsonstat):
    """Test handling of requests that exceed max cell limit"""
    def mock_get_metadata(table_id, output_format=None):
        # Return metadata showing total cells will exceed limit
        if output_format == "json-stat2":
            stat_metadata = {
                "version": "2.0",
                "class": "dataset",
                "id": ["Region", "Tid", "ContentsCode"],
                "size": [100, 100, 100],  # 1,000,000 cells total
                "dimension": {
                    "Region": {
                        "label": "Region",
                        "category": {
                            "index": {str(i): i for i in range(100)},
                            "label": {str(i): f"Region {i}" for i in range(100)}
                        }
                    },
                    "Tid": {
                        "label": "Year",
                        "category": {
                            "index": {str(i): i for i in range(100)},
                            "label": {str(i): str(i) for i in range(100)}
                        }
                    },
                    "ContentsCode": {
                        "label": "Contents",
                        "category": {
                            "index": {f"CODE{i}": i for i in range(100)},
                            "label": {f"CODE{i}": f"Metric {i}" for i in range(100)}
                        }
                    }
                }
            }
            return stat_metadata
            
        # Return regular metadata
        return large_metadata

    with patch.object(api_client, 'get_table_metadata', side_effect=mock_get_metadata):
        api_client.max_data_cells = 1000  # Set low limit
        with pytest.raises(PxAPIError, match="exceeds maximum"):
            api_client._calculate_cells(
                "TAB999",
                {
                    "Tid": ["*"],
                    "Region": ["*"],
                    "ContentsCode": ["*"]
                }
            )


# Test chunking functionality
def test_prepare_chunks(api_client, mock_metadata_jsonstat):
    """Test preparation of chunked requests"""
    with patch.object(api_client, '_calculate_cells', return_value=(4, {'Tid': 2, 'Region': 2})):
        chunks = api_client._prepare_chunks(
            table_id="TAB123",
            chunk_var="Tid",
            value_codes={"Tid": ["*"], "Region": ["00"]},
            chunk_size=1,
            metadata_stat=mock_metadata_jsonstat
        )
        
        assert len(chunks) == 2  # Should split into 2 chunks for 2 time periods
        assert all("Tid" in chunk for chunk in chunks)
        assert all(len(chunk["Tid"]) == 1 for chunk in chunks)
        assert all("Region" in chunk and chunk["Region"] == ["00"] for chunk in chunks)

def test_chunk_combination(api_client, large_metadata, large_metadata_jsonstat):
    """Test combining results from chunked requests"""
    def mock_get_metadata(table_id, output_format=None):
        return large_metadata_jsonstat if output_format == "json-stat2" else large_metadata
    
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.headers = {"content-type": "application/json"}
    
    def mock_json_response(region_codes):
        return {
            "version": "2.0",
            "class": "dataset",
            "dimension": {
                "Region": {
                    "label": "Region",
                    "category": {
                        "index": {code: idx for idx, code in enumerate(region_codes)},
                        "label": {code: f"Region {code}" for code in region_codes}
                    }
                },
                "Tid": {
                    "label": "Year",
                    "category": {
                        "index": {"2023": 0},
                        "label": {"2023": "2023"}
                    }
                },
                "ContentsCode": {
                    "label": "Contents",
                    "category": {
                        "index": {"CODE1": 0},
                        "label": {"CODE1": "Metric 1"}
                    }
                }
            },
            "id": ["Region", "Tid", "ContentsCode"],
            "size": [len(region_codes), 1, 1],
            "value": [int(code) * 100 for code in region_codes]
        }
    
    region_codes = ["01", "02", "03", "04", "05", "06"]
    mock_responses = []
    for i in range(0, len(region_codes), 2):
        chunk_codes = region_codes[i:i + 2]
        mock_responses.append(mock_json_response(chunk_codes))
    
    mock_response.json.side_effect = mock_responses
    
    with patch.object(api_client, 'get_table_metadata', side_effect=mock_get_metadata):
        with patch('requests.Session.request', return_value=mock_response):
            # Mock _calculate_cells to force chunking
            with patch.object(api_client, '_calculate_cells', return_value=(1000, {'Region': len(region_codes), 'Tid': 1, 'ContentsCode': 1})):
                # Set the max_data_cells to a value that forces multiple chunks
                api_client.max_data_cells = 200
                
                result = api_client.get_table_data(
                    table_id="TAB999",
                    value_codes={
                        "Region": region_codes,
                        "Tid": ["2023"],
                        "ContentsCode": ["CODE1"]
                    },
                    output_format=OutputFormat.JSON_STAT2
                )
                
                assert isinstance(result, list), "Chunked request should return list"
                assert len(result) == 3, "Should have 3 chunks of 2 regions each"
                
                # Verify each chunk has the correct regions
                regions_seen = []
                for chunk in result:
                    chunk_regions = list(chunk["dimension"]["Region"]["category"]["index"].keys())
                    assert len(chunk_regions) == 2, "Each chunk should have 2 regions"
                    regions_seen.extend(chunk_regions)
                
                # Verify we got all regions in the correct order
                assert regions_seen == region_codes, "Missing or extra regions in results"

# Integration tests (disabled by default)
@pytest.mark.integration
def test_live_api_connection():
    """Test connection to live API (integration test)"""
    config = PxAPIConfig(
        base_url="https://api.scb.se/OV0104/v2beta/api/v2",  # Updated URL for v2 API
        language="en"
    )
    client = PxAPI(config)
    
    # Test basic navigation
    root = client.navigator.get_root()
    assert root is not None
    
    # Test table search
    tables = client.find_tables(query="population", display=False)
    assert isinstance(tables, dict)
    assert "tables" in tables

if __name__ == "__main__":
    pytest.main([__file__])