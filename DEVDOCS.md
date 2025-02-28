# Development documentation

## How get_table_data() Works with Helper Functions

The `get_table_data()` method retrieves statistical data from the PxAPI-2 while intelligently managing large data requests through chunking. Here's a step-by-step explanation of how it works:

### Initial Setup and Validation

1. First, it validates the requested output format and sets default values for missing parameters.
2. It sets the chunk size (defaulting to the maximum allowed data cells if not specified).

### Metadata Retrieval and Processing

3. The method makes a single API call to get metadata in JSON-stat2 format.
4. It then transforms this metadata into the format needed by the helper functions, extracting:

- Variable types (Time, Content, Geographical, or Regular)
- Elimination information (whether variables can be omitted)

### Data Size Calculation

5. Using the `_calculate_cells()` helper method, it calculates:

- The total number of data cells to be retrieved
- The number of cells for each variable

6. This calculation considers selection patterns like wildcards and range expressions.

### Single Request vs. Chunking Decision

7. If the total cells is within the chunk size limit, it makes a single request using `_make_single_request()`.
8. If the data is too large, it initiates the chunking process:

### Smart Chunking Process

9. It calls `_get_chunk_variable()` to determine the optimal variable to chunk on:

- Avoids chunking on time or content variables
- Prefers geographical variables when possible
- Takes into account variable size and selection patterns

10. It then uses `_prepare_chunks()` to divide the request into manageable pieces:

- Splits the selected values of the chosen variable into groups
- Creates a separate request for each group of values
- Ensures chunks stay within the allowed size limits

### Executing Chunked Requests

11. For each chunk, it calls `_make_single_request()` to fetch the data.
12. It provides visual feedback with progress indicators during the process.
13. The results from all chunks are collected into a list.

### Final Output

14. For single requests, it returns the data directly in the requested format.
15. For chunked requests, it returns a list of data chunks in the requested format.

The method optimizes API usage by:

- Making the minimum necessary API calls
- Handling large data sets efficiently
- Providing helpful progress information
- Using smart prioritization for chunking decisions
