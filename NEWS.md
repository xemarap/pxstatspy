
# pxstatspy v1.0.1

Fix: Resolve IndexError in RateLimiter.wait_if_needed()
   
   - Add safety check before accessing self.calls[0]
   - Prevents crash when deque becomes empty after cleanup
   - Fixes initialization issue with Norwegian SSB API endpoint

# pxstatspy v1.0.0

- Updated base URL to the official SCB production endpoint in documentation: https://statistikdatabasen.scb.se/api/v2
- Removed all "beta" references from documentation and code

# pxstatspy v0.2.0

Bump version to 0.2.0 due to removal of Navigation API

Removed Navigation API functionality:

   - Removed NavigationItem dataclass and NavigationExplorer class
   - Removed navigation-related methods from PxAPI class
   - Updated imports and exports in __init__.py
   - Removed navigation tests
   - Updated README to reflect removed functionality
   - Updated tutorial

   This change aligns with SCB's removal of Navigation API from PxWebAPI 2.0

# pxstatspy v0.1.1

- Improve navigation display with icons and better ID formatting

# pxstatspy v0.1.0

- Initial release.