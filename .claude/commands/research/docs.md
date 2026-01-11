---
description: Generate or update documentation for code, APIs, or features
model: sonnet
---

# Generate Documentation

You are tasked with generating or updating technical documentation based on the codebase.

## Output Locations

Documentation goes to standard locations:
```
docs/                    # Public documentation
├── api/                 # API reference
├── guides/              # How-to guides
└── architecture/        # System design docs

.claude/artifacts/docs/            # Internal conventions (from /extract_conventions)
```

## Usage

### Document a specific file/module
```
/docs src/services/auth.ts
/docs src/api/
```

### Generate API documentation
```
/docs api
/docs api src/routes/
```

### Update README
```
/docs readme
```

### Document a feature
```
/docs feature "user authentication flow"
```

## Process

### For Code Documentation (`/docs path/to/file`)

1. **Read the target file(s)**

2. **Analyze**:
   - Public interfaces (exported functions, classes, types)
   - Function signatures and return types
   - Dependencies and imports
   - Error handling patterns

3. **Generate documentation**:
   - For TypeScript/JavaScript: JSDoc comments
   - For Python: Docstrings (Google or NumPy style, match existing)
   - For Go: GoDoc comments
   - For Java: Javadoc

4. **Output options**:
   - **Inline**: Add/update comments in the file itself
   - **External**: Create `docs/api/[module-name].md`

   Ask user preference if unclear.

### For API Documentation (`/docs api`)

1. **Find API definitions**:
   - OpenAPI/Swagger specs
   - Route definitions
   - Controller/handler files
   - GraphQL schemas

2. **Generate reference docs**:
   ```markdown
   # API Reference

   ## Authentication
   All endpoints require `Authorization: Bearer <token>` header.

   ## Endpoints

   ### POST /api/users
   Create a new user.

   **Request Body**:
   ```json
   {
     "email": "string (required)",
     "name": "string (required)",
     "role": "string (optional, default: 'user')"
   }
   ```

   **Response**: `201 Created`
   ```json
   {
     "id": "string",
     "email": "string",
     "name": "string",
     "createdAt": "ISO 8601 date"
   }
   ```

   **Errors**:
   - `400 Bad Request` - Invalid input
   - `409 Conflict` - Email already exists
   ```

3. **Save to**: `docs/api/reference.md` or update existing

### For README (`/docs readme`)

1. **Analyze project**:
   - Package.json / pyproject.toml / go.mod
   - Existing README content
   - Directory structure
   - Available scripts/commands

2. **Generate/update sections**:
   ```markdown
   # Project Name

   Brief description from package metadata.

   ## Quick Start

   ```bash
   # Installation
   [detected from package manager]

   # Run
   [detected from scripts]
   ```

   ## Development

   ### Prerequisites
   - [Runtime] version X.Y
   - [Dependencies]

   ### Setup
   [Step by step]

   ### Commands
   | Command | Description |
   |---------|-------------|
   | `npm run dev` | Start dev server |
   | `npm test` | Run tests |

   ## Project Structure
   ```
   src/
   ├── api/        # HTTP handlers
   ├── services/   # Business logic
   └── ...
   ```

   ## Contributing
   [Link to CONTRIBUTING.md or brief guidelines]
   ```

3. **Preserve existing content**: Don't overwrite custom sections

### For Feature Documentation (`/docs feature "description"`)

1. **Research the feature in codebase**:
   - Find all related files
   - Trace the flow (entry point to completion)
   - Identify configuration options

2. **Generate guide**:
   ```markdown
   # Feature: [Name]

   ## Overview
   [What it does, why it exists]

   ## How It Works
   [High-level flow with diagram if helpful]

   ## Usage

   ### Basic Usage
   ```[language]
   [Code example]
   ```

   ### Configuration
   | Option | Type | Default | Description |
   |--------|------|---------|-------------|
   | ... | ... | ... | ... |

   ### Advanced Usage
   [More complex scenarios]

   ## Related
   - [Link to related features]
   - [Link to API docs]
   ```

3. **Save to**: `docs/guides/[feature-name].md`

## Style Guidelines

1. **Match existing style**: Check `docs/` for tone and format
2. **Be concise**: Developers skim documentation
3. **Include examples**: Every feature needs a code example
4. **Keep current**: Note what version/commit this documents

## Validation

After generating docs:

1. **Check for broken links**
2. **Verify code examples compile/run**
3. **Ensure all public APIs are documented**

## Output

```
## Documentation Updated

Created/Modified:
- docs/api/reference.md (15 endpoints documented)
- docs/guides/authentication.md (new)

Coverage:
- API endpoints: 15/15 (100%)
- Public functions: 42/50 (84%)

Suggestions:
- 8 public functions in src/utils/ lack documentation
- Consider adding sequence diagram for auth flow
```
