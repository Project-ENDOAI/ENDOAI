# ENDOAI Project

This is the root documentation for the ENDOAI project. See `endoai/README.md` for details on project structure, usage, and contribution guidelines.

- Main code and documentation: [endoai/README.md](endoai/README.md)
- Copilot/AI assistant configuration: [.copilot/](.copilot/)
- Scripts and automation: [scripts/](scripts/)

## Continuous Integration

This project uses GitHub Actions for CI/CD. All pushes and pull requests to `main` are automatically tested and linted.

## Docker

A `Dockerfile` is provided for reproducible environments. Build and run with:

```bash
docker build -t endoai .
docker run -it endoai
```

## Logging

A standard logging utility is provided in `endoai/core/logger.py`. Use `get_logger(__name__)` in your modules.

## Testing

Unit and integration tests are in `endoai/tests/`. Run all tests with:

```bash
pytest endoai/tests/
```

## Documentation

To build documentation (if Sphinx is configured):

```bash
cd docs
make html
```
