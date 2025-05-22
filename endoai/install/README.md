# install

This directory contains installation and environment setup scripts for the ENDOAI project.

## Purpose

- Automate the installation of Python dependencies and system requirements.
- Provide scripts for verifying and troubleshooting the development environment.

## Structure

- `install_modules.py` — Installs all required Python modules.
- `check_install.py` — Verifies installation and environment setup.

## Usage

To install all dependencies:
```bash
python install/install_modules.py
```

To check your environment:
```bash
python install/check_install.py
```

## Guidelines

- Update these scripts when adding or removing dependencies.
- Document any manual installation steps required for your platform.
- Ensure compatibility with Python 3.8+.

## See Also

- [../requirements.txt](../requirements.txt) — List of Python dependencies.
- [../../README.md](../../README.md) — Project overview.
