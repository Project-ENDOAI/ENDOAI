# ENDOAI

ENDOAI is an advanced AI-based toolkit for the **diagnosis and surgical management of endometriosis**. This repository contains the **public components** of ENDOAI, including preprocessing scripts, basic segmentation pipelines, and educational materials.

## Public vs. Private Components

### Public Components (This Repository)
- Preprocessing scripts for MRI and ultrasound data.
- Basic segmentation pipelines (e.g., 3D U-Net implementation).
- Documentation, tutorials, and sample datasets.

### Private Components (Proprietary)
- Fine-tuned AI models (e.g., Swin-UNet, ensemble models).
- Proprietary datasets and annotations.
- Advanced risk mapping algorithms.
- Full-featured UI for surgical planning.
- APIs for premium features (e.g., lesion detection, risk mapping).

For access to private components, please contact us for licensing options.

## Project Structure

```
endoai/
├── core/                  # Core utilities and shared logic
├── data/                  # Dataset storage and data management
│   ├── raw/               # Raw MRI, ultrasound, surgical video data
│   ├── processed/         # Preprocessed data (e.g., normalized, resized)
├── models/                # Model definitions and checkpoints
├── notebooks/             # Jupyter notebooks for experimentation
├── pipelines/             # End-to-end ML/AI pipelines
├── src/                   # Source code for each workflow/module
│   ├── preoperative/      # Preoperative planning scripts
│   ├── intraoperative/    # Real-time guidance scripts
│   ├── decision_support/  # Decision support scripts
│   ├── reporting/         # Reporting tools
├── tests/                 # Unit and integration tests
├── install/               # Installation and environment setup scripts
├── requirements.txt       # Python dependencies
├── README.md              # Project documentation (this file)
└── LICENSE
```

## Getting Started

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd ENDOAI
   ```

2. Set up the environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Explore the project:
   - Use `notebooks/` for experimentation.
   - Refer to `src/` for modular scripts.
   - See `install/` for setup and environment scripts.

## Documentation

ENDOAI provides modular documentation for each major component:

- [core/README.md](core/README.md): Core utilities and shared logic.
- [data/README.md](data/README.md): Data management, raw and processed data.
- [models/README.md](models/README.md): Model definitions and guidelines.
- [notebooks/README.md](notebooks/README.md): Jupyter notebooks for experimentation.
- [pipelines/README.md](pipelines/README.md): End-to-end ML/AI pipelines.
- [src/README.md](src/README.md): Main source code modules.
- [src/preoperative/README.md](src/preoperative/README.md): Preoperative planning and segmentation.
- [src/intraoperative/README.md](src/intraoperative/README.md): Intraoperative (real-time) guidance.
- [src/decision_support/README.md](src/decision_support/README.md): Decision support algorithms.
- [src/reporting/README.md](src/reporting/README.md): Reporting and visualization tools.
- [tests/README.md](../tests/README.md): Unit and integration tests.
- [install/README.md](../install/README.md): Installation and environment setup.
- [validation/README.md](../validation/README.md): Model evaluation and validation.

For detailed usage, see the documentation in each folder and the [COPILOT.md](../../COPILOT.md) for coding standards.

## Contributing

We welcome contributions to the public components of ENDOAI. Please submit a pull request or open an issue for any suggestions or improvements.

## Coding Standards & AI Assistant Configuration

- See [COPILOT.md](../../COPILOT.md) for coding standards and preferences.
- See [.copilot/](../../.copilot/) for Copilot and AI assistant configuration and context files.

## License

This repository is licensed under the **Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)**. See the `LICENSE` file for details.
