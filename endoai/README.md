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
├── data/                   # Dataset storage
│   ├── raw/               # Raw MRI, ultrasound, surgical video data
│   ├── processed/         # Preprocessed data (e.g., normalized, resized)
├── models/                 # Saved models and checkpoints
├── notebooks/              # Jupyter notebooks for experimentation
├── src/                    # Source code for each module
│   ├── preoperative/       # Preoperative planning scripts
│   ├── intraoperative/     # Real-time guidance scripts
│   ├── decision_support/   # Decision support scripts
│   ├── reporting/          # Reporting tools
├── tests/                  # Unit tests for all modules
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
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

## Contributing

We welcome contributions to the public components of ENDOAI. Please submit a pull request or open an issue for any suggestions or improvements.

## License

This repository is licensed under the **Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)**. See the `LICENSE` file for details.
