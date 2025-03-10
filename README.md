# BS Meter

BS-Meter is an AI-powered text authenticity analyzer that leverages machine learning and Wittgenstein's language philosophy to quantify the likelihood of bullsh*t in any given text.

## Installation

### Prerequisites

Make sure you have Python installed. You can download it from [python.org](https://www.python.org/).

### Clone Repository

```bash
git clone https://github.com/The-H4CKER/bs-meter.git
cd bs-meter
```

### Create Virtual Environment (Recommended)

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows use `venv\Scripts\Activate.ps1`
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Add RoBERTa Model

Download [model.tensors](https://drive.google.com/file/d/1e3b62UMX4NAlWr6U4FQpq2gVn9ZSt1nk/view?usp=drive_link) and copy it to bs-meter/models/RoBERTa.

## Usage

```bash
python main.py
```

## Contributing

Contributions to enhance the BS Meter are welcome! Please submit issues and pull requests on GitHub. When contributing, ensure that:

- Your code adheres to the repository style guidelines.
- You provide tests for new features or bug fixes.
- You update documentation as needed.

## Citation

If you use the BS Meter in your research or projects, please cite the original paper:

> Trevisan, A., Giddens, H., Dillon, S., & Blackwell, A. F. (2024). *Measuring Bullshit in the Language Games played by ChatGPT*. arXiv preprint arXiv:2411.15129.

## Acknowledgements

This project is based on the research conducted by Alessandro Trevisan, Harry Giddens, Sarah Dillon, and Alan F. Blackwell. Their work provides the theoretical and experimental foundation for this BS Meter. We also acknowledge the contributions of the open-source community in developing the tools and libraries used in this project.

## License

This project is licensed under the MIT License. See the LICENSE file for details.
