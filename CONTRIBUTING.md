# Contributing to JPY→BDT Exchange-Rate Forecast

Thank you for your interest in contributing to this project! This guide will help you get started.

## How to Contribute

### 1. Report Issues

If you find any issues or bugs, please:
- Check if the issue has already been reported
- Create a new issue with a clear title and description
- Include steps to reproduce the issue
- Attach relevant error messages or screenshots

### 2. Suggest Enhancements

If you have ideas for improvements:
- Create a new issue with your enhancement suggestion
- Explain why this enhancement would be valuable
- If possible, suggest implementation ideas

### 3. Submit Pull Requests

To submit code changes:

#### Setup Your Environment
```bash
# Fork and clone the repository
git clone [your-fork-url]
cd jpy_bdt_prediction

# Create a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

#### Development Process
1. Create a new branch from `main` (branch naming: `feature/`, `fix/`, `docs/` prefixes recommended)
2. Make your changes
3. Test your changes
4. Commit your changes with clear commit messages
5. Push your branch to your fork
6. Create a pull request to the `main` branch
7. **All pull requests require at least 1 review before merging**
8. **Direct merging to main branch is disabled** - all changes must go through PR review

#### Branch Protection Rules
- **Main branch is protected from direct pushes**
- **Required review from repository owner before merging**
- **Code must pass all checks before merging**
- **Linear history enforced** (no merge commits)

#### Code Guidelines
- Follow PEP 8 guidelines for Python code
- Write clear comments and documentation
- Ensure all tests pass
- Add new tests for new functionality
- Maintain compatibility with Python 3.8+

### 4. Project Structure

```
jpy_bdt_prediction/
├── data/                    # Dataset files
├── fetch_data.py           # Data fetching script
├── train_lstm.py          # Model training script
├── build_dashboard.py     # Dashboard generation
├── robustness_check.py    # Model robustness checks
├── experiment_*.py        # Experiment scripts
├── dashboard.html         # Interactive dashboard
├── lstm_model.keras       # Trained LSTM model
├── requirements.txt       # Dependencies
├── README.md              # Project documentation
└── LICENSE                # MIT License
```

### 5. Testing

To test your changes:

```bash
# Run all scripts to ensure they work
python3 fetch_data.py
python3 train_lstm.py
python3 build_dashboard.py
```

### 6. Code of Conduct

Please be respectful and considerate when contributing. Remember:
- Be kind to other contributors
- Listen to constructive feedback
- Focus on the project goals
- Help others learn

## Contact

If you have questions about contributing, please:
- Open an issue
- Check the README for contact information
- Look for existing discussions

Thank you for helping make this project better!