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

### 5b. Charger comparison — one OEM baseline session (device-only)

**Constraint:** no USB meter, IR thermometer, or other lab gear. Logs + manual notes only.
**Safety:** hard gate. Do not treat empty/`null` `safety_pass` as a pass (fail closed).

1. Same phone, fixed power settings (screen off; airplane **or** fixed radios).
2. Note cable, port, ambient, case on/off; target SoC window **20–25% → 80%**.
3. Install Android platform-tools; enable USB debugging; confirm `adb devices`.
4. Probe (or skip — ADB sessions auto-probe). **Do not type model/OS by hand:**

```bash
python3 src/log_charger_session.py --probe-device
```

5. Start logging before plugging the OEM charger (auto-writes `device_info.json`):

```bash
python3 src/log_charger_session.py \
  --charger-id oem --run-id oem_r01 --repeat-index 1 \
  --source adb --interval 5 \
  --cable shared_usb_c_label_A --port device_usb_c \
  --ambient "room note" --event start
```

6. Stop (Ctrl-C) at ~80% SoC. Add absolute-stop notes (smell, OS thermal warning,
   connector/brick too hot to hold) to `data/charger_sessions/manifest.json`.
7. Leave `experiments/charger_comparison_results.json` `safety_pass` as `null`
   until OEM envelopes are measured from ≥3 successful repeats.

**iOS limits:** no ADB. Public APIs typically omit battery temperature and charge
current. Prefer Android for comparable heat/throttle evidence; if logging iOS,
leave `batt_temp_c` / `charge_ua` blank and record that heat evidence is weaker.

Schema dry-run without a device:

```bash
python3 src/log_charger_session.py --charger-id oem --run-id oem_dry --source demo --samples 3 --no-probe --no-manifest
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