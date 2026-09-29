# FaceLock

AI-Powered Facial Biometric Authentication & Secure Credential Recovery System

---

## Problem Statement

Traditional knowledge-based authentication mechanisms (passwords, PINs, and security questions) suffer from critical security and usability flaws:
1. **Vulnerability to Credential Attacks**: Weak, reused, or shared passwords are easily compromised through phishing, brute-force attacks, and credential stuffing.
2. **Insecure Password Recovery**: Traditional recovery flows rely heavily on SMS OTPs (susceptible to SIM swapping) or security questions (easily discovered through open-source intelligence).
3. **Flaws in Naive Facial Recognition**: Simple computer-vision implementations often rely on single instantaneous frames (prone to sensor noise), lack anti-spoofing presentation attack detection (PAD), store unprotected face embeddings in plaintext, and arbitrarily hardcode matching thresholds without empirical evaluation.

---

## Solution

**FaceLock** is an end-to-end, privacy-preserving facial authentication system that replaces fragile passwords with verifiable biometric ownership:
- **Multi-Frame Centroid Aggregation**: Collects 15 validated facial samples over a temporal window to eliminate high-frequency camera noise and micro-expression variations.
- **Dynamic Anti-Spoofing (Liveness Detection)**: Computes Eye Aspect Ratio (EAR) across 68 facial landmarks to enforce natural blink challenges, defeating 2D printed photographs and smartphone screen attacks.
- **Zero-Plaintext Template Encryption**: All 128-D biometric vectors are encrypted at rest using AES-128-CBC with HMAC-SHA256 integrity protection.
- **1:N Multi-User Identity Resolution**: Fast Euclidean distance scanning against encrypted candidate databases in under 50 ms.
- **Empirical Threshold Analysis**: Uses ROC, FAR/FRR curves, and Equal Error Rate (EER) modeling to scientifically calibrate operating thresholds ($\tau = 0.60$).

---

## Key Features

- **8-State Telemetry HUD**: Real-time OpenCV Heads-Up Display showing live EAR telemetry, facial target brackets, matching distance, and status state badges (`IDLE`, `DETECTING`, `REGISTERING`, `LIVENESS_CHECK`, `AUTHENTICATING`, `SUCCESS`, `DENIED`, `ERROR`).
- **Temporal Eye Aspect Ratio (EAR) Verification**: Mathematical blink detection with prolonged eye-closure rejection and 8.0-second challenge timeout.
- **Multi-Sample Centroid Registration**: 15-sample temporal aggregation projected onto the unit hypersphere ($L_2 = 1.0$).
- **Multi-User Account Repository**: Structured SQLite storage for metadata coupled with isolated per-user encrypted templates (`data/users/<id>/template.enc`).
- **Cryptographic Template Security**: Fernet (AES-128-CBC + HMAC-SHA256) authenticated encryption with automatic 256-bit master key generation.
- **Privacy-Preserving Audit Trail**: Structured JSON security logging with automated sanitization prohibiting passwords, keys, or raw vectors from ever reaching log files.
- **Offline ML Evaluation Suite**: CLI evaluator producing genuine/impostor distributions, confusion matrices, and ROC curves (`python -m evaluation.evaluator`).
- **Comprehensive Test Suite**: 95 hardware-isolated unit, integration, and security tests.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           FACELOCK APPLICATION                             │
│                                 (main.py)                                   │
└───────────────┬─────────────────────────────────────────────┬───────────────┘
                │                                             │
    ┌───────────▼───────────┐                     ┌───────────▼───────────┐
    │  WORKFLOWS SUBSYSTEM  │                     │   HUD & UI ENGINE     │
    │  ├── register.py      │                     │   └── ui_renderer.py  │
    │  └── authenticate.py  │                     │       (8 State HUD)   │
    └───────────┬───────────┘                     └───────────────────────┘
                │
    ┌───────────▼─────────────────────────────────────────────────────────────┐
    │                            CORE SUBSYSTEMS                              │
    │  ├── embedding.py   (dlib 128-D ResNet, Quality Gate, Centroid Agg)     │
    │  ├── liveness.py    (Facial Landmarks, EAR Blink Anti-Spoofing)         │
    │  ├── matcher.py     (Euclidean Metric Space Distance & Thresholds)      │
    │  ├── security.py    (AES-128-CBC + HMAC-SHA256 Template Encryption)     │
    │  ├── user_db.py     (SQLite User Repository & Encrypted File Storage)   │
    │  └── audit_logger.py(Append-Only JSON Audit Log & Rotation Policy)      │
    └───────────┬─────────────────────────────────────────────────────────────┘
                │
    ┌───────────▼───────────┐                     ┌───────────────────────────┐
    │   HARDWARE / I/O      │                     │   EVALUATION SUBSYSTEM    │
    │   └── camera.py       │                     │   ├── metrics.py          │
    │       (OpenCV Stream) │                     │   ├── plots.py            │
    │                       │                     │   └── evaluator.py        │
    └───────────────────────┘                     └───────────────────────────┘
```

---

## Technology Stack

- **Programming Language**: Python 3.10+ / 3.13
- **Computer Vision & Face Encodings**: `OpenCV (cv2)`, `face_recognition`, `dlib` (ResNet-34 128-D face encoder)
- **Numerical Processing**: `NumPy` (Vectorized metric distances and matrix math)
- **Cryptography**: `cryptography` (Fernet authenticated AES-128-CBC with HMAC-SHA256)
- **Database**: `SQLite3` (Parameterized relational metadata storage)
- **Data Visualization**: `Matplotlib` (ROC curves, FAR/FRR distributions, KDE overlays)
- **Testing**: `pytest` (95 hardware-isolated unit and integration tests)

---

## How It Works

### Registration
1. User presses **`[R]`** (Quick Register) or **`[U]`** (Enroll New Named User).
2. The camera captures incoming video frames; each frame is validated (exactly 1 face detected, bounding box $\ge 60 \times 60\text{ px}$).
3. Once 15 valid frames are collected, FaceLock computes the centroid:
   $$\bar{v} = \frac{1}{N} \sum_{i=1}^{N} v_i, \qquad v_{\text{final}} = \frac{\bar{v}}{\|\bar{v}\|_2}$$
4. The normalized vector is encrypted via AES and saved to `data/users/<id>/template.enc`, with structured user metadata recorded in SQLite.

### Liveness Detection
1. When authentication is requested (**`[A]`**), FaceLock starts an 8.0-second timer and prompts for 2 natural blinks.
2. 6 landmark coordinates per eye are tracked in real-time to compute the Eye Aspect Ratio (EAR).
3. A state transition sequence (`EYES_OPEN` $\to$ `EYES_CLOSED` $\to$ `EYES_OPEN`) verifies dynamic liveness. Static photographs or held-closed eyes (> 25 frames) trigger instant rejection.

### Authentication
1. Upon passing liveness, FaceLock extracts the 128-D candidate vector from the verified frame.
2. The vector is compared against all enrolled templates in Euclidean metric space:
   $$d(u, v) = \|u - v\|_2$$
3. If $\min(d) \le \tau$ (default $\tau = 0.60$), access is granted and the user identity is displayed on the HUD. If $\min(d) > \tau$, access is denied.

### Password Recovery
1. Authorized users can reset their active biometric template directly through the camera interface by pressing **`[R]`**.
2. Replaces fragile SMS OTPs with authenticated biometric ownership re-enrollment.

---

## Machine Learning / Computer Vision

### 128-D Deep Metric Representation
FaceLock utilizes dlib's deep convolutional ResNet model trained with triplet loss. The network maps facial crops into a 128-dimensional Euclidean space where distances directly correspond to facial dissimilarity:
- **Same Identity**: Yields tight clusters with small distances ($d \approx 0.30 - 0.52$).
- **Different Identities**: Yields orthogonal/distant vectors with large distances ($d \approx 1.15 - 1.55$).

### Eye Aspect Ratio (EAR) Formulation
$$\text{EAR} = \frac{\|p_2 - p_6\|_2 + \|p_3 - p_5\|_2}{2.0 \cdot \|p_1 - p_4\|_2}$$
- **Open Eyes**: $\text{EAR} \in [0.25, 0.38]$
- **Closed Eyes**: $\text{EAR} < 0.22$ (`EAR_THRESHOLD`)

---

## Security

- **Encrypted Biometric Templates**: 128-D float64 vectors (1024 raw bytes) are encrypted at rest using AES-128-CBC + HMAC-SHA256 authenticated envelopes.
- **Zero Plaintext Storage**: Passwords and raw embeddings are never written to database tables or text files.
- **Sanitized Audit Logging**: The security event logger (`data/logs/security_audit.log`) automatically strips passwords, encryption keys, and raw biometric coordinates before serialization.
- **Injection Resilience**: User IDs are validated against strict alphanumeric regex (`^[a-zA-Z0-9_-]{2,32}$`), eliminating path traversal (`../`) and SQL injection attacks.

---

## Evaluation

Experimentally obtained metrics generated by `python -m evaluation.evaluator` across 1,120 biometric comparisons:

| Metric | Measured Value | Operational Interpretation |
| :--- | :---: | :--- |
| **Total Genuine Pairs** | `525` | Same-subject comparisons across temporal variations |
| **Total Impostor Pairs** | `595` | Cross-subject comparisons across distinct identities |
| **Genuine Distance ($\mu \pm \sigma$)** | `0.5245 ± 0.0323` | Average intra-class distance [$0.4251, 0.6153$] |
| **Impostor Distance ($\mu \pm \sigma$)** | `1.4108 ± 0.0604` | Average inter-class distance [$1.2282, 1.5862$] |
| **Equal Error Rate (EER)** | `0.00%` | Achieved at operating threshold $\tau = 0.6200$ |
| **Default Threshold ($\tau = 0.60$) FAR** | `0.00%` | Zero impostor false accepts |
| **Default Threshold ($\tau = 0.60$) FRR** | `0.38%` | Genuine reject rate (2 out of 525 pairs) |
| **Overall Accuracy** | `99.82%` | High discrimination boundary separation |
| **ROC AUC** | `1.0000` | Area Under the Receiver Operating Curve |

---

## Installation

### Prerequisites
- Python 3.10, 3.11, 3.12, or 3.13
- CMake and C++ build tools (required for `dlib`)
- Working webcam

### Setup
```powershell
# 1. Clone the repository
git clone https://github.com/your-username/facelock.git
cd facelock

# 2. Create and activate a virtual environment
python -m venv .venv
.\.venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Copy environment configuration template
copy .env.example .env
```

---

## Configuration

All system parameters can be customized in [`config.py`](file:///c:/Users/RANI%20NAYAK/OneDrive/Projects/Face%20Recognition%20and%20Password-Reset%20System/config.py) or via `.env`:

| Parameter | Default | Description |
| :--- | :---: | :--- |
| `SAMPLE_COUNT` | `15` | Valid face frames required to complete registration |
| `MIN_FACE_SIZE` | `60` | Minimum bounding box width/height in pixels |
| `EAR_THRESHOLD` | `0.22` | Eye Aspect Ratio threshold for closed eyelids |
| `REQUIRED_BLINKS` | `2` | Number of natural blinks to fulfill liveness challenge |
| `LIVENESS_TIMEOUT_SECONDS` | `8.0` | Time window to complete blink verification |
| `MATCH_THRESHOLD` | `0.60` | Maximum Euclidean distance for positive match |

---

## Running the Application

Launch the live interactive OpenCV application:
```powershell
python main.py
```

### In-App Controls
| Hotkey | Action | Description |
| :---: | :---: | :--- |
| **`a`** | **Authenticate** | Initiates blink challenge, extracts face vector, and matches across enrolled profiles. |
| **`r`** | **Quick Register** | Collects 15 samples and updates the primary active credential. |
| **`u`** | **Enroll New User** | Enrolls an additional user profile into SQLite and encrypted storage. |
| **`l`** | **List Users** | Prints directory of registered accounts in console and HUD. |
| **`q`** | **Quit** | Releases camera hardware and exits cleanly. |

---

## Testing

Run the complete 95-test suite completely isolated from hardware:

```powershell
pytest -v
```

Execute individual test suites:
```powershell
# Security and edge cases
pytest tests/test_security_and_edge_cases.py -v

# Metric space mathematics
pytest tests/test_matcher_properties.py -v

# Liveness anti-spoofing
pytest tests/test_liveness.py -v

# Multi-user SQLite and encryption
pytest tests/test_multi_user.py -v
```

Run threshold analysis and generate plots:
```powershell
python -m evaluation.evaluator
```

---

## Project Structure

```
FaceLock/
├── config.py                 # Centralized configuration and tunable thresholds
├── main.py                   # Real-time OpenCV application and HUD event loop
├── requirements.txt          # Explicitly pinned dependencies
├── .env.example              # Environment variables template
├── .gitignore                # Protection against credential and secret leakage
├── README.md                 # Primary system documentation
├── ARCHITECTURE.md           # Modular architecture and sequence diagrams
├── DEMO_GUIDE.md             # 3-5 minute live presentation walkthrough
│
├── core/                     # Core computational and cryptographic modules
│   ├── embedding.py          # 128-D ResNet extraction, quality gates, centroid aggregation
│   ├── liveness.py           # 68-point landmarks, EAR formulation, blink detector
│   ├── matcher.py            # Euclidean distance computation and threshold classification
│   ├── security.py           # AES-128-CBC + HMAC-SHA256 template encryption
│   ├── user_db.py            # SQLite repository and encrypted template file manager
│   ├── audit_logger.py       # Append-only sanitized JSON security event logger
│   └── ui_renderer.py        # 8-state HUD rendering engine
│
├── workflows/                # Business logic orchestrators
│   ├── register.py           # Multi-frame enrollment and centroid registration
│   └── authenticate.py       # Liveness-gated 1:N multi-user authentication
│
├── evaluation/               # ML evaluation and threshold analysis suite
│   ├── evaluator.py          # Standalone CLI threshold evaluation runner
│   ├── metrics.py            # FAR, FRR, EER, ROC, precision, recall, F1 calculators
│   ├── plots.py              # Matplotlib visualizers (ROC, FAR/FRR, distributions)
│   └── plots/                # Output directory for generated PNG figures
│
├── docs/                     # In-depth technical guides
│   ├── ARCHITECTURE.md       # Full architecture specification
│   ├── EVALUATION.md         # Mathematical formulations and threshold analysis
│   └── SECURITY.md           # Threat models, cryptographic guarantees, and PAD
│
├── tests/                    # 95 hardware-isolated automated tests
│   ├── test_audit_logger.py
│   ├── test_evaluation_metrics.py
│   ├── test_evaluator.py
│   ├── test_liveness.py
│   ├── test_matcher_properties.py
│   ├── test_multi_user.py
│   ├── test_multiframe_registration.py
│   └── test_security_and_edge_cases.py
│
└── utils/                    # Hardware and I/O utilities
    └── camera.py             # OpenCV webcam context manager with lifecycle cleanup
```

---

## Limitations

1. **2D Camera Constraint**: Relies on standard RGB sensors. While EAR blink detection stops printed photos and static screens, advanced 3D silicone masks or deepfake video replays require infrared/depth sensors.
2. **Extreme Lighting & Occlusions**: Heavy backlighting, dark environments, sunglasses, or surgical masks can degrade landmark extraction.
3. **Local Key Storage**: In this implementation, the master AES key is persisted locally in `data/.facelock_secret.key`. Enterprise deployments should bind master keys to a Hardware Security Module (HSM) or TPM 2.0.

---

## Future Improvements

- **3D Depth / Infrared Integration**: Support Intel RealSense or Apple TrueDepth sensors for hardware-level anti-spoofing.
- **Challenge-Response Head Poses**: Dynamic prompts (*"Turn head left"*, *"Smile"*) to complement blink verification.
- **Hardware Enclave Key Protection**: Windows DPAPI or TPM-backed cryptographic master key protection.
- **Continuous Background Re-Authentication**: Low-overhead periodic verification for session lock protection.
