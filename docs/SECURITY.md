# FaceLock — Security Architecture & Threat Model

## 1. Executive Summary

FaceLock implements a defense-in-depth biometric security architecture combining:
- **Authenticated Symmetric Encryption at Rest** (AES-128-CBC + HMAC-SHA256 via Fernet).
- **Presentation Attack Detection (PAD)** via temporal Eye Aspect Ratio (EAR) blink verification.
- **Privacy-First Audit Trail** with automated sanitization prohibiting biometric vector leakage.
- **Input Sanitization & Injection Immunity** against path traversal and SQL injection vectors.

---

## 2. Biometric Template Storage & Cryptography

### Encryption at Rest Architecture
Raw biometric vectors (128-dimensional IEEE 754 float64 arrays, 1024 bytes) represent sensitive biometric PII. FaceLock **never** stores plaintext embeddings in the database or filesystem.

```
128-D Biometric Vector (1024 Bytes Raw)
       │
       ▼
Fernet Symmetric Envelope (AES-128-CBC + HMAC-SHA256)
       ├── Encryption Key: 256-bit URL-Safe Base64 (data/.facelock_secret.key)
       ├── Initialization Vector (IV): Cryptographically Random per payload
       └── Authenticated Tag: HMAC-SHA256 ensuring integrity and authenticity
       │
       ▼
Encrypted Template File: data/users/<user_id>/template.enc
```

### Integrity Verification
If an adversary attempts bit-flipping, byte truncation, or template substitution, [`core/security.py`](file:///c:/Users/RANI%20NAYAK/OneDrive/Projects/Face%20Recognition%20and%20Password-Reset%20System/core/security.py) detects HMAC failure upon decryption and raises a `ValueError`, which is trapped and logged as `STORAGE_ERROR`.

---

## 3. Liveness Challenge & Anti-Spoofing Layer

### 2D Presentation Attack Defenses
Static photograph attacks (printed photos, smartphone still images, social media portraits) are rejected by mandatory real-time blink detection:

```
                          ┌───────────────────────────┐
                          │     START CHALLENGE       │
                          │   (Timer: 8.0s limit)     │
                          └─────────────┬─────────────┘
                                        │
                                        ▼
                                ┌───────────────┐
                     ┌──────────┤  EAR >= 0.22? ├──────────┐
                     │          └───────┬───────┘          │
                     │ YES              │ NO (Eyes Closed) │
                     ▼                  ▼                  │
             ┌──────────────┐   ┌───────────────┐          │
             │  EYES_OPEN   │   │  EYES_CLOSED  │          │
             └──────┬───────┘   └───┬───────────┘          │
                    │               │ Held 1-25 frames     │
                    │   Blink Cycle │                      │
                    └───◄───────────┘                      │
                            │                              ▼
                  [2 Blinks Completed]             [Timeout / >25 Frames]
                            │                              │
                            ▼                              ▼
                    LIVENESS: PASSED               LIVENESS: FAILED
```

- **Prolonged Eye Closure Defense**: If an attacker presents a photo of a person with closed eyes, the detector counts consecutive closed frames. Exceeding `BLINK_MAX_FRAMES = 25` flags the attempt as a static spoof and rejects it.
- **Timeout Gating**: If 2 natural blinks are not registered within `LIVENESS_TIMEOUT_SECONDS = 8.0s`, access is immediately denied.

---

## 4. Threat Modeling Matrix

| Threat Vector | Attack Mechanism | FaceLock Mitigation | Risk Level |
| :--- | :--- | :--- | :---: |
| **2D Presentation Attack** | Printed photo or phone screen held in front of webcam. | Eye Aspect Ratio (EAR) blink challenge gating before feature extraction. | **Mitigated** |
| **Template Theft at Rest** | Access to disk / database to extract face embeddings. | AES-128-CBC + HMAC-SHA256 authenticated encryption on all template files. | **Mitigated** |
| **Ciphertext Tampering** | Manipulating encrypted template bits to bypass distance match. | HMAC-SHA256 tag verification rejects corrupted or altered bytes. | **Mitigated** |
| **Path Traversal Injection** | Malicious user IDs like `../../etc/passwd` or `..\System32`. | Strict regex `^[a-zA-Z0-9_-]{2,32}$` in `validate_user_id()`. | **Mitigated** |
| **SQL Injection** | SQL payloads in `user_id` or `display_name`. | Parameterized SQLite queries (`?` binding) across all database queries. | **Mitigated** |
| **Biometric Log Leakage** | Storing embeddings or coordinates in diagnostic logs. | Mandatory sanitizer strips embeddings, arrays, coordinates, and passwords. | **Mitigated** |
| **Log Flooding / DoS** | Excessive attempts filling disk space. | `RotatingFileHandler` with 2 MB limit and 5 backup archives. | **Mitigated** |

---

## 5. Security Limitations & Hardening Guidelines

1. **3D Masks & Video Replay**: Sophisticated 3D silicone masks or deepfake video sequences with simulated eye blinks require active depth sensors (IR / LiDAR / Structured Light) or challenge-response head pose prompts.
2. **Key Storage Hardening**: The master AES key in `data/.facelock_secret.key` is stored on the local file system. In enterprise deployments, master keys should be bound to **Hardware Security Modules (HSM)**, **TPM 2.0**, or **Windows DPAPI**.
3. **Camera Device Authentication**: Standard webcams do not sign video frames cryptographically. A compromised OS could inject synthetic video frames via virtual camera drivers (e.g. OBS Virtual Camera). Secure deployments should restrict virtual camera devices.
