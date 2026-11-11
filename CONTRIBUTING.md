# Contributing Guidelines & Contributor License Terms

Thank you for your interest in contributing to this project. To ensure clarity regarding rights, commercial distribution, and long-term maintenance, all contributors must agree to the licensing and contribution terms outlined below before their contributions can be accepted.

---

## 1. Project Licensing Model

This project is dual-licensed:
1. **Open Source Edition:** Licensed under the **GNU Affero General Public License v3.0 (AGPL-3.0-only)**.
2. **Commercial Edition:** Available under separate commercial/proprietary license agreements directly from the Project Maintainer(s) for environments where AGPL-3.0 compliance is not feasible.

---

## 2. Contributor Grant of Rights

By submitting a pull request, patch, code snippet, or any other material (collectively, "Contribution") to this repository or to the Project Maintainer(s), you agree to the following terms:

### 2.1 Copyright Retention
You retain full ownership of the copyright and intellectual property rights in and to your Contribution. Nothing in this agreement transfers copyright ownership from you to the Project Maintainer(s).

### 2.2 Inbound License Grant to Project Maintainer(s)
You hereby grant to the Project Maintainer(s) (and their designated successors and assigns) a perpetual, worldwide, non-exclusive, no-charge, royalty-free, irrevocable license to:
* Use, reproduce, prepare derivative works of, display, perform, distribute, and sublicense your Contribution and any derivative works thereof.
* Re-license and distribute your Contribution under alternative terms at the Maintainer's sole discretion, including but not limited to commercial, closed-source, and proprietary licenses.
* Grant downstream sublicenses to third-party commercial customers without any obligation to account to you or share revenue.

### 2.3 Outbound License to the Community
Your Contribution will simultaneously be made available to the public under the open-source terms specified in the repository (AGPL-3.0-only).

### 2.4 Contributor Representations
You represent and warrant that:
* You are the sole author of the Contribution and have legal authority to grant the rights set forth above.
* If your employer has intellectual property rights in your work, you have received explicit permission to make this Contribution on their behalf, or your employer has waived those rights.
* To the best of your knowledge, your Contribution does not infringe any third-party copyright, patent, trademark, trade secret, or other proprietary right.

---

## 3. Pull Request Process

1. **Acknowledgment:** In each Pull Request, ensure you have read and agree to these terms.
2. **Sign-Off:** Include a sign-off line in your commit messages:
   ```text
   I agree to the Contributor Guidelines and Inbound License Terms in CONTRIBUTING.md.
   ```
3. **CI Validation:** Automated checks (such as CLA verification bots, if active) will verify agreement prior to merging.

---

## 4. Code Standards & Testing

Before submitting code:
* Run `cargo fmt --all -- --check` to ensure formatting complies with standard Rust guidelines.
* Run `cargo clippy --all-targets -- -D warnings` to verify zero compiler warnings.
* Run `cargo test` to confirm all existing and added test suites pass.
* Include doc-tests and unit tests for any new public APIs or modules.
