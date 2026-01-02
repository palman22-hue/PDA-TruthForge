# Security Policy

## Supported versions

PDA-TruthForge is currently an experimental/prototype project and only the `main` branch is actively maintained.  
Other branches or forks are not covered by security support.

## Deployment model

PDA-TruthForge is designed for local use, on top of a locally installed LLM runtime (such as Ollama with the `mistral` model).  
This repository does **not** provide a public, maintainer-operated API or online service; any external exposure (for example, via a custom-built web service) is the responsibility of the deploying party.

## Data & privacy

- Input data (prompts, context) is processed locally via the chosen LLM runtime.  
- Any logs, configuration files, and session data are stored locally on the user’s system.  
- The maintainer does not receive automatic telemetry or user data.

Users are responsible for:
- Protecting any sensitive data they choose to send through PDA-TruthForge to a model.  
- Securing their own host system, operating system, and runtime (Ollama, Python environment, etc.).

## Reporting a vulnerability

If you discover a potential vulnerability or a serious misuse risk in PDA-TruthForge, please use **responsible disclosure**.

If Private Vulnerability Reporting is enabled for this repository:
1. Use the **“Report a vulnerability”** button on the repository’s GitHub page to submit a private report.

If Private Vulnerability Reporting is not available:
1. Open a minimal public issue that:
   - Only states that you have found a potential vulnerability.
   - Does **not** include technical details or exploit code.
2. Wait for the maintainer to respond with an appropriate private channel for sharing details.

Where possible, include:
- A description of the vulnerability or misuse scenario.  
- Reproduction steps or a proof-of-concept (shared only through a private channel).  
- Relevant environment details (OS, Python version, Ollama version, model version).

The goal is to provide an initial response within 14 days, including an indication of next steps.

## Scope

In scope:
- The code in this repository (`main` branch), including:
  - Orchestration and architecture code (e.g. `api.py`, orchestration/routing layers).  
  - Modules for memory, emotions, reflection, and related AI logic.  
  - Installation and configuration scripts referenced in the README.

Out of scope:
- External LLM runtimes (such as Ollama, Mistral, or other models and their own vulnerabilities).  
- Third-party services, wrappers, or deployments that expose PDA-TruthForge to the internet.  
- Hardware- and OS-specific issues that do not directly originate from the code in this repository.

## Responsible use

PDA-TruthForge is intended as an ethical foundation for LLM-based systems.  
Users are strongly discouraged from using the system for:

- Developing or supporting harmful, deceptive, or unethical AI applications.  
- Bypassing safety mechanisms of upstream models or platforms.

If you observe misuse of PDA-TruthForge in the wild, you may report it using the same contact path as above.
