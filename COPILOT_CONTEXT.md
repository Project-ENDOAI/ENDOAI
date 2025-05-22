# Copilot Context & Tools Instructions

This document explains how to add **context** for GitHub Copilot and other AI assistants, what "context" means, and how to use tools and folder selection to improve code suggestions.

---

## What is "Context"?

**Context** is the set of files, code, documentation, and configuration that Copilot or an AI assistant can "see" and use to generate relevant code suggestions. The more context you provide, the more accurate and helpful the AI's responses will be.

---

## How to Add Context

- **Open Files:** Open all relevant files in your editor. Copilot will use the content of open files as context.
- **Select Folders:** Use the folder selection or workspace features in your IDE to include entire directories. This helps Copilot understand project structure, dependencies, and relationships between modules.
- **Highlight Code:** Select or highlight code blocks you want Copilot to focus on. This is useful for refactoring or asking for improvements.
- **Provide Documentation:** Add or update README.md, docstrings, and comments. Copilot uses these to infer intent and domain knowledge.
- **Use Configuration Files:** Files like `pyproject.toml`, `requirements.txt`, `.env`, and `.gitignore` provide important context about dependencies and environment.

---

## What Does Folder Selection Do?

- **Folder selection** tells Copilot which parts of your project are most relevant for your current task.
- When you select a folder, Copilot will prioritize files in that folder for context.
- This is especially useful for large projects or monorepos.

**Examples:**
- Select `src/` to focus on source code.
- Select `tests/` to focus on testing and coverage.
- Select `notebooks/` for data exploration and prototyping.

### Selecting the Entire Project as Context

- **Selecting the entire project** gives Copilot access to all files, code, and documentation in your repository.
- This is useful when your question or task spans multiple modules, or when you want Copilot to understand the full project structure and dependencies.
- To select the entire project:
  - In VSCode, right-click your top-level project folder in the Explorer and choose "Add Folder to Workspace" (if not already).
  - Make sure the root folder is highlighted or selected before invoking Copilot or asking for suggestions.
  - Some Copilot integrations allow you to explicitly select "All files" or "Entire project" as context—use this option if available.

---

## What Are Instructions, Symbols, Related Files, and Problems?

### Instructions

- **Instructions** are special files (like `COPILOT.md`, `.copilot/INSTRUCTIONS.md`, or files with `instructions` in their name) that provide coding standards, preferences, or project-specific guidelines for Copilot and other AI assistants.
- These files help Copilot understand your expectations for code style, architecture, and best practices.
- Keep instructions up to date and place them at the project root or in relevant subfolders.

### Symbols

- **Symbols** refer to named entities in your codebase, such as functions, classes, variables, and methods.
- Copilot uses symbol information to understand code structure and relationships, enabling better navigation and more relevant suggestions.
- Keeping symbols well-named and documented improves Copilot's ability to assist you.

### Related Files

- **Related files** are files that are linked by import statements, usage, or project structure (e.g., a test file for a module, or a config file for a script).
- Copilot considers related files when generating suggestions, so keeping your project organized and using standard naming conventions (like `test_*.py` for tests) helps Copilot connect the dots.

### Problems

- **Problems** are issues detected by your IDE, linter, or test runner (e.g., syntax errors, failed tests, or warnings).
- Copilot can use problem reports to suggest fixes or improvements.
- Regularly check the "Problems" panel in your IDE and address issues for a smoother Copilot experience.

---

## What Do Tools Do?

- **Tools** are scripts or utilities that automate tasks (e.g., formatting, linting, testing, code generation).
- Copilot can use tool outputs (like test results or linter warnings) as context for suggestions.
- Common tools: `black`, `flake8`, `pytest`, `detect-secrets`, custom scripts in `scripts/`.

---

## Best Practices for Context

- Keep related files open when working on a feature or bug.
- Use descriptive names and docstrings for all functions, classes, and modules.
- Update documentation and READMEs regularly.
- Use `.gitignore` to exclude irrelevant files from context.
- When asking Copilot for help, specify which files or folders are relevant if possible.

---

## Example Workflow

1. Open the files you are working on and any related modules.
2. Select the relevant folder(s) in your IDE.
3. Run tools (e.g., tests, linters) and review their output.
4. Ask Copilot for suggestions, improvements, or documentation.
5. Review and edit Copilot's suggestions as needed.

---

## See Also

- [COPILOT.md](COPILOT.md) — Coding standards and project preferences.
- [README.md](README.md) — Project overview and structure.
