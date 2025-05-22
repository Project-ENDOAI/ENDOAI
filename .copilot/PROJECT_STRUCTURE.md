# Project Structure & Folder Generation Guide

This document explains how to update and synchronize the project folder structure, especially when adding new modules or reorganizing code. It is intended for both developers and AI assistants (like Copilot).

---

## How to Update the Project Structure

1. **Review the Main README**

   - The canonical project structure is described in `endoai/README.md` (and sometimes in the root `README.md`).
   - Use the directory tree in the README as the source of truth for what folders and files should exist.

2. **Generate Missing Folders**

   - Compare the actual folders in the repository to those listed in the README.
   - Create any missing directories (e.g., `src/preoperative/`, `src/intraoperative/`, `core/`, `pipelines/`, etc.).
   - Ensure each directory contains a `README.md` and, for Python packages, an `__init__.py`.

3. **Add Standard Files**

   - For each new or reorganized folder:
     - Add a `README.md` describing the folder’s purpose.
     - Add an `__init__.py` if the folder is a Python package/module.
     - Add example or template scripts if appropriate.

4. **Keep Documentation in Sync**

   - Whenever you add, remove, or rename a folder, update the directory tree in the main `README.md` and in the relevant subfolder `README.md`.
   - Ensure all documentation links and references are correct.

5. **Automate Where Possible**

   - Use scripts (e.g., `scripts/add_readme_and_init.py`) to automatically create missing `README.md` and `__init__.py` files.
   - Consider writing a script to parse the directory tree from the README and generate the folder structure automatically.

---

## How to Ask Copilot to Generate Folders and Files (Without a Script)

You can instruct Copilot (or another AI assistant) to generate folders and files by:

1. **Describing the Structure in Your Prompt**

   - Clearly specify the desired folder and file structure in your prompt or question.
   - **Tip:** Use a bullet list or tree format, and specify both the folder and the file type (e.g., `src/preoperative/README.md`).
   - **Tip:** If you want Copilot to generate content for each file, specify what kind of content you want (e.g., "with a standard README template" or "with an empty __init__.py").

2. **Referencing the Directory Tree in README**

   - Point Copilot to the directory tree in your `README.md` and ask it to generate any missing folders/files.
   - **Tip:** Say "Generate all missing folders and add a README.md and __init__.py to each, following the structure in endoai/README.md."

3. **Using Explicit Prompts for Content**

   - For each file, specify what content or template you want.
   - **Tip:** For Python packages, say "add an empty __init__.py" or "add a docstring to each __init__.py".
   - **Tip:** For documentation, say "add a standard README.md describing the folder's purpose".

4. **Combining with Folder Selection**

   - Select the relevant parent folder in your IDE before prompting Copilot, so it knows where to create new files.
   - **Tip:** If Copilot gets lost, try selecting only the immediate parent folder (not the whole project).

5. **Review and Confirm**

   - After Copilot generates the code blocks for new files, copy them into your project manually or use your IDE's tools to create the files.
   - **Tip:** If Copilot generates duplicate or misplaced files, clarify your prompt or break it into smaller steps.

---

## Troubleshooting Copilot Folder/File Generation

- If Copilot gets lost or generates files in the wrong place:
  - Break your request into smaller steps (e.g., "Create the folder src/preoperative/ and add README.md and __init__.py").
  - Specify the full relative path for each file.
  - Ask for one folder at a time if needed.
  - Remind Copilot to avoid overwriting existing files unless you want to update them.
  - Use the phrase "do not repeat existing code" to avoid duplication.

- **If new files or folders do not appear in VSCode Explorer:**
  - Click the "Refresh" button at the top of the Explorer pane.
  - Or, right-click in the Explorer and select "Refresh".
  - If you still do not see the files, try closing and reopening the folder or restarting VSCode.
  - On some systems, file changes made outside VSCode (e.g., by scripts or Copilot) may not show up until you refresh the Explorer.

---

## Example Workflow

1. Edit `endoai/README.md` to update the desired structure.
2. Run or write a script to:
    - Parse the directory tree from the README.
    - Create any missing folders.
    - Add standard files as needed.
3. Manually review and update `README.md` files in each folder to provide context and documentation.

---

## Example Prompt

> "Copilot, please generate all folders and files shown in the directory tree in `endoai/README.md`. For each folder, add a `README.md` and an `__init__.py` if it's a Python package. Use standard templates for each file."

---

## Best Practices

- Keep the README directory tree up to date with the actual project structure.
- Use clear, descriptive folder and file names.
- Document the purpose of each folder in its `README.md`.
- Use scripts to automate repetitive setup tasks.

---

## See Also

- [../endoai/README.md](../endoai/README.md) — Main project structure and documentation.
- [add_readme_and_init.py](../scripts/add_readme_and_init.py) — Script to add `README.md` and `__init__.py` files.
- [COPILOT.md](../COPILOT.md) — Coding standards and preferences.
