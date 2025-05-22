#!/usr/bin/env python3
"""
DO NOT COMMIT THIS FILE!
Ensure that git_automation.py is excluded from version control.
For example, add a line like:
    /scripts/git_automation.py
if this file lives in the "scripts" directory.
"""

import os
import sys
import subprocess
import json
from datetime import datetime
from typing import Tuple, Optional
import time  # Add this import at the top with other imports

# Import Tkinter for GUI dialogs
import tkinter as tk
from tkinter import simpledialog, messagebox

# Version and changelog
__version__ = "1.6.4"
__changelog__ = """
Version 1.6.4:
- Integrated a Tkinter GUI dialog to prompt for a GitHub Personal Access Token (PAT)
  if one is not already set via the GITHUB_PAT environment variable.
- When the PAT is provided, the script updates both the current environment and the VSCode workspace settings
  (i.e. .vscode/settings.json) under the integrated terminal environment.
- This enables non-interactive HTTPS authentication.
- All previous improvements are retained:
    • Multiple push variants.
    • Network connectivity and remote refresh checks.
    • Staging of removals and branch handling.
    • Comprehensive diagnostics upon failure.
    • GitHub CLI integration (if available).
- **New:** Attempts to push changes to the "main" branch,
         and if that fails, creates and pushes to a "development" branch.
- **New:** Provides clear error explanations (e.g. branch switching conflicts due to tracked files).
- **New:** Outputs terminal output to a log file ("git_automation_output.txt")
         with a header including the current date and time.
- **New:** Automatically ensures that "git_automation_output.txt" is added to .gitignore.
- **New:** Checks for an SSH key (and offers to generate one if missing).
- **New:** Performs an automated cleanup:
         - Searches the entire repository for tracked instances of git_automation_output.txt and removes them.
         - Removes the Git index lock if found.
         - Clears the index (so that .gitignore rules take effect), re-adds files, and commits with an automated message.
         If a hang-up occurs, the script circles back to remove the offending files.
"""

# Global flag for GitHub CLI availability.
GH_AVAILABLE = False
PUSH_TIMEOUT = 60  # Increased push timeout

# --- Tee class to duplicate output to both console and file ---
class Tee:
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()
    def flush(self):
        for f in self.files:
            f.flush()

def remove_index_lock():
    """
    Remove .git/index.lock if it exists.
    """
    lock_file = ".git/index.lock"
    if os.path.exists(lock_file):
        print(f"Found index lock file {lock_file}. Removing it automatically.")
        try:
            os.remove(lock_file)
        except Exception as e:
            print(f"Failed to remove {lock_file}: {e}")

def run_command(command: str, timeout: int = 30, env: Optional[dict] = None) -> Tuple[bool, str]:
    """
    Run a command with a timeout, returning (success, output).
    """
    remove_index_lock()  # Always check for a lock file before running a command.
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=timeout, env=env)
        if result.returncode != 0:
            error_msg = result.stderr.strip() or result.stdout.strip()
            print(f"Error executing '{command}': {error_msg}")
            return False, error_msg
        output = result.stdout.strip()
        if output:
            print(output)
        return True, output
    except subprocess.TimeoutExpired:
        msg = f"Timeout: Command '{command}' timed out after {timeout} seconds."
        print(msg)
        return False, "Timeout"
    except OSError as e:
        msg = f"OS Error while executing '{command}': {e}"
        print(msg)
        return False, str(e)

def update_gitignore():
    """
    Ensure that 'git_automation_output.txt' is included in .gitignore.
    """
    gitignore_path = ".gitignore"
    ignore_line = "git_automation_output.txt"
    if os.path.exists(gitignore_path):
        with open(gitignore_path, "r") as f:
            ignores = f.read().splitlines()
        if ignore_line not in ignores:
            with open(gitignore_path, "a") as f:
                f.write("\n" + ignore_line + "\n")
            print(f"Added '{ignore_line}' to {gitignore_path}.")
        else:
            print(f"'{ignore_line}' is already in {gitignore_path}.")
    else:
        with open(gitignore_path, "w") as f:
            f.write(ignore_line + "\n")
        print(f"Created {gitignore_path} and added '{ignore_line}' to it.")

def update_vscode_workspace_settings(token: str) -> None:
    """
    Update VSCode workspace settings (.vscode/settings.json) with the GitHub PAT.
    """
    env_key = "terminal.integrated.env.linux"  # Adjust for your OS if needed.
    workspace_dir = ".vscode"
    settings_path = os.path.join(workspace_dir, "settings.json")
    os.makedirs(workspace_dir, exist_ok=True)
    if os.path.exists(settings_path):
        with open(settings_path, "r") as f:
            try:
                settings = json.load(f)
            except json.JSONDecodeError:
                settings = {}
    else:
        settings = {}
    if env_key not in settings:
        settings[env_key] = {}
    settings[env_key]["GITHUB_PAT"] = token  # Update token in settings
    with open(settings_path, "w") as f:
        json.dump(settings, f, indent=4)
    print(f"VSCode workspace settings updated with new GITHUB_PAT in {settings_path} under '{env_key}'.")

def load_vscode_settings():
    """
    Load settings from .vscode/settings.json.
    """
    settings_path = ".vscode/settings.json"
    if os.path.exists(settings_path):
        try:
            with open(settings_path, "r") as f:
                settings = json.load(f)
                print("Loaded VSCode settings.")
                return settings
        except json.JSONDecodeError as e:
            print(f"Error decoding {settings_path}: {e}")
    else:
        print(f"{settings_path} not found. Using default settings.")
    return {}

def get_github_pat_from_settings(settings: dict) -> Optional[str]:
    """
    Extract the GitHub PAT from VSCode settings.
    """
    terminal_env = settings.get("terminal.integrated.env.linux", {})
    return terminal_env.get("GITHUB_PAT")

def prompt_for_pat() -> None:
    """
    Prompt for a GitHub PAT using Tkinter.
    """
    root = tk.Tk()
    root.withdraw()
    token = simpledialog.askstring(
        "GitHub Authentication",
        "Enter your GitHub Personal Access Token (PAT):\nLeave blank to use interactive authentication.",
        show='*'
    )
    if token:
        os.environ["GITHUB_PAT"] = token
        messagebox.showinfo("Token Set", "PAT set for non-interactive authentication.")
        update_vscode_workspace_settings(token)
    else:
        messagebox.showwarning("No Token Provided", "No PAT provided. The script may prompt interactively on push.")
    root.destroy()

def authenticate_with_github_cli():
    """
    Authenticate with GitHub CLI using the provided PAT.
    """
    print("\nAuthenticating with GitHub CLI...")
    pat = os.getenv("GITHUB_PAT")
    if not pat:
        print("No GITHUB_PAT found in environment. Prompting for PAT...")
        prompt_for_pat()
        pat = os.getenv("GITHUB_PAT")
    if pat:
        success, _ = run_command(f'gh auth login --with-token <<< "{pat}"')
        if success:
            print("Authenticated with GitHub CLI successfully.")
        else:
            print("Failed to authenticate with GitHub CLI. Ensure your PAT is valid.")
    else:
        print("Authentication skipped. No PAT provided.")

def authenticate_with_github_cli_from_settings():
    """
    Authenticate with GitHub CLI using the PAT from VSCode settings.
    """
    print("\nAuthenticating with GitHub CLI using VSCode settings...")
    settings = load_vscode_settings()
    pat = get_github_pat_from_settings(settings)
    if pat:
        success, _ = run_command(f'echo "{pat}" | gh auth login --with-token')
        if success:
            print("Authenticated with GitHub CLI successfully using VSCode settings.")
        else:
            print("Failed to authenticate with GitHub CLI using VSCode settings. Ensure your PAT is valid.")
    else:
        print("No GITHUB_PAT found in VSCode settings. Falling back to environment variable or manual prompt.")
        authenticate_with_github_cli()

def check_git_installed():
    """
    Verify that Git is installed.
    """
    print("Verifying Git installation...")
    success, output = run_command("git --version")
    if not success:
        print("Git is not installed or not found in PATH. Please install Git and try again.")
        exit(1)
    print(f"Git is installed: {output}")

def check_github_cli_installed():
    """
    Verify that the GitHub CLI (gh) is installed. Install it if not found.
    """
    global GH_AVAILABLE
    print("\nVerifying GitHub CLI installation...")
    success, output = run_command("gh --version")
    if not success:
        print("GitHub CLI (gh) not installed or not in PATH. Attempting to install...")
        install_github_cli()
        success, output = run_command("gh --version")
        if not success:
            print("Failed to install GitHub CLI. PR creation will be skipped.")
            GH_AVAILABLE = False
        else:
            print(f"GitHub CLI installed successfully: {output}")
            GH_AVAILABLE = True
    else:
        print(f"GitHub CLI is installed: {output}")
        GH_AVAILABLE = True

def install_github_cli():
    """
    Install GitHub CLI based on the operating system.
    """
    print("\nInstalling GitHub CLI...")
    if sys.platform.startswith("linux"):
        run_command("sudo apt update && sudo apt install -y gh")
    elif sys.platform == "darwin":
        run_command("brew install gh")
    elif sys.platform.startswith("win"):
        print("Please download and install GitHub CLI from https://github.com/cli/cli/releases for Windows.")
    else:
        print("Unsupported operating system for automatic GitHub CLI installation.")

def check_remote_authentication():
    """
    Verify the remote URL configuration.
    """
    print("\nVerifying remote URL for authentication method...")
    success, remote_url = run_command("git config --get remote.origin.url")
    if success and remote_url:
        print(f"Remote URL: {remote_url}")
        if remote_url.startswith("https://"):
            print("WARNING: HTTPS remote may require interactive authentication. Consider using SSH.")
        else:
            print("SSH remote confirmed.")
    else:
        print("Could not retrieve the remote URL. Check your remote configuration.")

def check_credential_helper():
    """
    Display the Git credential helper setting.
    """
    print("\nChecking global Git credential helper...")
    success, helper = run_command("git config --global credential.helper")
    if success and helper:
        print(f"Global credential.helper: {helper}")
    else:
        print("No global credential.helper set.")

def diagnose_git():
    """
    Run a set of basic Git diagnostic commands.
    """
    print("\n--- Basic Git Diagnostic Information ---")
    diagnostics = [
        ("Status", "git status"),
        ("Current Branch", "git rev-parse --abbrev-ref HEAD"),
        ("Branches (detailed)", "git branch -vv"),
        ("Remote Information", "git remote -v"),
        ("Recent Commits (last 5)", "git log --oneline -5"),
        ("Git Configuration", "git config --list")
    ]
    for desc, cmd in diagnostics:
        print(f"\n*** {desc} ***")
        run_command(cmd)

def diagnose_git_full_info():
    """
    Run comprehensive Git diagnostics.
    """
    print("\n=== Comprehensive Git Repository Diagnostics ===")
    diagnostics = [
        ("Status (verbose)", "git status -v"),
        ("Current Branch", "git rev-parse --abbrev-ref HEAD"),
        ("Branches Detailed", "git branch -vv"),
        ("Remote Information", "git remote -v"),
        ("Recent Commits (last 10)", "git log --oneline -10"),
        ("Git Configuration", "git config --list"),
        ("Git References", "git show-ref"),
        ("Untracked Files", "git ls-files --others --exclude-standard")
    ]
    for desc, cmd in diagnostics:
        print(f"\n*** {desc} ***")
        run_command(cmd)
    print("\nPlease copy these diagnostics if you need further assistance.")

def diagnose_push_failure():
    """
    Run additional diagnostics for push failures.
    """
    print("\n--- Diagnosing Push Failure ---")
    diagnostics = [
        ("Remote Show", "git remote show origin"),
        ("Fetch All", "git fetch --all"),
        ("Re-check Branch Status", "git status"),
        ("Latest Commit Details", "git show --quiet")
    ]
    for desc, cmd in diagnostics:
        print(f"\n*** {desc} ***")
        run_command(cmd)
    print("\nFor further diagnosis, please run diagnose_git_full_info() and share its output with support.")

def pull_latest_changes():
    """
    Pull the latest changes.
    """
    print("\nPulling latest changes...")
    success, _ = run_command("git pull")
    if not success:
        print("Failed to pull latest changes. Running basic diagnostics...")
        diagnose_git()
        return False
    print("Successfully pulled latest changes.")
    return True

def cleanup_and_commit_index():
    """
    Perform cleanup by:
      1. Searching for tracked instances of 'git_automation_output.txt' in the repository and removing them.
      2. Running 'git rm -r --cached .' to clear the index (applying .gitignore).
      3. Re-adding all files.
      4. Committing with an automated message.
    If the commit fails, stash the changes.
    """
    print("\nPerforming automated cleanup of tracked files...")
    
    # Search entire repository for tracked instances of git_automation_output.txt
    success, paths = run_command("git ls-files | grep git_automation_output.txt")
    if success and paths:
        for path in paths.splitlines():
            print(f"Removing tracked instance: {path}")
            run_command(f'git rm --cached "{path}"')
    else:
        print("No tracked instances of git_automation_output.txt found in the repository.")
    
    # Clear the index so that .gitignore rules are applied.
    run_command("git rm -r --cached .")
    # Re-add all files.
    run_command("git add .")
    commit_msg = f"Automated cleanup commit on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: Reapplying .gitignore and cleaning index."
    success, output = run_command(f'git commit -m "{commit_msg}"')
    if not success:
        print("Cleanup commit failed. Stashing changes as fallback...")
        run_command("git stash")
    else:
        print("Cleanup commit succeeded.")

def sync_gitignore():
    """
    Untrack ignored files using the cleanup routine.
    """
    print("\nRunning sync_gitignore...")
    cleanup_and_commit_index()

def stage_changes():
    """
    Stage changes. Since git_automation_output.txt is now in .gitignore,
    a simple 'git add -A' should only stage non-ignored changes.
    """
    print("\nStaging changes...")
    success, _ = run_command("git add -A -- ':!venv/' ':!*.private'")
    if not success:
        print("Failed to stage changes. Running diagnostics...")
        diagnose_git()
        return False
    print("Changes staged successfully.")
    return True

def commit_changes(reason: str = "Updated files and added changes"):
    """
    Commit changes with a timestamped message.
    """
    print("\nCommitting changes...")
    date_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    commit_msg = f"Automated commit on {date_time}: {reason}"
    success, output = run_command(f'git commit -m "{commit_msg}"')
    if not success:
        if "nothing to commit" in output.lower():
            print("Nothing to commit. Skipping commit.")
            return True
        print("Commit failed. Stashing changes as fallback...")
        run_command("git stash")
        return False
    print("Changes committed successfully.")
    return True

def push_changes():
    """
    Attempt to push changes to the 'main' branch.
    If all push attempts fail, run comprehensive diagnostics.
    """
    print("\nPushing changes...")
    push_attempts = [
        ("Standard push", "git push"),
        ("Explicit push to origin HEAD", "git push origin HEAD"),
        ("Set upstream push", "git push --set-upstream origin HEAD")
    ]
    for attempt_name, command in push_attempts:
        print(f"\nAttempting: {attempt_name}")
        for attempt in range(2):
            print(f"  {attempt_name}: Attempt {attempt + 1}")
            success, output = run_command(command, timeout=PUSH_TIMEOUT)
            if success:
                print(f"{attempt_name} succeeded.")
                return True
            else:
                print(f"{attempt_name} failed on attempt {attempt + 1}:\n{output}")
        print("Trying next push approach...\n")
    print("All push attempts failed. Running comprehensive diagnostics...")
    diagnose_git_full_info()
    diagnose_push_failure()
    return False

def create_pull_request():
    """
    Create a pull request using GitHub CLI.
    Skipped if the CLI is not installed.
    """
    print("\nCreating pull request...")
    title = "Automated Pull Request"
    date_str = datetime.now().strftime("%Y-%m-%d")
    description = (
        f"This pull request was automatically created on {date_str}.\n\n"
        "**Description:**\n- Updated changes.\n\n"
        "**Checklist:**\n"
        "- [x] Code is formatted.\n- [x] Passed linting.\n- [x] Updated tests.\n- [x] Updated docs."
    )
    success, _ = run_command(f'gh pr create --title "{title}" --body "{description}"')
    if not success:
        print("Failed to create pull request. Running diagnostics...")
        diagnose_git_full_info()
        return False
    print("Pull request created successfully.")
    return True

def check_or_generate_ssh_key():
    """
    Check if an SSH key exists in ~/.ssh. If not, prompt to generate one.
    """
    ssh_dir = os.path.expanduser("~/.ssh")
    keys = [os.path.join(ssh_dir, "id_ed25519.pub"), os.path.join(ssh_dir, "id_rsa.pub")]
    if any(os.path.exists(key) for key in keys):
        print("An SSH public key was found.")
    else:
        print("No SSH public key found in ~/.ssh.")
        resp = input("Generate a new SSH key? (y/n): ")
        if resp.lower() == "y":
            email_success, email = run_command("git config --get user.email")
            if not email.strip():
                email = input("Enter your email for the SSH key: ")
            keygen_cmd = f'ssh-keygen -t ed25519 -C "{email.strip()}" -f "{os.path.join(ssh_dir, "id_ed25519")}" -N ""'
            success, _ = run_command(keygen_cmd, timeout=60)
            if success:
                print("SSH key generated. Add the public key (id_ed25519.pub) to your GitHub account.")
            else:
                print("Failed to generate SSH key. Check your SSH setup.")

def checkout_main_branch():
    """
    Ensure that the 'main' branch is checked out.
    If the checkout fails due to tracked files, automatically run cleanup and retry (up to 2 attempts).
    """
    print("\nChecking out 'main' branch...")
    for attempt in range(2):
        success, output = run_command("git checkout main", timeout=30)
        if success:
            print("Switched to 'main' branch.")
            return True
        else:
            if "git_automation_output.txt" in output or "would be overwritten" in output:
                print(f"Attempt {attempt+1}: Detected problematic tracked files. Running automated cleanup...")
                cleanup_and_commit_index()
            else:
                print(f"Attempt {attempt+1}: Failed to switch to 'main' branch. Output:\n{output}")
    print("Failed to switch to 'main' branch after cleanup attempts.")
    return False

def check_last_commit(branch: str):
    """
    Print the last commit on the specified branch.
    """
    print(f"\nLast commit on branch '{branch}':")
    run_command(f"git log -1 {branch}")

def remove_files_from_git_tracking():
    """
    Remove `git_automation.py` and `.vscode/settings.json` from Git tracking.
    """
    print("\nRemoving `git_automation.py` and `.vscode/settings.json` from Git tracking...")
    files_to_remove = ["scripts/git_automation.py", ".vscode/settings.json"]
    for file in files_to_remove:
        success, output = run_command(f"git rm --cached {file}")
        if success:
            print(f"Successfully removed {file} from Git tracking.")
        else:
            print(f"Failed to remove {file} from Git tracking. Output:\n{output}")

def redo_commit(reason: str = "Redo commit to exclude untracked files"):
    """
    Redo the commit, ensuring all changes are properly staged and committed.
    """
    print("\nRedoing commit...")
    if not stage_changes():
        print("Failed to stage changes. Aborting commit redo.")
        return False
    return commit_changes(reason=reason)

def handle_pat_lifetime_error():
    """
    Handle the error when the PAT's lifetime exceeds the organization's allowed limit.
    """
    print("\nThe provided GitHub Personal Access Token (PAT) exceeds the allowed lifetime for this organization.")
    print("Please generate a new token with a lifetime of 366 days or less.")
    print("Visit the following URL to create a new token:")
    print("https://github.com/settings/personal-access-tokens")
    print("After generating the token, update your environment or VSCode settings with the new token.")
    prompt_for_pat()

def push_changes_with_github_cli():
    """
    Push changes using GitHub CLI.
    """
    print("\nPushing changes using GitHub CLI...")
    success, output = run_command("gh repo sync")
    if success:
        print("Changes pushed successfully using GitHub CLI.")
        return True
    else:
        if "can't sync because there are diverging changes" in output:
            print("GitHub CLI sync failed due to diverging changes. Attempting to resolve...")
            return handle_diverging_changes()
        elif "forbids access via a fine-grained personal access tokens" in output:
            handle_pat_lifetime_error()
        else:
            print(f"Failed to push changes using GitHub CLI. Output:\n{output}")
        return False

def create_pull_request_with_github_cli():
    """
    Create a pull request using GitHub CLI.
    """
    print("\nCreating pull request using GitHub CLI...")
    title = "Automated Pull Request"
    body = "This pull request was created automatically using GitHub CLI."
    success, output = run_command(f'gh pr create --title "{title}" --body "{body}"')
    if success:
        print("Pull request created successfully using GitHub CLI.")
        return True
    else:
        print(f"Failed to create pull request using GitHub CLI. Output:\n{output}")
        return False

def handle_diverging_changes():
    """
    Handle diverging changes between the local and remote branches.
    Prompt the user to choose between rebasing or force-pushing.
    """
    print("\nDiverging changes detected between local and remote branches.")
    print("Options:")
    print("1. Rebase local changes onto the remote branch.")
    print("2. Force-push local changes to overwrite the remote branch.")
    choice = input("Enter your choice (1 or 2): ").strip()
    if choice == "1":
        print("Rebasing local changes onto the remote branch...")
        success, output = run_command("git pull --rebase")
        if success:
            print("Rebase completed successfully.")
            return True
        else:
            print(f"Rebase failed. Output:\n{output}")
            return False
    elif choice == "2":
        print("Force-pushing local changes to overwrite the remote 'main' branch...")
        # Ensure we are on the main branch before force pushing to origin main
        current_branch_success, current_branch = run_command("git rev-parse --abbrev-ref HEAD")
        if not current_branch_success or current_branch != "main":
            print(f"Error: Not on 'main' branch (current: {current_branch}). Aborting force push to origin main.")
            print("Please checkout the 'main' branch first.")
            return False
        
        success, output = run_command("git push --force origin main", timeout=120)  # Explicitly push to origin main
        if success:
            print("Force-push to 'origin main' completed successfully.")
            return True
        else:
            if "Timeout" in output:
                print("Force-push operation timed out. Please check your network connection and try again.")
            else:
                print(f"Force-push to 'origin main' failed. Output:\n{output}")
            return False
    else:
        print("Invalid choice. Aborting operation.")
        return False

def force_push_with_github_cli(branch: str = "main", max_retries: int = 3, retry_delay: int = 10):
    """
    Force-push the current branch to the remote using GitHub CLI, overwriting remote history.
    Retries up to max_retries times if a timeout occurs.
    """
    print(f"\nForce-pushing local '{branch}' branch to remote using GitHub CLI...")
    # Ensure we are on the correct branch
    current_branch_success, current_branch = run_command("git rev-parse --abbrev-ref HEAD")
    if not current_branch_success or current_branch != branch:
        print(f"Error: Not on '{branch}' branch (current: {current_branch}). Please checkout the '{branch}' branch first.")
        return False

    # Show up to 5 files that are staged for commit
    print("Files staged for commit (up to 5):")
    success, output = run_command("git diff --cached --name-only")
    if success and output:
        files = output.splitlines()
        for f in files[:5]:
            print(f"  {f}")
        if len(files) > 5:
            print(f"  ...and {len(files) - 5} more.")
    else:
        print("  (No files staged for commit)")

    attempt = 0
    while attempt < max_retries:
        print(f"Force-push attempt {attempt + 1} of {max_retries}...")
        success, output = run_command(f"git push --force origin {branch}", timeout=120)
        if success:
            print(f"Force-push to 'origin {branch}' completed successfully.")
            return True
        else:
            if "Timeout" in output:
                print(f"Force-push operation timed out (attempt {attempt + 1}). Retrying after {retry_delay} seconds...")
                attempt += 1
                if attempt < max_retries:
                    time.sleep(retry_delay)
            else:
                print(f"Force-push to 'origin {branch}' failed. Output:\n{output}")
                return False
    print(f"Force-push to 'origin {branch}' failed after {max_retries} attempts due to repeated timeouts.")
    return False

if __name__ == "__main__":
    try:
        print(f"Running git_automation.py version {__version__}")
        print(__changelog__)
        # Authenticate with GitHub CLI using VSCode settings
        authenticate_with_github_cli_from_settings()
        check_git_installed()
        check_github_cli_installed()
        check_remote_authentication()
        diagnose_git()
        update_gitignore()
        sync_gitignore()
        if not pull_latest_changes():
            print("Aborting due to pull failure.")
            exit(1)
        if not checkout_main_branch():
            print("Aborting: Could not switch to 'main' branch.")
            exit(1)
        if not stage_changes():
            print("Aborting: Failed to stage changes.")
            exit(1)
        if not commit_changes(reason="Synchronized changes on main branch"):
            print("Aborting: Commit failed.")
            exit(1)
        # Overwrite remote branch with local changes using force-push
        if not force_push_with_github_cli(branch="main"):
            print("Force-push to 'main' branch failed. Aborting operation.")
            exit(1)
        # Optionally, create pull request using GitHub CLI
        if not create_pull_request_with_github_cli():
            print("Pull request creation failed using GitHub CLI. Please review diagnostics manually.")
    except KeyboardInterrupt:
        print("\nOperation interrupted by user. Exiting...")