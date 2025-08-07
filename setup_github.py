#!/usr/bin/env python3
"""
GitHub Repository Setup Script
Automates the process of creating a private GitHub repository and pushing the project.
"""

import os
import subprocess
import sys
from pathlib import Path

def run_command(command, cwd=None):
    """Run a shell command and return the result"""
    try:
        result = subprocess.run(command, shell=True, cwd=cwd, capture_output=True, text=True)
        return result.returncode == 0, result.stdout, result.stderr
    except Exception as e:
        return False, "", str(e)

def check_git_installed():
    """Check if git is installed"""
    success, stdout, stderr = run_command("git --version")
    if not success:
        print("❌ Git is not installed. Please install Git first.")
        return False
    print(f"✅ Git found: {stdout.strip()}")
    return True

def check_github_cli():
    """Check if GitHub CLI is installed"""
    success, stdout, stderr = run_command("gh --version")
    if not success:
        print("⚠️  GitHub CLI not found. You'll need to create the repository manually.")
        return False
    print(f"✅ GitHub CLI found: {stdout.split()[2]}")
    return True

def initialize_git_repo():
    """Initialize git repository"""
    if os.path.exists(".git"):
        print("✅ Git repository already initialized")
        return True
    
    success, stdout, stderr = run_command("git init")
    if not success:
        print(f"❌ Failed to initialize git repository: {stderr}")
        return False
    
    print("✅ Git repository initialized")
    return True

def create_github_repo(repo_name, description, private=True):
    """Create GitHub repository using GitHub CLI"""
    visibility = "--private" if private else "--public"
    command = f'gh repo create "{repo_name}" --description "{description}" {visibility} --source=. --remote=origin --push'
    
    success, stdout, stderr = run_command(command)
    if not success:
        print(f"❌ Failed to create GitHub repository: {stderr}")
        print("Please create the repository manually on GitHub and then run:")
        print(f"git remote add origin https://github.com/YOUR_USERNAME/{repo_name}.git")
        return False
    
    print("✅ GitHub repository created and pushed")
    return True

def add_and_commit_files():
    """Add all files and make initial commit"""
    # Add all files
    success, stdout, stderr = run_command("git add .")
    if not success:
        print(f"❌ Failed to add files: {stderr}")
        return False
    
    # Check if there are files to commit
    success, stdout, stderr = run_command("git status --porcelain")
    if not stdout.strip():
        print("✅ No changes to commit")
        return True
    
    # Make initial commit
    success, stdout, stderr = run_command('git commit -m "Initial commit: Real-time sign language translation project"')
    if not success:
        print(f"❌ Failed to commit files: {stderr}")
        return False
    
    print("✅ Files committed successfully")
    return True

def push_to_github():
    """Push to GitHub"""
    success, stdout, stderr = run_command("git push -u origin main")
    if not success:
        # Try master branch if main doesn't exist
        success, stdout, stderr = run_command("git push -u origin master")
        if not success:
            print(f"❌ Failed to push to GitHub: {stderr}")
            return False
    
    print("✅ Code pushed to GitHub successfully")
    return True

def main():
    print("🚀 GitHub Repository Setup for Real-Time Sign Language Translation")
    print("=" * 60)
    
    # Check prerequisites
    if not check_git_installed():
        sys.exit(1)
    
    has_github_cli = check_github_cli()
    
    # Get repository details
    repo_name = input("Enter repository name (default: real-time-sign-language-translation): ").strip()
    if not repo_name:
        repo_name = "real-time-sign-language-translation"
    
    description = input("Enter repository description (default: Real-time sign language to speech translation using deep learning): ").strip()
    if not description:
        description = "Real-time sign language to speech translation using deep learning"
    
    private_input = input("Make repository private? (y/n, default: y): ").strip().lower()
    private = private_input != 'n'
    
    print(f"\n📋 Repository Details:")
    print(f"   Name: {repo_name}")
    print(f"   Description: {description}")
    print(f"   Private: {private}")
    
    confirm = input("\nProceed with setup? (y/n): ").strip().lower()
    if confirm != 'y':
        print("Setup cancelled.")
        return
    
    print("\n🔄 Setting up repository...")
    
    # Initialize git repository
    if not initialize_git_repo():
        sys.exit(1)
    
    # Add and commit files
    if not add_and_commit_files():
        sys.exit(1)
    
    # Create GitHub repository
    if has_github_cli:
        if not create_github_repo(repo_name, description, private):
            print("\n📝 Manual Setup Instructions:")
            print("1. Go to https://github.com/new")
            print(f"2. Create a new repository named '{repo_name}'")
            print(f"3. Description: '{description}'")
            print(f"4. Make it {'private' if private else 'public'}")
            print("5. Don't initialize with README, .gitignore, or license")
            print("6. Run the following commands:")
            print(f"   git remote add origin https://github.com/YOUR_USERNAME/{repo_name}.git")
            print("   git push -u origin main")
    else:
        print("\n📝 Manual Setup Instructions:")
        print("1. Go to https://github.com/new")
        print(f"2. Create a new repository named '{repo_name}'")
        print(f"3. Description: '{description}'")
        print(f"4. Make it {'private' if private else 'public'}")
        print("5. Don't initialize with README, .gitignore, or license")
        print("6. Run the following commands:")
        print(f"   git remote add origin https://github.com/YOUR_USERNAME/{repo_name}.git")
        print("   git push -u origin main")
    
    print("\n✅ Setup complete!")
    print(f"🌐 Your repository will be available at: https://github.com/YOUR_USERNAME/{repo_name}")

if __name__ == "__main__":
    main()
