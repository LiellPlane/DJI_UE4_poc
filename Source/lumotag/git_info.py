"""
Very simple Git information module.
Gets latest commit name and sequence number from local Git repository.
Returns GitInfo Pydantic class or None if anything goes wrong - no exceptions raised.
"""

import os
import subprocess
from typing import Optional

from git import Repo
from pydantic import BaseModel


class GitInfo(BaseModel):
    """Git repository information."""
    commit_message: str
    commit_count: int
    short_hash: str
    
    def get_version_string(self) -> str:
        """Get a formatted version string."""
        return f"Commit {self.commit_count}: {self.commit_message}"


def get_git_info() -> Optional[GitInfo]:
    """
    Get latest commit information from Git repository.
    
    Returns:
        GitInfo: Pydantic model with commit info, or None if any error occurs
    """
    try:
        # Start from current directory and drill up to find .git
        current_path = os.path.abspath('.')
        
        # Keep drilling up until we find .git or reach filesystem root
        while True:
            if os.path.exists(os.path.join(current_path, '.git')):
                # Found .git directory
                break
                
            parent_path = os.path.dirname(current_path)
            if parent_path == current_path:  # Reached filesystem root
                return None
                
            current_path = parent_path
        
        # Initialize repo and get info
        repo = Repo(current_path)
        if repo.bare:
            return None
            
        latest_commit = repo.head.commit
        
        # Use fast git command for commit count instead of iterating
        try:
            commit_count = int(subprocess.check_output(
                ['git', 'rev-list', '--count', 'HEAD'], 
                cwd=current_path,
                stderr=subprocess.DEVNULL
            ).decode('ascii').strip())
        except (subprocess.CalledProcessError, ValueError):
            # Fallback to slow method if git command fails
            commit_count = sum(1 for _ in repo.iter_commits())
        
        return GitInfo(
            commit_message=latest_commit.message.strip(),
            commit_count=commit_count,
            short_hash=latest_commit.hexsha[:7]
        )
        
    except:
        return None


def get_version_string() -> str:
    """
    Get a simple version string for display.
    
    Returns:
        str: "Commit #123: Fix camera bug (a1b2c3d)" or "No Git info available"
    """
    info = get_git_info()
    if info is None:
        return "No Git info available"
    
    return info.get_version_string()


def main():
    """Test function to verify the module works correctly."""
    print("Testing git_info module...")
    print("-" * 50)
    
    # Test get_git_info()
    info = get_git_info()
    if info is not None:
        print("✅ Git repository found!")
        print(f"   Commit message: {info.commit_message}")
        print(f"   Commit count: {info.commit_count}")
        print(f"   Short hash: {info.short_hash}")
        print()
        print(f"   Version string: {info.get_version_string()}")
    else:
        print("❌ No Git repository found")
    
    print()
    print("-" * 50)
    
    # Test get_version_string()
    version = get_version_string()
    print(f"get_version_string() result: {version}")


if __name__ == "__main__":
    main()
