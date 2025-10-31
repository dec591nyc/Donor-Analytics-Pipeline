#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Environment Variable Loader
Loads configuration from .env file for secure credential management
"""

import os
from pathlib import Path
from typing import Optional


class EnvLoader:
    """Load environment variables from .env file"""
    
    @staticmethod
    def load_env_file(env_path: Optional[Path] = None) -> dict:
        """
        Load environment variables from .env file
        
        Args:
            env_path: Path to .env file. If None, searches in:
                      1. Current directory
                      2. Project root
                      3. Parent directories (up to 3 levels)
        
        Returns:
            Dictionary of environment variables
        """
        if env_path is None:
            env_path = EnvLoader._find_env_file()
        
        if env_path is None or not env_path.exists():
            return {}
        
        env_vars = {}
        
        try:
            with open(env_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    
                    # Skip empty lines and comments
                    if not line or line.startswith('#'):
                        continue
                    
                    # Parse key=value
                    if '=' in line:
                        key, value = line.split('=', 1)
                        key = key.strip()
                        value = value.strip()
                        
                        # Remove quotes if present
                        if value.startswith('"') and value.endswith('"'):
                            value = value[1:-1]
                        elif value.startswith("'") and value.endswith("'"):
                            value = value[1:-1]
                        
                        env_vars[key] = value
                        
                        # Also set in os.environ for compatibility
                        os.environ[key] = value
            
            return env_vars
            
        except Exception as e:
            print(f"Warning: Could not load .env file: {e}")
            return {}
    
    @staticmethod
    def _find_env_file() -> Optional[Path]:
        """Search for .env file in common locations"""
        
        # Start from current working directory
        current = Path.cwd()
        
        # Check current directory
        env_path = current / '.env'
        if env_path.exists():
            return env_path
        
        # Check parent directories (up to 3 levels)
        for _ in range(3):
            current = current.parent
            env_path = current / '.env'
            if env_path.exists():
                return env_path
        
        # Check script directory
        script_dir = Path(__file__).parent
        env_path = script_dir / '.env'
        if env_path.exists():
            return env_path
        
        # Check project root (donations_pipeline_Claude parent)
        project_root = script_dir.parent
        env_path = project_root / '.env'
        if env_path.exists():
            return env_path
        
        return None
    
    @staticmethod
    def get_env(key: str, default: str = "") -> str:
        """
        Get environment variable with fallback to .env file
        
        Args:
            key: Environment variable name
            default: Default value if not found
        
        Returns:
            Environment variable value or default
        """
        # First check OS environment
        value = os.environ.get(key)
        if value is not None:
            return value
        
        # If not in OS env, try loading from .env
        env_vars = EnvLoader.load_env_file()
        return env_vars.get(key, default)
    
    @staticmethod
    def get_email_config() -> dict:
        """
        Get email configuration from environment
        
        Returns:
            Dictionary with email configuration
        """
        return {
            'enabled': EnvLoader.get_env('EMAIL_ENABLED', 'false').lower() == 'true',
            'to_addresses': [
                e.strip() 
                for e in EnvLoader.get_env('EMAIL_TO', '').split(',') 
                if e.strip()
            ],
            'cc_addresses': [
                e.strip() 
                for e in EnvLoader.get_env('EMAIL_CC', '').split(',') 
                if e.strip()
            ],
            'from_address': EnvLoader.get_env('EMAIL_FROM', ''),
            'smtp_server': EnvLoader.get_env('SMTP_SERVER', 'smtp.gmail.com'),
            'smtp_port': int(EnvLoader.get_env('SMTP_PORT', '587')),
            'smtp_username': EnvLoader.get_env('SMTP_USERNAME', ''),
            'smtp_password': EnvLoader.get_env('SMTP_PASSWORD', ''),
        }


# Convenience function for direct import
def load_env(env_path: Optional[Path] = None) -> dict:
    """Load environment variables from .env file"""
    return EnvLoader.load_env_file(env_path)


def get_env(key: str, default: str = "") -> str:
    """Get environment variable"""
    return EnvLoader.get_env(key, default)