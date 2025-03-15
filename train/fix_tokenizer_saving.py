#!/usr/bin/env python
"""
Script to fix tokenizer saving in the training pipeline.

This script:
1. Updates the trainer.py file to ensure proper tokenizer saving
2. Adds a function to verify that the tokenizer is properly saved
3. Provides a patch that can be applied to existing trained models

Usage:
    python fix_tokenizer_saving.py --mode [check|fix|patch] [--model_path path/to/model]
"""

import os
import sys
import argparse
import shutil
import json
from transformers import AutoTokenizer
from typing import Dict, Optional, List, Any


def parse_args():
    parser = argparse.ArgumentParser(description="Fix tokenizer saving in training pipeline")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["check", "fix", "patch"],
        default="check",
        help="Mode: check (verify trainer.py), fix (update trainer.py), patch (fix existing model)"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        help="Path to model directory (required for patch mode)"
    )
    parser.add_argument(
        "--trainer_path",
        type=str,
        default="/share/pi/nigam/users/calebwin/med-s1/train/trainer.py",
        help="Path to trainer.py file"
    )
    parser.add_argument(
        "--source_model",
        type=str,
        help="Path to source model for copying tokenizer config (for patch mode)"
    )
    return parser.parse_args()


def check_trainer_file(trainer_path: str) -> Dict[str, Any]:
    """Check if trainer.py has proper tokenizer saving."""
    with open(trainer_path, 'r') as f:
        content = f.read()
    
    # Check for tokenizer saving line
    has_tokenizer_save = "self.tokenizer.save_pretrained" in content
    
    # Check for additional tokenizer configuration saving
    has_save_with_config = "save_pretrained(save_dir, legacy_format=False" in content
    has_save_with_kwargs = "save_pretrained(save_dir, **kwargs" in content
    
    # Check for chat template preservation
    preserves_chat_template = "chat_template" in content and "tokenizer" in content
    
    return {
        "has_tokenizer_save": has_tokenizer_save,
        "has_save_with_config": has_save_with_config,
        "has_save_with_kwargs": has_save_with_kwargs,
        "preserves_chat_template": preserves_chat_template,
        "needs_fixing": not (has_save_with_config or has_save_with_kwargs or preserves_chat_template)
    }


def fix_trainer_file(trainer_path: str) -> bool:
    """Update trainer.py to ensure proper tokenizer saving."""
    # First make a backup
    backup_path = f"{trainer_path}.bak"
    shutil.copy2(trainer_path, backup_path)
    print(f"Created backup at {backup_path}")
    
    with open(trainer_path, 'r') as f:
        content = f.readlines()
    
    # Find the tokenizer save line
    tokenizer_save_line = -1
    for i, line in enumerate(content):
        if "self.tokenizer.save_pretrained" in line:
            tokenizer_save_line = i
            break
    
    if tokenizer_save_line == -1:
        print("Could not find tokenizer save line in trainer.py")
        return False
    
    # Replace the simple save with a more comprehensive one
    original_line = content[tokenizer_save_line]
    indentation = original_line[:len(original_line) - len(original_line.lstrip())]
    
    # Create the new lines with proper indentation
    new_lines = [
        f"{indentation}# Save tokenizer with all configuration\n",
        f"{indentation}try:\n",
        f"{indentation}    # Preserve chat template and other configurations\n",
        f"{indentation}    self.tokenizer.save_pretrained(\n",
        f"{indentation}        save_dir,\n",
        f"{indentation}        legacy_format=False,\n",
        f"{indentation}        save_config=True\n",
        f"{indentation}    )\n",
        f"{indentation}    print(f\"Saved tokenizer with full configuration to {{save_dir}}\")\n",
        f"{indentation}except Exception as e:\n",
        f"{indentation}    print(f\"Error saving tokenizer with full configuration: {{e}}\")\n",
        f"{indentation}    # Fallback to basic save\n",
        f"{indentation}    self.tokenizer.save_pretrained(save_dir)\n"
    ]
    
    # Replace the line
    content[tokenizer_save_line] = "".join(new_lines)
    
    # Write the updated content
    with open(trainer_path, 'w') as f:
        f.writelines(content)
    
    print(f"Updated {trainer_path} with improved tokenizer saving")
    return True


def verify_tokenizer_saving(model_path: str) -> Dict[str, Any]:
    """Verify if a model's tokenizer has all necessary configurations."""
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        
        # Check for essential attributes
        has_chat_template = hasattr(tokenizer, 'chat_template') and tokenizer.chat_template is not None
        has_special_tokens = hasattr(tokenizer, 'special_tokens_map') and tokenizer.special_tokens_map is not None
        
        # Check tokenizer config file
        tokenizer_config_path = os.path.join(model_path, 'tokenizer_config.json')
        has_config_file = os.path.exists(tokenizer_config_path)
        
        config_contains_chat_template = False
        if has_config_file:
            with open(tokenizer_config_path, 'r') as f:
                config = json.load(f)
                config_contains_chat_template = 'chat_template' in config
        
        return {
            "has_chat_template": has_chat_template,
            "has_special_tokens": has_special_tokens,
            "has_config_file": has_config_file,
            "config_contains_chat_template": config_contains_chat_template,
            "needs_patching": not (has_chat_template and config_contains_chat_template)
        }
    except Exception as e:
        print(f"Error verifying tokenizer: {e}")
        return {
            "error": str(e),
            "needs_patching": True
        }


def patch_model_tokenizer(model_path: str, source_model: Optional[str] = None) -> bool:
    """Patch an existing model's tokenizer with proper configuration."""
    if not os.path.exists(model_path):
        print(f"Model path does not exist: {model_path}")
        return False
    
    # First verify the current state
    verification = verify_tokenizer_saving(model_path)
    if not verification.get("needs_patching", True):
        print(f"Model at {model_path} does not need patching")
        return True
    
    # Determine source model for tokenizer config
    if source_model is None:
        # Try to infer from model name
        model_name = os.path.basename(model_path)
        if "llama" in model_name.lower():
            source_model = "meta-llama/Llama-3.1-8B-Instruct"
        elif "huatuo" in model_name.lower():
            source_model = "FreedomIntelligence/HuatuoGPT-o1-8B"
        else:
            print("Could not infer source model, please specify with --source_model")
            return False
    
    print(f"Using source model: {source_model}")
    
    try:
        # Load source tokenizer
        source_tokenizer = AutoTokenizer.from_pretrained(source_model, trust_remote_code=True)
        
        # Load target tokenizer
        target_tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        
        # Copy chat template if available
        if hasattr(source_tokenizer, 'chat_template') and source_tokenizer.chat_template is not None:
            target_tokenizer.chat_template = source_tokenizer.chat_template
            print(f"Copied chat template from {source_model}")
        
        # Save the updated tokenizer
        target_tokenizer.save_pretrained(
            model_path,
            legacy_format=False,
            save_config=True
        )
        
        print(f"Successfully patched tokenizer for {model_path}")
        
        # Verify the patch
        verification_after = verify_tokenizer_saving(model_path)
        if verification_after.get("needs_patching", True):
            print("Warning: Model still needs patching after attempted fix")
            return False
        
        return True
    except Exception as e:
        print(f"Error patching tokenizer: {e}")
        return False


def main():
    args = parse_args()
    
    if args.mode == "check":
        if not os.path.exists(args.trainer_path):
            print(f"Trainer file not found: {args.trainer_path}")
            sys.exit(1)
        
        check_result = check_trainer_file(args.trainer_path)
        print("\nTrainer.py Tokenizer Saving Check:")
        for key, value in check_result.items():
            print(f"  {key}: {value}")
        
        if check_result["needs_fixing"]:
            print("\nRecommendation: Run with --mode=fix to update trainer.py")
        else:
            print("\nTrainer.py appears to have proper tokenizer saving")
    
    elif args.mode == "fix":
        if not os.path.exists(args.trainer_path):
            print(f"Trainer file not found: {args.trainer_path}")
            sys.exit(1)
        
        success = fix_trainer_file(args.trainer_path)
        if success:
            print("\nSuccessfully updated trainer.py")
            print("New models trained with this version will have proper tokenizer configuration")
        else:
            print("\nFailed to update trainer.py")
    
    elif args.mode == "patch":
        if not args.model_path:
            print("--model_path is required for patch mode")
            sys.exit(1)
        
        success = patch_model_tokenizer(args.model_path, args.source_model)
        if success:
            print("\nSuccessfully patched model tokenizer")
        else:
            print("\nFailed to patch model tokenizer")


if __name__ == "__main__":
    main()