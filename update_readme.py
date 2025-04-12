#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to update only the Installation section of the README.md file.
"""

import re

def update_installation_section():
    """Update the Installation section of README.md with requirements.txt instructions."""
    readme_path = "README.md"
    
    # Read the current README content
    with open(readme_path, 'r') as file:
        readme_content = file.read()
    
    # Define the new installation section
    new_installation = """## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/tornado-damage-recognition.git
cd tornado-damage-recognition
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

Alternatively, install individual packages:
```bash
pip install torch torchvision numpy pandas matplotlib seaborn scikit-learn tqdm pillow opencv-python tensorboard thop torchsummary kornia
```
"""
    
    # Use regex to find and replace the Installation section
    # Look for the section starting with "## Installation" and ending before the next section heading
    pattern = r'## Installation.*?(?=\n## )'
    updated_readme = re.sub(pattern, new_installation, readme_content, flags=re.DOTALL)
    
    # Write the updated content back to README.md
    with open(readme_path, 'w') as file:
        file.write(updated_readme)
    
    print("README.md Installation section updated successfully!")

if __name__ == "__main__":
    update_installation_section()