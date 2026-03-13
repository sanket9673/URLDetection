import pandas as pd
import numpy as np

# Revert my bad change from feature builder
with open("src/feature_engineering/feature_builder.py", "r") as f:
    code = f.read()

# I will replace the standardization block with something smarter or just revert it
import re
code = re.sub(r"\s*# STANDARDIZE URLs to avoid dataset bias.*?\n\s*logger\.info\(f\"Cleaned malformed", "\n        logger.info(f\"Cleaned malformed", code, flags=re.DOTALL)

with open("src/feature_engineering/feature_builder.py", "w") as f:
    f.write(code)

