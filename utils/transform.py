"""
Created on : 2024-07-08
Created by : Mythezone
Updated by : Mythezone
Email      : mythezone@gmail.com
FileName   : ~/project/SimLOB/simlob-refined/utils/transform.py
Description: Transform the LOB data into other forms of data.
---
Updated    : 
---
Todo       : 
"""


# Insert the path into sys.path for importing.
import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


from typing import Optional,List,Tuple,Dict,Union,Protocol



