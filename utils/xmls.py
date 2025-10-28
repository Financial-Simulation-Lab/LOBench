"""
Created on : 2024-07-02
Created by : Ren Junji, Wu Yue
Updated by : Mythezone
Email      : ---
FileName   : ~/simlob-refined/utils/update_xml.py
Description: Updating XML configuration file.
Updated    : 
    !   v1.0: 
           * Adding two args to the update_xml function, makes the function more flexible.
                - call_index: It's essential useful when using multithreading to generate data by TheSimulator.
                - xml_path: Custom selection of the Template .xml file path.
            * Change the parameter of 'name' into function which makes the function reusable.
            
    !   v1.1:
            * Add xml_path, csv_path and tmp_xml_path to the parameters of update_xml function.
                - csv_path: the output path of the final data csv file path.
                - xml_path: the output of the updated xml file path.
                - tmp_xml_path : the template of the xml file of PGPS.
            
Todo       : 
"""


import xml.etree.ElementTree as ET

import sys
import os
import tempfile
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    
from utils.file import create_or_pass

# default path
# default_output_folder = "data/xml_data/"
# default_simdata_folder = "data/simu_data/"
from config import ConfigManager as CM 

cm = CM()

calibrate_template_xml_file = os.path.join(cm.folders.xml,"template","calibrate_template.xml")
train_template_xml_file = os.path.join(cm.folders.xml,"template","train_data_template.xml")

default_template_xml_file = os.path.join(cm.folders.xml,"template","default_data_template.xml")
random_seed = cm.random_seed 

def update_xml(params,
               call_index="0000",
               template="calibrate",
               xml_path=cm.folders.xml,
               csv_path=cm.folders.csv,
               name='translob',
               tmp=False,
               pd=False,
               debug=False) :
    """Update the template xml file.
    If tmp is true, the middian xml file will generated in the memory and will be delete afterward.
    If pd is true, the middian csv file will generated in the memory and will be delete afterward.

    Args:
        params (array): The parameters in PGPS model.
        rand_seed (int): The random seed set in TheSimulator.
        call_index (str, optional): used for multithreading. Defaults to "0000".
        xml_path (str, optional): the output of the updated xml file path.
        csv_path (str, optional): the output path of the final data csv file path.
        tmp_xml_path(str, optional):  The path of the template xml configuration file.
        Defaults to "./".
        name (str, optional): The name of the Model. Defaults to 'translob'.
        tmp (bool, optional): If using the tmp file and return the tmp file path.
        pd (bool, optional): If out put the dataframe as return.
        debug (bool, optional): If output the messages to console.
    Returns:
        results (List<str>[2]): [tmp_config_file_path, tmp_csv_file_path], default: [None,None]
    """
    
    config_output_file = os.path.join(xml_path,'xml_'+name,call_index+'.xml')
    create_or_pass(config_output_file)
    
    
    data_output_filename = os.path.join(csv_path,'csv_'+name,call_index+'.csv')
    create_or_pass(data_output_filename)
    
    # with open(tmp_xml_path, 'rb') as f:
    if template == 'calibrate':
        template_xml_file = calibrate_template_xml_file
    elif template == 'train':
        template_xml_file = train_template_xml_file
    else:
        template_xml_file = default_template_xml_file
    
    tree = ET.parse(template_xml_file)
    if len(params)==1:
        params = params[0]
    
    root = tree.getroot()
    root.set('lambda_init', str(params[0])) 
    root.set('c_lambda', str(params[1])) 
    
    root[2][0].set('frequency', str(params[2])) 
    root[3][0].set('frequency', str(params[3])) 
    
    root.set('increment', str(params[4])) 
    root[2][0].set('delta', str(params[5])) 
    
    # * This outputFile is where the Simulated data well be put.
    
    
    root.set('random_seed', str(random_seed))
    
    result =[None,None]
    
    if pd:
        f_csv= tempfile.NamedTemporaryFile(delete=False,suffix='.csv') 
        root[4].set('outputFile', f_csv.name)
        if debug:
            print("The CSV visual file has been saved.")
        result[1]=f_csv.name
    else:
        root[4].set('outputFile', data_output_filename)
        result[1]=data_output_filename
    
    
    
    if not tmp:
        # tree.write(config_output_file, 'UTF-8')
        result[0]=config_output_file
    else:
        f_xml = tempfile.NamedTemporaryFile(delete=False,suffix='.xml') 
        result[0]=f_xml.name
            
    tree.write(result[0],encoding='UTF-8',xml_declaration=True)
    return result
            
if __name__ == "__main__":
    # 测试 update_xml 方法
    para=[193.32,3.47,0.2962,0.01853,0.0015,0.02909]
    # tmp_file_name=update_xml(para,rand_seed=1234,tmp=True)
    # print(tmp_file_name)
    # tmp_xml_file,tmp_csv_file = update_xml(para,xml_path='utils/test.xml',tmp=True,pd=True)
    # print(tmp_xml_file,tmp_csv_file)
    