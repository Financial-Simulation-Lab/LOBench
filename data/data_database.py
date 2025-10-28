"""
Created on : 2024-07-13
Created by : Mythezone
Updated by : Mythezone
Email      : mythezone@gmail.com
FileName   : ~/project/SimLOB/simlob-refined/calibrate/dataset/database.py
Description: Save the data into mongo database for later usage.
---
Updated    : 
---
Todo       : 
"""


# Insert the path into sys.path for importing.
import random
import sys
import os

from tqdm import tqdm
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from typing import Optional,List,Tuple,Dict,Union,Protocol


import pymongo
import hashlib
from pymongo import MongoClient
import numpy as np 
import pandas as pd 

from utils.single import SingletonMeta
from utils.file import compute_array_hash, save_pd_to_hashed_folder

from config import ConfigManager as CM
from datetime import datetime 
from utils.logger import setup_logger



class RawDataManager(metaclass=SingletonMeta):
    cm = CM()
    mongo_uri = cm.simulation_database.url
    db_name = cm.simulation_database.name
    collection_name = cm.simulation_database.collection
    raw_csv_folder = cm.folders.raw
    database_log = cm.logs.database
    logger = setup_logger(4,"RawDataManager")
    
    def __init__(self,debug=False):
        self.client = MongoClient(self.mongo_uri)
        self.db = self.client[self.db_name]
        self.collection = self.db[self.collection_name]
        # self.base_raw_csv = ""
        self.debug = debug
        

    def _hash_params(self, params: List )-> str :
        """
        Generate a hash for the given parameters.
        """
        return compute_array_hash(params)

    def insert(self, params: List , file_path:str ):
        """
        Insert a new record with parameters and file path.
        """
        hash_key = self._hash_params(params)
        
        record = {
            "_id": hash_key,
            "params": params,
            "file_path": file_path,
            "length":0,
            
        }
        try:
            self.collection.insert_one(record)
            if self.debug:
                print(f"Record inserted with hash: {hash_key}")
            else:
                self.logger.info(f"Record inserted with hash: {hash_key}")
        except pymongo.errors.DuplicateKeyError:
            if self.debug:
                print(f"Record with hash {hash_key} already exists.")
            else:
                self.logger.error(f"Record with hash {hash_key} already exists.")
            
    def insert_pd(self,params:List, df:pd.DataFrame):
        """
        Insert a new record in DataFrame format with parameters.
        """        
        
        if isinstance(params, np.ndarray):
            params = params.tolist()
        hash_key = compute_array_hash(params)
        
        # Test if the params exists in the database.
        existing_record = self.collection.find_one({"_id": hash_key})
        if existing_record:
            if self.debug:
                print(f"Record with hash {hash_key} already exists. Not inserting.")
            else:
                self.logger.warning(f"{hash_key} is exist.")
                
            return
        
        file_path = save_pd_to_hashed_folder(params, df, base_folder=self.raw_csv_folder )
        
        l = df.shape[0]
        
        current_time = datetime.now()
        
        record = {
            "_id": hash_key,
            "params": params,
            "file_path": file_path,
            "length":l,
            "create_at": current_time,
            "update_at": current_time,
            "random_seed": self.cm.random_seed,
            # "parameter_accuracy": self.cm.accuracy,
            "has_raw_data": True 
        }

        try:
            self.collection.insert_one(record)
            if self.debug:
                print(f"Record inserted with hash: {hash_key}")
            else:
                self.logger.info(f"{hash_key} generated and saved.")
        except Exception as e :
            if self.debug:
                print(f"Record with hash {hash_key} insert failed: \nError: {e}")
            else:
                self.logger.error(f"{hash_key} Error: {e}")

    def update_file_path(self, params, new_file_path):
        """
        Update the file path for the given parameters.
        """
        hash_key = self._hash_params(params)
        result = self.collection.update_one(
            {"_id": hash_key},
            {"$set": {"file_path": new_file_path}}
        )
        if result.matched_count:
            print(f"Record with hash {hash_key} updated.")
        else:
            print(f"No record found with hash {hash_key}.")
            
    def update_by_id(self, hash_key, dct):
        result = self.collection.update_one(
            {"_id": hash_key},
            {"$set": dct}
        )
        if result.matched_count:
            print(f"Record with hash {hash_key} updated.")
        else:
            print(f"No record found with hash {hash_key}.")
        
        

    def delete(self, params):
        """
        Delete the record for the given parameters.
        """
        hash_key = self._hash_params(params)
        result = self.collection.delete_one({"_id": hash_key})
        if result.deleted_count:
            print(f"Record with hash {hash_key} deleted.")
        else:
            print(f"No record found with hash {hash_key}.")

    def select(self, params):
        """
        Select and return the file path for the given parameters.
        """
        hash_key = self._hash_params(params)
        record = self.collection.find_one({"_id": hash_key})
        if record:
            return record
        else:
            if self.debug:
                print(f"No record found with hash {hash_key}.")
            return None
        
    def get_train_data(self,num=100, seed = None):
        """
        Get the train data from the database.
        """
        train_length = 10000
        
        if seed is not None:
            random.seed(seed)
        
        pipeline = [
            {"$match": {"length": {"$gt": train_length}, "has_negative": False}},
            {"$sample": {"size": num}}  
        ]
        
        records = self.collection.aggregate(pipeline)
        
        return list(records)
        
    
    
    def has_negative(self, record):
        """
        Check if the record has negative values.
        """
        file_path = record["file_path"]
        has_negative = record.get("has_negative", None)
        if has_negative is not None:
            return has_negative
        
        df = pd.read_csv(file_path)
        if (df < 0).any().any():
            has_negative = True 
        else:
            has_negative = False 
        self.update_by_id(record["_id"], {"has_negative": has_negative})
        return has_negative
    
    def update_all(self, func: callable):
        """
        Update all records that do not have the 'has_negative' attribute.
        """
        # 查询所有没有 'has_negative' 属性的记录，并将它们存储在列表中
        records = list(self.collection.find({"has_negative": {"$exists": False}}))
        
        # 获取记录的数量
        total_records = len(records)
        
        # 使用 tqdm 添加进度条
        with tqdm(total=total_records, desc="Updating records (0/{})".format(total_records)) as pbar:
            for i, record in enumerate(records, 1):
                func(record)
                pbar.set_description("Updating records ({}/{})".format(i, total_records))
                pbar.update(1)
            
    def get_non_negative_records(self, records):
        """
        Check if the record has negative values.
        """
        non_negative_records = []
        for record in records:
            file_path = record["file_path"]
            df = pd.read_csv(file_path)
            if (df < 0).any().any():
                
                continue 
            else:
                # print(f"Record with hash {record['_id']} has no negative values.")
                non_negative_records.append(record)
        return non_negative_records
    
          
    

# Usage example
if __name__ == "__main__":
    # manager = RawDataManager()
    rdm = RawDataManager()
    # params = {"param1": "value1", "param2": "value2"}
    # file_path = "/path/to/csv/file.csv"

    # # Insert
    # manager.insert(params, file_path)

    # # Select
    # selected_path = manager.select(params)
    # print(f"Selected file path: {selected_path}")

    # # Update
    # new_file_path = "/new/path/to/csv/file.csv"
    # manager.update(params, new_file_path)

    # # Delete
    # manager.delete(params)
    
    # config = CM()

    # db_name = config.database.name
    # print(db_name)
    
    # df = pd.DataFrame([1,2,3,4,5])
    # test_params = [1,2,3,4,5,6]
    # rdm.insert_pd(test_params,df)
    # test_np_params = np.array([1,2,3,4,5,6])
    # rdm.insert_pd(test_np_params,df)
    params = [
    193.32,
    3.47,
    0.2962,
    0.01853,
    0.0015,
    0.02909
  ]
    # rdm.select_or_generate(params)
    # result = rdm.get_train_data(10)
    # print(len(result))
    # for r in result:
    #     print(r["has_negative"])
    # non_neg = rdm.get_non_negative_records(result)
    # print(len(non_neg))
    # print(rdm.check_negative(result[0]))
    
    rdm.update_all_if_has_negative()
