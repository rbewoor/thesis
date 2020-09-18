## -----------------------------------------------------------------------------
### Goal: Upload files from local (laptop) folder to specific Google drive folder-
##        Use the CL parameters to specifiy which files
##        NOTES:
##              1) The os.listdir on the local machine is used and the start and end indexex slice the output of it.
##                 Checked, its supposed to be deterministic!
##              2) Uses the oauth authetication method. So download the credentials file first. Token will be created based on this file.
##              3) The source folder on local machine is hardcoded
##              4) The target folder on Google Drive is specified using the folder-id (end of the url when you open that folder). This is also hardcoded
##              5) When lot of files to upload, found can run the script in different shells with the suitable start and end indices to speed it all up.
##
## -----------------------------------------------------------------------------
## Command line arguments:
## 1) -sidx   :start index for the os.listdir() output array slicing
## 2) -eidx   :end   index for the os.listdir() output array slicing
## 3) -sf     :status frequency paramter
##             print a statement showing how many processed till then
## -----------------------------------------------------------------------------
## Usage example:
##    python3 gdrive_upload_CocoTrain2017_1.py -sidx 5000 -eidx 10000 -sf 10
##    The above command, will upload the files in os.listdir(source_folder)[sidx:eidx] array. Will print a status output every 10th upload
## -----------------------------------------------------------------------------

from __future__ import print_function

import pickle
import os
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request

import argparse
import sys
import datetime
import time

argparser = argparse.ArgumentParser(
    description='upload files from a location to google drive folder')

argparser.add_argument(
    '-sidx',
    '--startindex',
    type=int,
    help='starting position in array')

argparser.add_argument(
    '-eidx',
    '--endindex',
    help='ending position in array')

argparser.add_argument(
    '-sf',
    '--statusfrequency',
    type=int,
    help='show status after how many inserts to neo4j')

def my_upload_file(_drive_service, _source_file_with_full_path, _gdrive_target_folder_id):
    try:
        # Call the Drive v3 API to upload a file
        ## reference: https://developers.google.com/drive/api/v3/manage-uploads
        file_metadata = {'name': os.path.basename(_source_file_with_full_path),
                         #'parents': [{'id': to_folder_id, 'kind': 'drive#file'}]# 'kind': 'drive#childList'}]
                         'parents': [_gdrive_target_folder_id]
                        }
        media = MediaFileUpload(_source_file_with_full_path, mimetype='image/jpeg')
        file = _drive_service.files().create(body=file_metadata,
                                        media_body=media,
                                        fields='id').execute()
        #print(f"File ID: {file.get('id')}")
        return 1 # all good
    except:
        return 0 # problem

def _main_(args):
    # process command line arguments
    start_idx    = args.startindex           # -sidx parameter
    end_idx      = args.endindex             # -eidx parameter
    print_status_freq = args.statusfrequency # -sf parameter

    try:
        start_idx = int(start_idx)
        end_idx = int(end_idx)
        print_status_freq =  int(print_status_freq)
    except Exception as e:
        print(f"Problem with CLI arguments: {e}\nNothing processed.")
        return

    # If modifying these scopes, delete the file token.pickle.
    SCOPES = ['https://www.googleapis.com/auth/drive']

    creds = None
    gdrive_creds_json_loc = r'/home/rohit/PyWDUbuntu/thesis/gdrive_pil_20200701/credentials_pil_thesis_work_oauth_client_id_20200916.json'

    CREDS_PICKLE_TOP_FOLDER = r'/home/rohit/PyWDUbuntu/thesis/gdrive_pil_20200701/'
    creds_pickle_file_loc = r'/home/rohit/PyWDUbuntu/thesis/gdrive_pil_20200701/token_pil_thesis_work_oauth_client_id_20200916.pickle'
    # The file token.pickle stores the user's access and refresh tokens, and is
    # created automatically when the authorization flow completes for the first
    # time.
    if os.path.exists(creds_pickle_file_loc):
        with open(creds_pickle_file_loc, 'rb') as token:
            creds = pickle.load(token)
    # If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                gdrive_creds_json_loc, SCOPES)
            creds = flow.run_local_server(port=0)
        # Save the credentials for the next run
        with open(creds_pickle_file_loc, 'wb') as token:
            pickle.dump(creds, token)

    drive_serviceservice = build('drive', 'v3', credentials=creds)
    print(f"\n\ntype of drive_serviceservice should say    googleapiclient.discovery.Resource =\n{type(drive_serviceservice)}\n\n")

    source_folder_local = r'/media/rohit/DATA/EverythingD/01SRH-BDBA Acads/Thesis/StoryGenerator/Data/COCO_train2017_118k/'
    gdrive_target_folder_id = r'1n9jymR4BvQ-qsMLLM8Hcqj5xjcMU4Su6'  ## Rishabh_DO_NOT_TOUCH_PLS_1  folder

    files_arr = [source_folder_local + f for f in os.listdir(source_folder_local)]
    print(f"\nFound {len(files_arr)} files in the source folder = {source_folder_local}")

    files_to_upload_arr = files_arr[start_idx: end_idx]
    del files_arr

    print(f"\n\nGoing to start attempting upload from the files array\nStart index = {start_idx} , End index = {end_idx}")
    print(f"First file = {files_to_upload_arr[0]}\nLast file = {files_to_upload_arr[-1]}\n\n")

    start_timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    start_tick = time.time()

    print(f"\n\nStarted at = {start_timestamp}\n")

    files_processed = 0
    count_files_uploaded = 0
    count_problem = 0

    for each_file_to_upload in files_to_upload_arr:
        if files_processed % print_status_freq == 0:
            print(f"At {datetime.datetime.now().strftime('%H:%M:%S')} , files_processed = {files_processed} , count_files_uploaded = {count_files_uploaded} , count_problem = {count_problem}")
        if my_upload_file(drive_serviceservice, each_file_to_upload, gdrive_target_folder_id):
            count_files_uploaded += 1
        else:
            count_problem += 1
        files_processed += 1
    
    end_timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    end_tick = time.time()

    print(f"\n\nProcessed for Start index = {start_idx} , End index = {end_idx}")
    print(f"Started at = {start_timestamp}\nEnded at = {end_timestamp}\nTime taken = {end_tick - start_tick} seconds")
    print(f"files_processed = {files_processed} , count_files_uploaded = {count_files_uploaded} , count_problem = {count_problem}")
    
    return



if __name__ == '__main__':
    args = argparser.parse_args()
    _main_(args)