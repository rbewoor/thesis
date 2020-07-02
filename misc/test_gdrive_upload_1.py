## References used in the project:
## Webiste: https://stackoverflow.com/questions/44620369/create-new-file-drive-api-python
##        : https://developers.google.com/drive/api/v3/quickstart/python
## Goal: Inserts all files from a target folder to a target folder on google drive
#from __future__ import print_function
import os
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
import pickle

#SOURCE_DIR = r"/media/rohit/DATA/EverythingD/01SRH-BDBA Acads/Thesis/StoryGenerator/Data/COCO_test2017_41k/test2017"
#SOURCE_DIR = r"/home/rohit/PyWDUbuntu/thesis/Imgs2Detect"
TARGET_DIR_ID = r"INSERT THE DRIVE URL ENDING PART" # google drive index - check from url
TESTFIILE = r"/home/rohit/PyWDUbuntu/thesis/gdrive_pil_20200701/testfile.txt"

# set GET_LIST_RUN to True - to only find the directories in the drive, list them and exit prorgram.
GET_LIST_RUN = False

# If modifying these scopes, delete the file token.pickle.
SCOPES = ['https://www.googleapis.com/auth/drive']

def main():
    global TARGET_DIR_ID, GET_LIST_RUN, TESTFIILE #, SOURCE_DIR
    # If modifying these scopes, delete the file token.pickle.
    SCOPES = ['https://www.googleapis.com/auth/drive']
    creds = None
    gdrive_creds_json_loc = r'/home/rohit/PyWDUbuntu/thesis/gdrive_pil_20200701/credentials.json'
    creds_pickle_file_loc = r'/home/rohit/PyWDUbuntu/thesis/gdrive_pil_20200701/token.pickle'
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
        #with open('token.pickle', 'wb') as token:
        #    pickle.dump(creds, token)
        with open(creds_pickle_file_loc, 'wb') as token:
            pickle.dump(creds, token)

    drive_service = build('drive', 'v3', credentials=creds)
    
    # get the folder list - if the GET_LIST_RUN flag is True
    if GET_LIST_RUN:
        results = drive_service.files().list(
            pageSize=10, fields="nextPageToken, files(id, name)").execute()
        items = results.get('files', [])
        if not items:
            print('No files found.')
        else:
            print('Files:')
            for item in items:
                print(u'{0} ({1})'.format(item['name'], item['id']))
        return
    
    with open(TESTFIILE, "w") as f:
      f.write('nothing of interest')
    
    file_2_upload_name = os.path.basename(TESTFIILE)
    file_metadata = {'name': file_2_upload_name, 'parents': [TARGET_DIR_ID]}
    media = MediaFileUpload(TESTFIILE, mimetype='text/txt')
    file = drive_service.files().create(body=file_metadata,
                                    media_body=media,
                                    fields='id').execute()
    print(f"File ID: {file.get('id')}")
    
    print(f"\n\nNormal exit from program.\n")
    
    #image_files_arr = [os.path.join(SOURCE_DIR, f) for f in os.listdir(SOURCE_DIR) if os.path.isfile(os.path.join(SOURCE_DIR, f))]
    #img_files_count = len(image_files_arr)
    #print(f"\n\nSource Dir: {SOURCE_DIR}\nImages found count = {img_files_count}")

    #count_files_upload = 0
    #for each_img in image_files_arr:
    #    file_2_upload = os.path.basename(each_img)
    #    file_metadata = {'name': file_2_upload, 'parents': [TARGET_DIR_ID]}
    #    media = MediaFileUpload(each_img, mimetype='image/jpeg')
    #    file = drive_service.files().create(body=file_metadata,
    #                                    media_body=media,
    #                                    fields='id').execute()
    #    #print(f"File ID: {file.get('id')}")
    #    if file.get('id') is not None:
    #        count_files_upload += 1
    
    ## show summary
    #print(f"Images in source dir: {img_files_count}\nUploaded images: {count_files_upload}")

    print(f"\n\nNormal exit from program.\n")

if __name__ == '__main__':
    main()
