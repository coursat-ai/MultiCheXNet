import os
import json
# your authentication credentials:
# jsonfile has two values username and key
kaggle_auth_json_file = 'drive/My Drive/courses/kaggle_keys.json'
#the output path were the data is saved
dataset_path='dataset/'
#booleans to download data
download_RSNA_bool =True
download_SIIM_ACR_bool =True

def download_from_kaggle(user ,key, kaggle_api_command,download_path=None ):
    if '.kaggle' not in os.listdir('/root'):
        os.system("mkdir ~/.kaggle")
    os.system("touch /root/.kaggle/kaggle.json")
    os.system("chmod 666 /root/.kaggle/kaggle.json")
    with open('/root/.kaggle/kaggle.json', 'w') as f:
        f.write('{"username":"%s","key":"%s"}' % (user, key))
    os.system("chmod 600 /root/.kaggle/kaggle.json")
    if download_path is not None:
      if not os.path.isdir(download_path):
        os.mkdir(download_path)
    else:
      donload_path="./"
    os.system("cd {} ; {} ".format( download_path,kaggle_api_command))
    os.system("cd {}; unzip *.zip >/dev/null".format(download_path))
    os.system("cd {}; rm *.zip".format(download_path))

def download_SIIM_ACR(user,key,dataset_path):
  SIIM_ACR_path = os.path.join(dataset_path,"SIIM_ACR")
  if not os.path.isdir(SIIM_ACR_path):
    os.mkdir(SIIM_ACR_path)
  download_from_kaggle(user,key, "kaggle datasets download -d abhishek/siim-dicom-images", SIIM_ACR_path)
  download_from_kaggle(user,key, "kaggle competitions download -c siim-acr-pneumothorax-segmentation", SIIM_ACR_path)

def download_RSNA(user,key,dataset_path):
  RSNA_ACR_path = os.path.join(dataset_path,"RSNA_ACR")
  if not os.path.isdir(RSNA_ACR_path):
    os.mkdir(RSNA_ACR_path)
  download_from_kaggle(user,key, "kaggle competitions download -c rsna-pneumonia-detection-challenge", RSNA_ACR_path)

def download_data(kaggle_auth_json_file, dataset_path='dataset',SIIM_ACR=False, RSNA=False):
  user_info_dict = json.load(open(kaggle_auth_json_file,'r'))
  user,key = user_info_dict['user'] , user_info_dict['key']
  if not os.path.isdir(dataset_path):
    os.mkdir(dataset_path)
  if SIIM_ACR ==True:
    download_SIIM_ACR(user,key,dataset_path)
  if RSNA ==True:
    download_RSNA(user,key,dataset_path)

      
download_data(kaggle_auth_json_file , dataset_path , SIIM_ACR=True ,RSNA=download_RSNA_bool )