# import library
import numpy as np
import pandas as pd
import cv2
import os
import re
from PIL import Image
from skimage import color
from matplotlib import pyplot as plt
from matplotlib import image as mpimg
import base64
from numba import jit

from patchify import patchify, unpatchify
from tensorflow import keras
import segmentation_models_3D as sm

#-------------------------- load CT slice img -> 3d image --------------------------
@jit
def load_imgdata(image_list, img_size):
  img_3d = []
  for upload_file in image_list:
    # images = Image.open(upload_file)
    # st.image(images)

    image = np.array(Image.open(upload_file).convert('L'))
    # resize to img_size
    image = cv2.resize(image, (img_size, img_size), interpolation = cv2.INTER_AREA)
    img_3d.append(image)
  # convert to array
  array_3d_img = np.array(img_3d)
  # print(array_3d_img.shape)
  return array_3d_img

# def load_img1_case(img_path, img_size):
#   dirFiles = []
#   img_3d = []

#   for i in os.listdir(img_path):
#     dirFiles.append(i)
#     dirFiles.sort(key=lambda f: int(re.sub('\D', '', f)))

#   for file in dirFiles:
#     if file[11:] == 'jpg':
#       file_name  = file[:10]+'.jpg'
#       file_path = os.path.join(img_path, file_name)
      
#       if os.path.exists(file_path): #have file return => True, no file return => False
#         image = np.array(Image.open(file_path).convert('L'))

#         # resize to img_size
#         image = cv2.resize(image, (img_size, img_size), interpolation = cv2.INTER_AREA)

#       else:
#         print('no image file:',file_name)
        
#       img_3d.append(image)
#     else: continue

#   # convert to array
#   array_3d_img = np.array(img_3d)
#   print(array_3d_img.shape)
#   return array_3d_img

#-------------------------- padding and patchify function --------------------------

def padding(image, slice_vol):
  slice_, row, col= image.shape #(119,256,256)
  #เพิ่ม slice เข้าไป
  if slice_vol > slice_: 
    slice_padding = slice_vol - slice_
    padding_array = np.zeros((slice_padding, row, col))
    image_paded = np.concatenate((image,padding_array), axis=0)
  #เอา slice ออก
  else: 
    image_paded = image[:slice_vol,:,:]
  return np.asarray(np.array(image_paded), dtype="uint8" )

def pad_patchify(padded,x,y,z):
  patches = patchify(padded, (x, y, z), step=x)
  input_img = np.reshape(patches, (-1, patches.shape[3], patches.shape[4], patches.shape[5]))
  return input_img

#-------------------------- prediction --------------------------

#@st.cache # cache the function so predictions aren't always redone (Streamlit refreshes every click)
@jit
def predict_function(my_model, backbone, large_image, padding_size):
  padded = padding(large_image, padding_size) # padding
  input_img = pad_patchify(padded,128,256,256) # patchify
  preprocess_input = sm.get_preprocessing(backbone)
  patch_3ch = np.stack((input_img,)*3, axis=-1)
  patch_3ch_input = preprocess_input(patch_3ch)
  # predict
  prediction = my_model.predict(patch_3ch_input)
  prediction_argmax = np.argmax(prediction, axis=4)
  # reshape
  prediction_reshape = np.reshape(prediction_argmax,(padding_size,256,256)) # padding_size=128, 256
  # select original slice length and Convert to uint8
  prediction_result = prediction_reshape[:large_image.shape[0],:,:].astype(np.uint8)
  return prediction_result

#@st.cache # cache the function so predictions aren't always redone (Streamlit refreshes every click)
@jit
def predict(my_model, backbone, large_image, padding_size):
  if large_image.shape[0] <= 128:
    prediction_result = predict_function(my_model, backbone, large_image, padding_size)
  elif ((large_image.shape[0] > 128) and (large_image.shape[0] <= 175)):
    slice_volumn = large_image.shape[0]
    # set center range 
    start_lung = int((slice_volumn/2)-64)
    end_lung = int((slice_volumn/2)+64)

    if start_lung != 0:
      start_range = np.zeros((int(start_lung-1), 256, 256))
      end_range = np.zeros((int(slice_volumn-end_lung+1), 256, 256))
      # predict
      result = predict_function(my_model, backbone, large_image[start_lung-1:end_lung-1,:,:], padding_size)
      # combine result: start_range, prediction_result, end_range
      prediction_result = np.concatenate((start_range, result, end_range), axis=0)
    else:
      end_range = np.zeros((int(slice_volumn-end_lung), 256, 256))
      # predict
      result = predict_function(my_model, backbone, large_image[start_lung:end_lung,:,:], padding_size)
      # combine result: start_range, prediction_result, end_range
      prediction_result = np.concatenate((result, end_range), axis=0)

  else:
    if large_image.shape[0] <= 256:
      prediction_result = predict_function(my_model, backbone, large_image[::2,:,:], padding_size) # predict skipping slice
    else:
      print('CT slice volume is to large')
      
  return prediction_result

# ---------------------------------- lesion predict --------------------------------

def crop_img(large_image, lung_result):
  if large_image.shape[0]<=175:
    img_crop = large_image.copy()
    for i in range(large_image.shape[0]):
        for j in range(large_image.shape[1]):
            for k in range(large_image.shape[2]):
                if lung_result[i,j,k] != 0: #ข้างใน
                    pass
                else: #ข้างนอก
                    img_crop[i,j,k] = 0
  else:
    img_crop = large_image[::2,:,:].copy()
    for i in range(large_image[::2,:,:].shape[0]):
        for j in range(large_image.shape[1]):
            for k in range(large_image.shape[2]):
                if lung_result[i,j,k] != 0: #ข้างใน
                    pass
                else: #ข้างนอก
                    img_crop[i,j,k] = 0
  return img_crop

def contrast_CLAHE(image, ts):
    clahe = cv2.createCLAHE(clipLimit=ts, tileGridSize=(8,8))
    img_list = []
    for i in range(image.shape[0]):
        cl_img =  clahe.apply(image[i,:,:])
        img_list.append(cl_img)
    new_img = np.asarray(np.array(img_list), dtype="uint8" )
    return(new_img)

# ---------------------------------- TSS ------------------------------------------
@jit
def sum_pixel(lung_result, lesion_result):
  '''
  Label class
  RUL :right upper lobe:  1
  RLL :right lower lobe:  2
  RML :right middle lobe: 3
  LUL :left upper lobe:   4
  LLL :left lower lobe:   5
  '''
  area_lobe = {'BG':0 ,'RUL':0, 'RLL':0, 'RML':0, 'LUL':0, 'LLL':0,'ERROR':0}
  area_lesion_lobe = {'BG':0 ,'RUL':0, 'RLL':0, 'RML':0, 'LUL':0, 'LLL':0}
  for i in range(lung_result.shape[0]):
    for j in range(lung_result.shape[1]):
      for k in range(lung_result.shape[2]):
        if lung_result[i,j,k] == 0: 
          area_lobe["BG"] += 1 
          if lesion_result[i,j,k] == 1 : area_lesion_lobe['BG']+=1
          else: continue
        elif lung_result[i,j,k] == 1: 
          area_lobe["RUL"] += 1
          if lesion_result[i,j,k] == 1 : area_lesion_lobe['RUL']+=1
          else: continue
        elif lung_result[i,j,k] == 2: 
          area_lobe["RLL"] += 1
          if lesion_result[i,j,k] == 1 : area_lesion_lobe['RLL']+=1
          else: continue
        elif lung_result[i,j,k] == 3: 
          area_lobe["RML"] += 1
          if lesion_result[i,j,k] == 1 : area_lesion_lobe['RML']+=1
          else: continue
        elif lung_result[i,j,k] == 4: 
          area_lobe["LUL"] += 1
          if lesion_result[i,j,k] == 1 : area_lesion_lobe['LUL']+=1
          else: continue
        elif lung_result[i,j,k] == 5: 
          area_lobe["LLL"] += 1 
          if lesion_result[i,j,k] == 1 : area_lesion_lobe['LLL']+=1
          else: continue
        else: area_lobe["ERROR"] += 1
  # print('***** DONE *****')
  # print(score_lobe)
  # print(score_lesion_lobe)
  return area_lobe, area_lesion_lobe

def PI_CTscore(Lesion_area, Lobe_area):
  quotient = Lesion_area / Lobe_area
  # PI = round(quotient * 100,2) 
  PI = round(quotient * 100)
  # PI = int(quotient * 100) 
  if PI == 0: CT_score = 0
  elif PI <= 5: CT_score = 1
  elif PI <= 25: CT_score = 2
  elif PI <= 50: CT_score = 3
  elif PI <= 75: CT_score = 4
  else: CT_score = 5
  return(PI, CT_score) 


def TSS_score(area_lobe, area_lesion_lobe):
  # PI_BG, CT_BG = PI_CTscore(area_lesion_lobe['BG'], area_lobe['BG'])
  PI_RUL, CT_RUL = PI_CTscore(area_lesion_lobe['RUL'], area_lobe['RUL'])
  PI_RLL, CT_RLL = PI_CTscore(area_lesion_lobe['RLL'], area_lobe['RLL'])
  PI_RML, CT_RML = PI_CTscore(area_lesion_lobe['RML'], area_lobe['RML'])
  PI_LUL, CT_LUL = PI_CTscore(area_lesion_lobe['LUL'], area_lobe['LUL'])
  PI_LLL, CT_LLL = PI_CTscore(area_lesion_lobe['LLL'], area_lobe['LLL'])

  # make dataframe
  CT_score_df = {'Lobe':['Right Upper Lobe (RUL)','Right Lower Lobe (RLL)','Right Middle Lobe (RML)',
                          'Left Upper Lobe (LUL)','Left Lower Lobe (LLL)'],
                'Percentage of Infection':[PI_RUL, PI_RLL,PI_RML,PI_LUL,PI_LLL],
                'Score':[CT_RUL, CT_RLL, CT_RML, CT_LUL, CT_LLL]}

  CT_score_df2 = {"RUL":[PI_RUL, CT_RUL],
                 "RLL": [PI_RLL, CT_RLL],
                 "RML": [PI_RML, CT_RML],
                 "LUL": [PI_LUL, CT_LUL],
                 "LLL": [PI_LLL, CT_LLL]}

  # CT_each_PI = {'RUL':PI_RUL, 'RLL':PI_RLL, 'RML':PI_RML, 'LUL':PI_LUL, 'LLL':PI_LLL}
  # CT_each_Score = {'RUL':CT_RUL, 'RLL':CT_RLL, 'RML':CT_RML, 'LUL':CT_LUL, 'LLL':CT_LLL}
  TSS = CT_RUL+ CT_RLL+ CT_RML+ CT_LUL+ CT_LLL
  if TSS == 0: Type ="Normal"
  elif TSS <= 7: Type ="Mild"
  elif TSS <= 17: Type ="Moderate"
  else: Type ="Severe"

  # Total lung Involvement
  lung_involve = area_lobe['RUL']+area_lobe['RLL']+area_lobe['RML']+area_lobe['LUL']+area_lobe['LLL']
  lesion_involve = area_lesion_lobe['RUL']+area_lesion_lobe['RLL']+area_lesion_lobe['RML']+area_lesion_lobe['LUL']+area_lesion_lobe['LLL']
  involvement = round((lesion_involve/lung_involve)*100, 2)
  involve_percent = str(involvement) + '%'


  return TSS, Type, involve_percent, pd.DataFrame.from_dict(CT_score_df), pd.DataFrame.from_dict(CT_score_df2,orient='index', columns=['Infection (%)', 'Score'])

#-----------------------------------Overy---------------------------------------

def overay(result_oredict, image, index: int):
  colors = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (1, 0, 1)]
  # resize 256 to 512
  result512 = cv2.resize(result_oredict[index,:,:], (512, 512), interpolation = cv2.INTER_NEAREST)
  result_image = color.label2rgb(result512, image[index,:,:], alpha=0.2, bg_label=0, bg_color=(0, 0, 0), colors = colors)
  # result_image = color.label2rgb(result_oredict[slice__,:,:], image[slice__,:,:], alpha=0.2, bg_label=0, bg_color=(0, 0, 0), colors = colors)

  fig, axs = plt.subplots(1, 1, sharey=True)
  # axs.set_title('Prediction')
  axs.imshow(result_image, cmap='Greys')
  axs.axis('off')
  fig.tight_layout(pad=0)
  fig.patch.set_facecolor('#000000')

  return fig

#-----------------------------------Save---------------------------------------
def create_download_link(val, filename, type_file):
  b64 = base64.b64encode(val)  # val looks like b'...'
  return f'<a href="data:application/octet-stream;base64,{b64.decode()}" download="{filename}">Download {type_file}</a>'