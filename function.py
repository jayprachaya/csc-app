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

@jit
def load_img1_case(img_path, img_size):
  dirFiles = []
  img_3d = []

  for i in os.listdir(img_path):
    dirFiles.append(i)
    dirFiles.sort(key=lambda f: int(re.sub('\D', '', f)))

  for file in dirFiles:
    if file[11:] == 'jpg':
      file_name  = file[:10]+'.jpg'
      file_path = os.path.join(img_path, file_name)
      
      if os.path.exists(file_path): #have file return => True, no file return => False
        image = np.array(Image.open(file_path).convert('L'))

        # resize to img_size
        image = cv2.resize(image, (img_size, img_size), interpolation = cv2.INTER_AREA)

      else:
        print('no image file:',file_name)
        
      img_3d.append(image)
    else: continue

  # convert to array
  array_3d_img = np.array(img_3d)
  print(array_3d_img.shape)
  return array_3d_img

#-------------------------- padding and patchify function --------------------------
@jit
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
@jit
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

#   print(prediction_result.shape)
  return prediction_result

# ---------------------------------- lesion predict --------------------------------
@jit
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
@jit
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
  score_lobe = {'BG':0 ,'RUL':0, 'RLL':0, 'RML':0, 'LUL':0, 'LLL':0,'ERROR':0}
  score_lesion_lobe = {'BG':0 ,'RUL':0, 'RLL':0, 'RML':0, 'LUL':0, 'LLL':0}
  for i in range(lung_result.shape[0]):
    for j in range(lung_result.shape[1]):
      for k in range(lung_result.shape[2]):
        if lung_result[i,j,k] == 0: 
          score_lobe["BG"] += 1 
          if lesion_result[i,j,k] == 1 : score_lesion_lobe['BG']+=1
          else: continue
        elif lung_result[i,j,k] == 1: 
          score_lobe["RUL"] += 1
          if lesion_result[i,j,k] == 1 : score_lesion_lobe['RUL']+=1
          else: continue
        elif lung_result[i,j,k] == 2: 
          score_lobe["RLL"] += 1
          if lesion_result[i,j,k] == 1 : score_lesion_lobe['RLL']+=1
          else: continue
        elif lung_result[i,j,k] == 3: 
          score_lobe["RML"] += 1
          if lesion_result[i,j,k] == 1 : score_lesion_lobe['RML']+=1
          else: continue
        elif lung_result[i,j,k] == 4: 
          score_lobe["LUL"] += 1
          if lesion_result[i,j,k] == 1 : score_lesion_lobe['LUL']+=1
          else: continue
        elif lung_result[i,j,k] == 5: 
          score_lobe["LLL"] += 1 
          if lesion_result[i,j,k] == 1 : score_lesion_lobe['LLL']+=1
          else: continue
        else: score_lobe["ERROR"] += 1
  # print('***** DONE *****')
  # print(score_lobe)
  # print(score_lesion_lobe)
  return score_lobe, score_lesion_lobe
@jit
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
@jit
def TSS_score(score_lobe_, score_lesion_lobe_):
  PI_BG, CT_BG = PI_CTscore(score_lesion_lobe_['BG'], score_lobe_['BG'])
  PI_RUL, CT_RUL = PI_CTscore(score_lesion_lobe_['RUL'], score_lobe_['RUL'])
  PI_RLL, CT_RLL = PI_CTscore(score_lesion_lobe_['RLL'], score_lobe_['RLL'])
  PI_RML, CT_RML = PI_CTscore(score_lesion_lobe_['RML'], score_lobe_['RML'])
  PI_LUL, CT_LUL = PI_CTscore(score_lesion_lobe_['LUL'], score_lobe_['LUL'])
  PI_LLL, CT_LLL = PI_CTscore(score_lesion_lobe_['LLL'], score_lobe_['LLL'])

  # make dataframe
  CT_score_df = {'Lobe':['Right Upper Lobe (RUL)','Right Lower Lobe (RLL)','Right Middle Lobe (RML)',
                         'Left Upper Lobe (LUL)','Left Lower Lobe (LLL)'],
                'Percent Infection (PI%)':[PI_RUL, PI_RLL, PI_RML, PI_LUL, PI_LLL],
                'Score':[CT_RUL, CT_RLL, CT_RML, CT_LUL, CT_LLL]}

  # CT_each_PI = {'RUL':PI_RUL, 'RLL':PI_RLL, 'RML':PI_RML, 'LUL':PI_LUL, 'LLL':PI_LLL}
  # CT_each_Score = {'RUL':CT_RUL, 'RLL':CT_RLL, 'RML':CT_RML, 'LUL':CT_LUL, 'LLL':CT_LLL}
  TSS = CT_RUL+ CT_RLL+ CT_RML+ CT_LUL+ CT_LLL
  if TSS == 0: Type ="Normal"
  elif TSS <= 7: Type ="Mild"
  elif TSS <= 17: Type ="Moderate"
  else: Type ="Severe"

  return TSS, Type, pd.DataFrame.from_dict(CT_score_df)


# ---------------------------------- TSS version 2------------------------------------------
@jit
def PI_version2(Lesion_area, Lobe_area):
  if Lesion_area == 0 or Lobe_area == 0:
    PI = 0 
  else: 
    quotient = Lesion_area / Lobe_area
    PI = quotient * 100
  return PI

@jit
def sum_pixel_version2(lung_result_, lesion_result_):
  '''
  Label class
  RUL :right upper lobe:  1
  RLL :right lower lobe:  2
  RML :right middle lobe: 3
  LUL :left upper lobe:   4
  LLL :left lower lobe:   5
  '''
  score_lobe = {'BG':0 ,'RUL':0, 'RLL':0, 'RML':0, 'LUL':0, 'LLL':0,'ERROR':0}
  score_lesion_lobe = {'BG':0 ,'RUL':0, 'RLL':0, 'RML':0, 'LUL':0, 'LLL':0}
  for i in range(lung_result_.shape[0]):
    for j in range(lung_result_.shape[1]):
      if lung_result_[i,j] == 0: 
        score_lobe["BG"] += 1 
        if lesion_result_[i,j] == 1 : score_lesion_lobe['BG']+=1
        else: continue
      elif lung_result_[i,j] == 1: 
        score_lobe["RUL"] += 1
        if lesion_result_[i,j] == 1 : score_lesion_lobe['RUL']+=1
        else: continue
      elif lung_result_[i,j] == 2: 
        score_lobe["RLL"] += 1
        if lesion_result_[i,j] == 1 : score_lesion_lobe['RLL']+=1
        else: continue
      elif lung_result_[i,j] == 3: 
        score_lobe["RML"] += 1
        if lesion_result_[i,j] == 1 : score_lesion_lobe['RML']+=1
        else: continue
      elif lung_result_[i,j] == 4: 
        score_lobe["LUL"] += 1
        if lesion_result_[i,j] == 1 : score_lesion_lobe['LUL']+=1
        else: continue
      elif lung_result_[i,j] == 5: 
        score_lobe["LLL"] += 1 
        if lesion_result_[i,j] == 1 : score_lesion_lobe['LLL']+=1
        else: continue
      else: score_lobe["ERROR"] += 1

  return score_lobe, score_lesion_lobe

@jit
def CT_score(PI_):
  if PI_ == 0: CT_score = 0
  elif PI_ <= 5: CT_score = 1
  elif PI_ <= 25: CT_score = 2
  elif PI_ <= 50: CT_score = 3
  elif PI_ <= 75: CT_score = 4
  else: CT_score = 5
  return CT_score

# TSS ver2
'''
Percentage of each CT slice for all lung lobe
'''
@jit
def TSS_score_version2(lung_result, lesion_result):
  score_total_PI = {'RUL':[], 'RLL':[], 'RML':[], 'LUL':[], 'LLL':[]}
  count_lobe_appear = {'RUL':0, 'RLL':0, 'RML':0, 'LUL':0, 'LLL':0}
  for i in range(lung_result.shape[0]):
    score_lobe_, score_lesion_lobe_ = sum_pixel_version2(lung_result[i,:,:], lesion_result[i,:,:])
    PI_RUL = PI_version2(score_lesion_lobe_['RUL'], score_lobe_['RUL'])
    PI_RLL = PI_version2(score_lesion_lobe_['RLL'], score_lobe_['RLL'])
    PI_RML = PI_version2(score_lesion_lobe_['RML'], score_lobe_['RML'])
    PI_LUL = PI_version2(score_lesion_lobe_['LUL'], score_lobe_['LUL'])
    PI_LLL = PI_version2(score_lesion_lobe_['LLL'], score_lobe_['LLL'])

    score_total_PI['RUL'].append(PI_RUL)
    score_total_PI['RLL'].append(PI_RLL)
    score_total_PI['RML'].append(PI_RML)
    score_total_PI['LUL'].append(PI_LUL)
    score_total_PI['LLL'].append(PI_LLL)

    if  score_lobe_['RUL'] != 0 :count_lobe_appear['RUL'] += 1
    if  score_lobe_['RLL'] != 0 :count_lobe_appear['RLL'] += 1
    if  score_lobe_['RML'] != 0 :count_lobe_appear['RML'] += 1
    if  score_lobe_['LUL'] != 0 :count_lobe_appear['LUL'] += 1
    if  score_lobe_['LLL'] != 0 :count_lobe_appear['LLL'] += 1
  
  PI_RUL = sum(score_total_PI['RUL'])/count_lobe_appear['RUL']
  PI_RLL = sum(score_total_PI['RLL'])/count_lobe_appear['RLL']
  PI_RML = sum(score_total_PI['RML'])/count_lobe_appear['RML']
  PI_LUL = sum(score_total_PI['LUL'])/count_lobe_appear['LUL']
  PI_LLL = sum(score_total_PI['LLL'])/count_lobe_appear['LLL']

  CT_RUL = CT_score(round(PI_RUL))
  CT_RLL = CT_score(round(PI_RLL))
  CT_RML = CT_score(round(PI_RML))
  CT_LUL = CT_score(round(PI_LUL))
  CT_LLL = CT_score(round(PI_LLL))

  CT_score_df = {'Lobe':['Right Upper Lobe (RUL)','Right Lower Lobe (RLL)','Right Middle Lobe (RML)',
                          'Left Upper Lobe (LUL)','Left Lower Lobe (LLL)'],
                'Percent Infection (PI%)':[round(PI_RUL), round(PI_RLL), round(PI_RML), round(PI_LUL), round(PI_LLL)],
                'Score':[CT_RUL, CT_RLL, CT_RML, CT_LUL, CT_LLL]}

  # TSS
  TSS = CT_RUL+ CT_RLL+ CT_RML+ CT_LUL+ CT_LLL
  if TSS == 0: Type ="Normal"
  elif TSS <= 7: Type ="Mild"
  elif TSS <= 17: Type ="Moderate"
  else: Type ="Severe"

  return TSS, Type, pd.DataFrame.from_dict(CT_score_df)

# TSS ver3
'''
Percentage of each CT slice for all lesion (not all lung lobe)
'''
@jit
def TSS_score_version3(lung_result, lesion_result):
  score_total_PI = {'RUL':[], 'RLL':[], 'RML':[], 'LUL':[], 'LLL':[]}
  count_lobe_appear = {'RUL':0, 'RLL':0, 'RML':0, 'LUL':0, 'LLL':0}
  for i in range(lung_result.shape[0]):
    score_lobe_, score_lesion_lobe_ = sum_pixel_version2(lung_result[i,:,:], lesion_result[i,:,:])
    PI_RUL = PI_version2(score_lesion_lobe_['RUL'], score_lobe_['RUL'])
    PI_RLL = PI_version2(score_lesion_lobe_['RLL'], score_lobe_['RLL'])
    PI_RML = PI_version2(score_lesion_lobe_['RML'], score_lobe_['RML'])
    PI_LUL = PI_version2(score_lesion_lobe_['LUL'], score_lobe_['LUL'])
    PI_LLL = PI_version2(score_lesion_lobe_['LLL'], score_lobe_['LLL'])

    score_total_PI['RUL'].append(PI_RUL)
    score_total_PI['RLL'].append(PI_RLL)
    score_total_PI['RML'].append(PI_RML)
    score_total_PI['LUL'].append(PI_LUL)
    score_total_PI['LLL'].append(PI_LLL)

    if  score_lesion_lobe_['RUL'] != 0 :count_lobe_appear['RUL'] += 1
    if  score_lesion_lobe_['RLL'] != 0 :count_lobe_appear['RLL'] += 1
    if  score_lesion_lobe_['RML'] != 0 :count_lobe_appear['RML'] += 1
    if  score_lesion_lobe_['LUL'] != 0 :count_lobe_appear['LUL'] += 1
    if  score_lesion_lobe_['LLL'] != 0 :count_lobe_appear['LLL'] += 1
  
  # Average PI% -----------------
  # RUL
  if count_lobe_appear['RUL'] != 0:
    PI_RUL = sum(score_total_PI['RUL'])/count_lobe_appear['RUL']
  else:
    PI_RUL = 0
    
  # RLL
  if count_lobe_appear['RLL'] != 0:
    PI_RLL = sum(score_total_PI['RLL'])/count_lobe_appear['RLL']
  else:
    PI_RLL = 0

  # RML
  if count_lobe_appear['RML'] != 0:
    PI_RML = sum(score_total_PI['RML'])/count_lobe_appear['RML']
  else:
    PI_RML= 0

  # LUL
  if count_lobe_appear['LUL'] != 0:
    PI_LUL = sum(score_total_PI['LUL'])/count_lobe_appear['LUL']
  else:
    PI_LUL = 0

  # RLL
  if count_lobe_appear['LLL'] != 0:
    PI_LLL = sum(score_total_PI['LLL'])/count_lobe_appear['LLL']
  else:
    PI_LLL = 0

  CT_RUL = CT_score(round(PI_RUL))
  CT_RLL = CT_score(round(PI_RLL))
  CT_RML = CT_score(round(PI_RML))
  CT_LUL = CT_score(round(PI_LUL))
  CT_LLL = CT_score(round(PI_LLL))

  CT_score_df = {'Lobe':['Right Upper Lobe (RUL)','Right Lower Lobe (RLL)','Right Middle Lobe (RML)',
                          'Left Upper Lobe (LUL)','Left Lower Lobe (LLL)'],
                'Percent Infection (PI%)':[round(PI_RUL), round(PI_RLL), round(PI_RML), round(PI_LUL), round(PI_LLL)],
                'Score':[CT_RUL, CT_RLL, CT_RML, CT_LUL, CT_LLL]}

  # TSS
  TSS = CT_RUL+ CT_RLL+ CT_RML+ CT_LUL+ CT_LLL
  if TSS == 0: Type ="Normal"
  elif TSS <= 7: Type ="Mild"
  elif TSS <= 17: Type ="Moderate"
  else: Type ="Severe"

  return TSS, Type, pd.DataFrame.from_dict(CT_score_df)

#-----------------------------------Overy---------------------------------------#
@jit
def overay(result_oredict, image, slice__):
  colors = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (1, 0, 1)]
  # resize 256 to 512
  result512 = cv2.resize(result_oredict[slice__,:,:], (512, 512), interpolation = cv2.INTER_NEAREST)
  result_image = color.label2rgb(result512, image[slice__,:,:], alpha=0.2, bg_label=0, bg_color=(0, 0, 0), colors = colors)
  # result_image = color.label2rgb(result_oredict[slice__,:,:], image[slice__,:,:], alpha=0.2, bg_label=0, bg_color=(0, 0, 0), colors = colors)

  fig, axs = plt.subplots(1, 1, sharey=True)
  #axs.set_title('Prediction')
  axs.imshow(result_image, cmap='Greys')
  axs.axis('off')
  fig.tight_layout(pad = 0)
  fig.patch.set_facecolor('#000000')
  #fig.suptitle('Slice: '+str(zzz))
  show_pyplot = plt.show()

  return show_pyplot
