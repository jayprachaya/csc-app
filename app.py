from email.mime import image
from turtle import width
import streamlit as st
from PIL import Image
from keras.models import load_model
from function import *
from matplotlib import pyplot as plt
from fpdf import FPDF
from datetime import datetime
from datetime import date
import os

def main():
    #page setting
    st.set_page_config(page_title="CSC", layout = 'wide')
    # CURRENT_THEME = "Dark"
    st.title('COVID-19 Severity Calculator (CSC)')
    primary_clr = st.get_option("theme.primaryColor")
    st.set_option('deprecation.showPyplotGlobalUse', False)
    with st.expander('See more Description'):
        st.write('''
        This is web application for classification severity of COVID-19 patient levels 
        that Normal, Mild, Moderate or Severe.
        ''')

    if 'patient_number' not in st.session_state:
        st.session_state.patient_number = None
    if 'raw_img3d' not in st.session_state:
        st.session_state.raw_img3d = None
    if 'image_3d' not in st.session_state:
        st.session_state.image_3d = None
    if 'lesion_result' not in st.session_state:
        st.session_state.lesion_result = None
    if 'TSS' not in st.session_state:
        st.session_state.TSS = None
    if 'Type_' not in st.session_state:
        st.session_state.Type_ = None
    if 'df' not in st.session_state:
        st.session_state.df = None
    if 'overay_re' not in st.session_state:
        st.session_state.overay_re = None
    if 'summa' not in st.session_state:
        st.session_state.summa = None

    st.sidebar.title('INFORMATION')

    patient_number = st.sidebar.text_input('Please Fill the Hospital Number', placeholder = 'Hospital Number')
    st.session_state.patient_number = patient_number
    upload_file = st.sidebar.file_uploader('Choose CT scan File', accept_multiple_files = True)



    place_container  = st.empty()
    result_empty = st.empty()
    predic, ___, save_button = st.sidebar.columns([1,1,1])

    if len(upload_file) == 0 :
        st.sidebar.warning('Please Choose the CT-Scan Files')
    elif len(upload_file) > 256 :
        st.warning('Please upload the CT-Scan Files less than 257 images')
    elif st.session_state.lesion_result is None:
        with place_container.container():
            img1 = st.slider('', min_value = 1, max_value = len(upload_file), key='pre_sli')
            ___, img_col, ___ = st.columns([1,2,1])
            img = Image.open(upload_file[img1])
            img_col.image(img, use_column_width = 'always')
    else:
        place_container.empty()
        with place_container.container():
            col_1,col_2 = st.columns([1,1])
            if st.session_state.raw_img3d.shape[0] <= 128:
                img1 = st.slider('', min_value = 1, max_value = len(upload_file), key='pro_sli')-1
                img = Image.open(upload_file[img1])
                overay_re = overay(st.session_state.lesion_result, st.session_state.raw_img3d, img1)
            elif ((st.session_state.raw_img3d.shape[0] > 128) and (st.session_state.raw_img3d.shape[0] <= 175)):
                start_lung = int((len(upload_file)/2)-64)
                end_lung = int((len(upload_file)/2)+64)
                img1 = st.slider('', min_value = 1, max_value = end_lung-start_lung, key='pro_sli')-1
                img = Image.open(upload_file[start_lung:end_lung:][img1])
                overay_re = overay(st.session_state.lesion_result, st.session_state.raw_img3d[start_lung:end_lung:,:,:], img1)
            elif st.session_state.raw_img3d.shape[0] <= 256:
                img1 = st.slider('', min_value = 1, max_value = int(len(upload_file)/2), key='pro_sli')-1
                img = Image.open(upload_file[::2][img1])
                overay_re = overay(st.session_state.lesion_result, st.session_state.raw_img3d[::2,:,:], img1)
            col_2.pyplot(overay_re)
            col_1.image(img, use_column_width = 'always')

        #Table result
        result_empty.empty()
        with result_empty.container():
            st.subheader('Results')
            col1, col2 = st.columns(2)
            col1.metric(label = 'Total Severity Score', value = st.session_state.TSS )
            col1.metric(label = 'Severity Type', value = st.session_state.Type_)
            col2.dataframe(st.session_state.df)
            pdf = FPDF()
            pdf = FPDF(orientation = 'P', unit = 'mm', format = 'A4')
            pdf.add_page()
            page_width = pdf.w - 2*pdf.l_margin
            pdf.ln(10)
            pdf.set_font('Courier', 'B', 16)
            pdf.cell(page_width, 0.0, align = 'C', txt = "COVID-19 Patient's Information", ln = 1)
            pdf.ln(2)
            pdf.cell(0, 0, txt = '_'*55)
            pdf.ln(10)
            pdf.set_font('Courier', '', 14)
            pdf.cell(page_width, h = 5.0, align = 'L', txt = 'Date: ' + date.today().strftime("%b-%d-%Y") + '   Time:' + datetime.now().strftime("%H.%M.%S"), ln = 1)
            pdf.ln(5)
            pdf.cell(page_width, h = 5.0, align = 'L', txt = 'Hospital number (HN): ' + st.session_state.patient_number, ln = 1)
            pdf.ln(5)
            pdf.cell(page_width, h = 5.0, align = 'L', txt = 'Total Severity Type(TSS): ' + str(st.session_state.TSS), ln = 1)
            pdf.ln(5)
            pdf.cell(page_width, h = 5.0, align = 'L', txt = 'Severity Type : ' + str(st.session_state.Type_), ln = 1)
            pdf.ln(5)
            pdf.cell(page_width, h = 5.0, align = 'L', txt = 'Percentage of Infection Table', ln = 1)
            pdf.ln(3)
            pdf.set_font('Courier', '', 11)
            columnNameList = list(st.session_state.df.columns)
            for header in columnNameList[:-1]:
                pdf.cell(60, 10, header, 1, 0, 'C')
            pdf.cell(60, 10, columnNameList[-1], 1, 2, 'C')
            pdf.cell(-120)
            pdf.set_font('Courier', '', 11)
            for row in range(0, len(st.session_state.df)):
                for col_num, col_name in enumerate(columnNameList):
                    if col_num != len(columnNameList) - 1:
                        pdf.cell(60, 10, str(st.session_state.df['%s' % (col_name)].iloc[row]),1 ,0 , 'C')
                    else:
                        pdf.cell(60, 10, str(st.session_state.df['%s' % (col_name)].iloc[row]),1 ,2 , 'C')
                        pdf.cell(-120)
            pdf.cell(60, 10, "", 0, 2)
            pdf.cell(20)
            col2.download_button(label = 'Save', data = pdf.output(dest='S').encode('latin-1'), file_name = patient_number + '_' + st.session_state.Type_ + '.pdf')

    if predic.button('Predict'):
        # input file to 3d
        raw_img3d = load_imgdata(upload_file,512)
        st.session_state.raw_img3d = raw_img3d

        image_3d = load_imgdata(upload_file,256)
        st.session_state.image_3d = image_3d

        # Load lung model ---------------------------
        current_path = os.getcwd() # getting the current path
        lung_model_path = os.path.join(current_path, 'Lung_Model5.h5')
        lung_model = load_model(lung_model_path, compile=False)

        #setting parameter
        BACKBONE = 'densenet169'
        padding_size = 128

        #lung prediction
        lung_result = predict(lung_model, BACKBONE,image_3d, padding_size)

        # Load lesion model ---------------------------
        lesion_model_path = os.path.join(current_path, 'Lesion_Model43.h5')
        lesion_model = load_model(lesion_model_path, compile=False)

        # image croped and predict
        # img_crop  = crop_img(image_3d, lung_result)
        img_crop_  = crop_img(image_3d, lung_result)

        # CLAHE (contrast adjustment)
        img_crop  = contrast_CLAHE(img_crop_, 3)
        # Predict
        lesion_result = predict(lesion_model, BACKBONE, img_crop, padding_size)
        st.session_state.lesion_result = lesion_result
        # run_count = run_count+1 # Predict finish!

        # TSS version 1
        score_lobe, score_lesion_lobe = sum_pixel(lung_result, lesion_result)

        TSS, Type_, df  = TSS_score(score_lobe, score_lesion_lobe)
        st.session_state.TSS = TSS
        st.session_state.Type_ = Type_
        st.session_state.df = df
        # TSS version 2
        # TSS, Type_, df  = TSS_score_version2(lung_result, lesion_result)

        # TSS version 3 ---------------------------
        #TSS, Type_, df  = TSS_score_version3(lung_result, lesion_result)
        place_container.empty()
        with place_container.container():
            col_1,col_2 = st.columns([1,1])
            if raw_img3d.shape[0] <= 128:
                img1 = st.slider('', min_value = 1, max_value = len(upload_file), key='pro_sli')-1
                img = Image.open(upload_file[img1])
                overay_re = overay(st.session_state.lesion_result, st.session_state.raw_img3d, img1)
            elif ((raw_img3d.shape[0] > 128) and (raw_img3d.shape[0] <= 175)):
                start_lung = int((len(upload_file)/2)-64)
                end_lung = int((len(upload_file)/2)+64)
                img1 = st.slider('', min_value = 1, max_value = end_lung-start_lung, key='pro_sli')-1
                img = Image.open(upload_file[start_lung:end_lung:][img1])
                overay_re = overay(st.session_state.lesion_result, st.session_state.raw_img3d[start_lung:end_lung:,:,:], img1)
            elif raw_img3d.shape[0] <= 256:
                img1 = st.slider('', min_value = 1, max_value = int(len(upload_file)/2), key='pro_sli')-1
                img = Image.open(upload_file[::2][img1])
                overay_re = overay(st.session_state.lesion_result, st.session_state.raw_img3d[::2,:,:], img1)
            col_1,col_2 = st.columns([1,1])
            col_2.pyplot(overay_re)
            col_1.image(img, use_column_width = 'always')

        #Table result
        result_empty.empty()
        with result_empty.container():
            st.subheader('Results')
            col1, col2 = st.columns(2)
            col1.metric(label = 'Total Severity Score', value = st.session_state.TSS )
            col1.metric(label = 'Severity Type', value = st.session_state.Type_ )
            col2.dataframe(st.session_state.df)
            pdf = FPDF()
            pdf = FPDF(orientation = 'P', unit = 'mm', format = 'A4')
            pdf.add_page()
            page_width = pdf.w - 2*pdf.l_margin
            pdf.ln(10)
            pdf.set_font('Courier', 'B', 16)
            pdf.cell(page_width, 0.0, align = 'C', txt = "COVID-19 Patient's Information", ln = 1)
            pdf.ln(2)
            pdf.cell(0, 0, txt = '_'*55)
            pdf.ln(10)
            pdf.set_font('Courier', '', 14)
            pdf.cell(page_width, h = 5.0, align = 'L', txt = 'Date: ' + date.today().strftime("%b-%d-%Y") + '   Time:' + datetime.now().strftime("%H.%M.%S"), ln = 1)
            pdf.ln(5)
            pdf.cell(page_width, h = 5.0, align = 'L', txt = 'Hospital number(HN): ' + st.session_state.patient_number, ln = 1)
            pdf.ln(5)
            pdf.cell(page_width, h = 5.0, align = 'L', txt = 'Total Severity Type(TSS): ' + str(st.session_state.TSS), ln = 1)
            pdf.ln(5)
            pdf.cell(page_width, h = 5.0, align = 'L', txt = 'Severity Type: ' + str(st.session_state.Type_), ln = 1)
            pdf.ln(5)
            pdf.cell(page_width, h = 5.0, align = 'L', txt = 'Percentage of Infection Table', ln = 1)
            pdf.ln(3)
            pdf.set_font('Courier', '', 11)
            columnNameList = list(st.session_state.df.columns)
            for header in columnNameList[:-1]:
                pdf.cell(60, 10, header, 1, 0, 'C')
            pdf.cell(60, 10, columnNameList[-1], 1, 2, 'C')
            pdf.cell(-120)
            pdf.set_font('Courier', '', 11)
            for row in range(0, len(st.session_state.df)):
                for col_num, col_name in enumerate(columnNameList):
                    if col_num != len(columnNameList) - 1:
                        pdf.cell(60, 10, str(st.session_state.df['%s' % (col_name)].iloc[row]),1 ,0 , 'C')
                    else:
                        pdf.cell(60, 10, str(st.session_state.df['%s' % (col_name)].iloc[row]),1 ,2 , 'C')
                        pdf.cell(-120)
            pdf.cell(60, 10, "", 0, 2)
            pdf.cell(20)
            col2.download_button(label = 'Save', data = pdf.output(dest='S').encode('latin-1'), file_name = patient_number + '_' + st.session_state.Type_ + '.pdf')

if __name__ == '__main__':
    main()
