import streamlit as st
from PIL import Image
from keras.models import load_model
from function import *
from matplotlib import pyplot as plt
from fpdf import FPDF
from tempfile import NamedTemporaryFile
from datetime import datetime
from datetime import date
import os
from io import BytesIO

def main():
    st.set_option('deprecation.showPyplotGlobalUse', False)


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
    if 'involve' not in st.session_state:
        st.session_state.involve = None
    if 'df' not in st.session_state:
        st.session_state.df = None
    if 'df2' not in st.session_state:
        st.session_state.df2 = None
    if 'overay_re' not in st.session_state:
        st.session_state.overay_re = None
    if 'summa' not in st.session_state:
        st.session_state.summa = None

    # page setting
    st.set_page_config(page_title = 'CSC', layout = 'wide')
    st.title('COVID-19 Severity Calculator (CSC)')

    st.sidebar.title('INFORMATION')
    with st.sidebar.expander('See more Description'):
        st.write('''This is web application for classification severity of COVID-19 patient levels
that Normal, Mild, Moderate or Severe.''')
        st.write('------------------------')
        st.write('RUL: Right Upper Lobe')
        st.write('RLL: Right Lower Lobe')
        st.write('RML: Right Middle Lobe')
        st.write('LUL: Left Upper Lobe')
        st.write('LLL: Left Lower Lobe')
        st.write('Infection (%): Percentage of Infection')

    st.session_state.patient_number = st.sidebar.text_input('Please Fill the Hospital Number', placeholder = 'Hospital Number')
    upload_file = st.sidebar.file_uploader('Choose CT scan File', type = ['jpg'], accept_multiple_files = True)
    x = upload_file[0]
    print(x)
    place_container = st.empty()
    predic, _, save_pdf = st.sidebar.columns([1, 2, 1])
    _, link_html = st.sidebar.columns([2, 2])

    if len(upload_file) == 0 :
        st.sidebar.warning('Please Choose the CT-Scan Files')

    elif len(upload_file) > 256 :
        st.warning('Please upload the CT-Scan Files less than 257 images')

    elif st.session_state.lesion_result is None:
        with place_container.container():
            img_index = st.slider('', min_value = 1, max_value = len(upload_file), key = 'pre_sli')-1
            _, img_col, _ = st.columns([1, 2, 1])
            img = Image.open(upload_file[img_index])
            img_col.image(img, use_column_width = 'always')
    else:
        place_container.empty()
        with place_container.container():
            _, col1 = st.columns([2, 6])
            if st.session_state.raw_img3d.shape[0] <= 175:
                img_index = col1.slider('', min_value = 1, max_value = len(upload_file), key = 'pro_sli')-1
                img = Image.open(upload_file[img_index])
                fig = overay(st.session_state.lesion_result, st.session_state.raw_img3d, img_index)
                col_1, col_2, col_3 = st.columns([2, 3, 3])
                col_3.pyplot(fig)
                col_2.image(img, use_column_width = 'always')

                # result
                col_1.metric(label = 'Total Severity Score', value = st.session_state.TSS)
                col_1.metric(label = 'Severity Type', value = st.session_state.Type_)
                col_1.metric(label = 'Lung Involvement', value = st.session_state.involve)
                col_1.dataframe(st.session_state.df2)

            elif st.session_state.raw_img3d.shape[0] <= 256:
                img_index = col1.slider('', min_value = 1, max_value = int(len(upload_file)/2), key = 'pro_sli')-1
                img = Image.open(upload_file[::2][img_index])
                fig = overay(st.session_state.lesion_result, st.session_state.raw_img3d[::2, :, :], img_index)
                col_1, col_2, col_3 = st.columns([2, 3, 3])
                col_3.pyplot(fig)
                col_2.image(img, use_column_width = 'always')

                # result
                col_1.metric(label = 'Total Severity Score', value = st.session_state.TSS)
                col_1.metric(label = 'Severity Type', value = st.session_state.Type_)
                col_1.metric(label = 'Lung Involvement', value = st.session_state.involve)
                col_1.dataframe(st.session_state.df2)
            else:
                img_index = col1.slider('', min_value = 1, max_value = len(upload_file), key = 'pro_sli')-1
                img = Image.open(upload_file[img_index])
                st.image(img, use_column_width = 'always')

            # col2.subheader('Results')

    if predic.button('Predict'):
        # input file to 3d
        raw_img3d = load_imgdata(upload_file, 512)
        st.session_state.raw_img3d = raw_img3d

        image_3d = load_imgdata(upload_file, 256)
        st.session_state.image_3d = image_3d

        # Load lung model ---------------------------
        current_path = os.getcwd() # getting the current path
        lung_model_path = os.path.join(current_path, 'Lung_Model5.h5')
        lung_model = load_model(lung_model_path, compile = False)

        #setting parameter
        BACKBONE = 'densenet169'
        padding_size = 128

        #lung prediction
        lung_result = predict(lung_model, BACKBONE, image_3d, padding_size)

        # Load lesion model ---------------------------
        lesion_model_path = os.path.join(current_path, 'Lesion_Model43.h5')
        lesion_model = load_model(lesion_model_path, compile = False)

        # image croped and predict
        # img_crop = crop_img(image_3d, lung_result)
        img_crop_ = crop_img(image_3d, lung_result)

        # CLAHE (contrast adjustment)
        img_crop = contrast_CLAHE(img_crop_, 3)
        # Predict
        lesion_result = predict(lesion_model, BACKBONE, img_crop, padding_size)
        st.session_state.lesion_result = lesion_result

        # TSS version 1
        score_lobe, score_lesion_lobe = sum_pixel(lung_result, lesion_result)

        TSS, Type_, involve, df, df2 = TSS_score(score_lobe, score_lesion_lobe)
        st.session_state.TSS = TSS
        st.session_state.Type_ = Type_
        st.session_state.involve = involve
        st.session_state.df = df
        st.session_state.df2 = df2

        place_container.empty()
        with place_container.container():
            col2, col1 = st.columns([2, 6])
            if raw_img3d.shape[0] <= 175:
                img_index = col1.slider('', min_value = 1, max_value = len(upload_file), key = 'pro_sli')-1
                img = Image.open(upload_file[img_index])
                fig = overay(st.session_state.lesion_result, st.session_state.raw_img3d, img_index)

            elif raw_img3d.shape[0] <= 256:
                img_index = col1.slider('', min_value = 1, max_value = int(len(upload_file)/2), key = 'pro_sli')-1
                img = Image.open(upload_file[::2][img_index])
                fig = overay(st.session_state.lesion_result, st.session_state.raw_img3d[::2, :, :], img_index)
            else:
                img_index = col1.slider('', min_value = 1, max_value = len(upload_file), key = 'pro_sli')-1
                img = Image.open(upload_file[img_index])

            # col2.subheader('Results')

            col_1, col_2, col_3 = st.columns([2, 3, 3])
            col_3.pyplot(fig)
            col_2.image(img, use_column_width = 'always')

            # result
            col_1.metric(label = 'Total Severity Score', value = st.session_state.TSS)
            col_1.metric(label = 'Severity Type', value = st.session_state.Type_)
            col_1.metric(label = 'Lung Involvement', value = st.session_state.involve)
            col_1.dataframe(st.session_state.df2)

    if save_pdf.button('Save'):
        pdf = FPDF()
        pdf = FPDF(orientation = 'P', unit = 'mm', format = 'A4')
        pdf.add_page()
        page_width = pdf.w - 2*pdf.l_margin
        pdf.ln(10)
        pdf.set_font('Courier', 'B', 16)
        pdf.cell(page_width, 0.0, align = 'C', txt = '''COVID-19 Patient's Information''', ln = 1)
        pdf.ln(2)
        pdf.cell(0, 0, txt = '_'*55)
        pdf.ln(10)
        pdf.set_font('Courier', '', 14)
        pdf.cell(page_width, h = 5.0, align = 'L', txt = 'Date: ' + date.today().strftime('%b-%d-%Y') + '   Time:' + datetime.now().strftime('%H.%M.%S'), ln = 1)
        pdf.ln(5)
        pdf.cell(page_width, h = 5.0, align = 'L', txt = 'Hospital Number(HN): ' + st.session_state.patient_number, ln = 1)
        pdf.ln(5)
        pdf.cell(page_width, h = 5.0, align = 'L', txt = 'Total Severity Type(TSS): ' + str(st.session_state.TSS), ln = 1)
        pdf.ln(5)
        pdf.cell(page_width, h = 5.0, align = 'L', txt = 'Severity Type: ' + str(st.session_state.Type_), ln = 1)
        pdf.ln(5)
        pdf.cell(page_width, h = 5.0, align = 'L', txt = 'Lung Involvement: ' + str(st.session_state.involve), ln = 1)
        pdf.ln(5)
        pdf.set_font('Courier', 'B', 14)
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
                    pdf.cell(60, 10, str(st.session_state.df['%s' % (col_name)].iloc[row]), 1, 0, 'C')
                else:
                    pdf.cell(60, 10, str(st.session_state.df['%s' % (col_name)].iloc[row]), 1, 2, 'C')
                    pdf.cell(-120)
        pdf.cell(60, 10, '', 0, 2)
        pdf.set_font('Courier', 'B', 14)
        pdf.cell(page_width, h = 5.0, align = 'L', txt = 'Example Lung Image', ln = 1)
        pdf.ln(10)
        with NamedTemporaryFile(delete=False, suffix=".png") as tmpfile,NamedTemporaryFile(delete=False, suffix=".jpg") as rawimg:
            if st.session_state.raw_img3d.shape[0] <= 175:
                fig = overay(st.session_state.lesion_result, st.session_state.raw_img3d, 20)
                rawimg.write(upload_file[20].getvalue())
            elif st.session_state.raw_img3d.shape[0] <= 256:
                fig = overay(st.session_state.lesion_result, st.session_state.raw_img3d[::2,:,:], int(20/2))
                rawimg.write(upload_file[int(20/2)].getvalue())
            fig.savefig(tmpfile.name)
            pdf.image(rawimg.name, x = 25, y = 170, w = 50, h = 50)
            pdf.image(tmpfile.name, x = 25, y = 220, w = 50, h = 50)
        with NamedTemporaryFile(delete=False, suffix=".png") as tmpfile,NamedTemporaryFile(delete=False, suffix=".jpg") as rawimg:
            if st.session_state.raw_img3d.shape[0] <= 175:
                fig = overay(st.session_state.lesion_result, st.session_state.raw_img3d, 40)
                rawimg.write(upload_file[40].getvalue())
            elif st.session_state.raw_img3d.shape[0] <= 256:
                fig = overay(st.session_state.lesion_result, st.session_state.raw_img3d[::2,:,:], int(40/2))
                rawimg.write(upload_file[int(40/2)].getvalue())
            fig.savefig(tmpfile.name)
            pdf.image(rawimg.name, x = 80, y = 170, w = 50, h = 50)
            pdf.image(tmpfile.name, x = 80, y = 220, w = 50, h = 50)
        with NamedTemporaryFile(delete=False, suffix=".png") as tmpfile,NamedTemporaryFile(delete=False, suffix=".jpg") as rawimg:
            if st.session_state.raw_img3d.shape[0] <= 175:
                fig = overay(st.session_state.lesion_result, st.session_state.raw_img3d, 60)
                rawimg.write(upload_file[60].getvalue())
            elif st.session_state.raw_img3d.shape[0] <= 256:
                fig = overay(st.session_state.lesion_result, st.session_state.raw_img3d[::2,:,:], int(60/2))
                rawimg.write(upload_file[int(60/2)].getvalue())
            fig.savefig(tmpfile.name)
            pdf.image(rawimg.name, x = 135, y = 170, w = 50, h = 50)
            pdf.image(tmpfile.name, x = 135, y = 220, w = 50, h = 50)
        name_file = st.session_state.patient_number + '_' + st.session_state.Type_
        stream = BytesIO()
        st.session_state.df.set_index('Lobe').T.to_csv(stream, encoding = 'utf-8')
        html1 = create_download_link(pdf.output(dest = 'S').encode('latin-1'), name_file + '.pdf', 'PDF File')
        html2 = create_download_link(stream.getvalue(), name_file+'.csv', 'CSV File')
        link_html.markdown(html1, unsafe_allow_html = True)
        link_html.markdown(html2, unsafe_allow_html = True)

if __name__ == '__main__':
    main()
