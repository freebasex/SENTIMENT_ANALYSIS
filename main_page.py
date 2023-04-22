import streamlit as st
import pandas as pd
import numpy as np
import string
import seaborn as sns
import nltk
from nltk.corpus import stopwords
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import pickle
import speech_recognition as sr
import plotly.express as px
import plotly.graph_objects as go
import io
from my_functions import *


st.title("PROJECT SENTIMENT ANALYSIS")

# Sử dụng mã HTML để định dạng văn bản
st.write(
    '<style>'
    'h1 { font-size: 25px; font-family: Tahoma; font-weight: bold; }'
    'ul { list-style-image: url("https://www.freeiconspng.com/uploads/heart-icon-clip-art--clipart-best-28.png"); }'
    '</style>'
    '<h1> ĐỒ ÁN DATA SCIENCES - MACHINE LEARNING </h1>', unsafe_allow_html=True)

teacher="Người hướng dẫn: Cô Khuất Thùy Phương"
student='Thực hiện: Phạm Thủy Tú - Nguyễn Thị Trần Lộc'

bullet_list1=[teacher,student]

format_bullet(bullet_list1)

st.image('image.jpeg')

df_cleaned=pd.read_csv('data_cleaned_18042023.csv')
df_neg=df_cleaned[df_cleaned['class']==0]
df_pos=df_cleaned[df_cleaned['class']==1]

pkl_filename='LogisticRegression_model.pkl'
with open(pkl_filename,'rb') as file:
    LogisticRegression_model=pickle.load(file)

pkl_filename='tfidf_model.pkl'
with open(pkl_filename,'rb') as file:
    tfidf_model=pickle.load(file)



##LOAD EMOJICON
file = open('files/emojicon.txt', 'r', encoding="utf8")
emoji_lst = file.read().split('\n')
emoji_dict = {}
for line in emoji_lst:
    key, value = line.split('\t')
    emoji_dict[key] = str(value)
file.close()
#################
##LOAD EMOJICON
file = open('files/emojicon.txt', 'r', encoding="utf8")
emoji_lst = file.read().split('\n')
emoji_dict = {}
for line in emoji_lst:
    key, value = line.split('\t')
    emoji_dict[key] = str(value)
file.close()
#################
#LOAD TEENCODE
file = open('files/teencode.txt', 'r', encoding="utf8")
teen_lst = file.read().split('\n')
teen_dict = {}
for line in teen_lst:
    key, value = line.split('\t')
    teen_dict[key] = str(value)
file.close()
###############
#LOAD TRANSLATE ENGLISH -> VNMESE
file = open('files/english-vnmese.txt', 'r', encoding="utf8")
english_lst = file.read().split('\n')
english_dict = {}
for line in english_lst:
    key, value = line.split('\t')
    english_dict[key] = str(value)
file.close()
################
#LOAD wrong words
file = open('files/wrong-word.txt', 'r', encoding="utf8")
wrong_lst = file.read().split('\n')
file.close()
#################
#LOAD STOPWORDS
file = open('files/vietnamese-stopwords.txt', 'r', encoding="utf8")
stopwords_lst = file.read().split('\n')
file.close()

#@st.cache_data
def clean_text(comment):
    # # Xử lý tiếng việt thô
    comment = process_text(comment, emoji_dict, teen_dict, wrong_lst)
    # Chuẩn hóa unicode tiếng việt
    comment = covert_unicode(comment)
    # Kí tự đặc biệt
    comment = process_special_word(comment)
    # postag_thesea
    comment = process_postag_thesea(comment)
    #  remove stopword vietnames
    comment = remove_stopword(comment, stopwords_lst)
    return comment 

st.markdown(
    """
    <style>
    .sidebar .sidebar-content {
        font-family: 'Tahoma';
        font-size: 20px;
        font-weight: bold;
    }
    </style>
    """,
    unsafe_allow_html=True
)
with st.sidebar:
    st.header("SENTIMENT ANALYSIS")
    overview = st.selectbox("**Chức năng**", ["GIỚI THIỆU BÀI TOÁN", "TỔNG QUAN DỮ LIỆU", "PHƯƠNG PHÁP THỰC HIỆN",'SENTIMENT ANALYSIS','ĐÁNH GIÁ KẾT QUẢ','HƯỚNG PHÁT TRIỂN'], key="sidebar1")
    
    if overview:
        st.session_state.sidebar1_option = overview

if st.session_state.sidebar1_option == "GIỚI THIỆU BÀI TOÁN":
    st.subheader("GIỚI THIỆU CHUNG")
    
    with st.expander("**PHÂN TÍCH CẢM XÚC KHÁCH HÀNG**"):
        text1= "Phân tích cảm xúc là quá trình phân tích văn bản kỹ thuật số để xác định xem tin nhắn mang sắc thái cảm xúc tích cực, tiêu cực hay trung lập. Ngày nay, nhiều công ty sở hữu khối lượng lớn dữ liệu văn bản như email, bản ghi cuộc trò chuyện hỗ trợ khách hàng, bình luận trên mạng xã hội và đánh giá. Các công cụ phân tích cảm xúc có thể quét văn bản này để tự động xác định thái độ của tác giả đối với một chủ đề. Các công ty sử dụng thông tin chuyên sâu thu được từ quá trình phân tích cảm xúc để cải thiện dịch vụ khách hàng và tăng độ uy tín cho thương hiệu. "
        text2= "Một hệ thống phân tích cảm xúc giúp các công ty cải thiện sản phẩm và dịch vụ của họ dựa trên phản hồi thật sự và cụ thể của khách hàng."
        text3= "Hệ thống phân tích cảm xúc giúp các doanh nghiệp cải thiện dịch vụ sản phẩm thông qua việc tìm hiểu dịch vụ nào hiệu quả và không hiệu quả. Nhà tiếp thị có thể phân tích bình luận trên các trang đánh giá trực tuyến, câu trả lời khảo sát và bài đăng trên mạng xã hội để thu thập thông tin chuyên sâu hơn về tính năng sản phẩm cụ thể."
        text4= "Các nhà tiếp thị sử dụng nhiều công cụ phân tích cảm xúc để đảm bảo rằng chiến dịch quảng cáo sẽ tạo ra phản ứng mong đợi. Họ theo dõi các cuộc hội thoại trên nhiều nền tảng mạng xã hội với mục tiêu giữ vững trạng thái tích cực trên dịch vụ, sản phẩm họ đang kinh doanh. Nếu cảm xúc thực không đạt kỳ vọng, các nhà tiếp thị sẽ thay đổi chiến dịch dựa trên quá trình phân tích dữ liệu theo thời gian thực."
        
        bullet_list2=[text1,text2,text3,text4]
        format_bullet(bullet_list2)
    
    st.markdown("""
    <div style='border: 2px solid black; padding: 10px; font-family: Tahoma, sans-serif; font-size: 20px; color: blue;text-align: justify;background-color: #dcd0ff;'>
        Shopee là một hệ sinh thái thương mại “all in one”, trong đó có shopee.vn, là một website thương mại điện tử đứng top 1 của Việt Nam và khu vực Đông Nam Á. Dựa trên lịch sử những bình luận và đánh giá của khách hàng đã có trước đó ở https://shopee.vn/...
        => Mục tiêu/ vấn đề: Xây dựng mô hình dự đoán giúp người bán hàng có thể biết được những phản hồi nhanh chóng của khách hàng về sản phẩm hay dịch vụ của họ (tích cực, tiêu cực hay trung tính), điều này giúp cho người bán biết được tình hình kinh doanh, hiểu được ý kiến của khách hàng từ đó giúp họ cải thiện hơn trong dịch vụ, sản phẩm.
    </div>""", unsafe_allow_html=True)

elif st.session_state.sidebar1_option == "TỔNG QUAN DỮ LIỆU":
    st.subheader('TỔNG QUAN VỀ DỮ LIỆU')
    st.write('Nguồn dữ liệu: Products_Shopee_comments.csv')
    
    st.markdown("""
    <style>
        .css-1e3jvgh.e19wd2j90 .st-eb .st-ec {
            font-family: 'Tahoma' !important;
            text-align: justify !important;
            font-size: 20px !important;
        }
    </style>""", unsafe_allow_html=True)

    with st.expander("**THÔNG TIN CHUNG VỀ DỮ LIỆU**"):
        st.code('Bao gồm các thuộc tính: product_id, category, sub_category, user, rating và comment.')
        review_data(df)
        
        df_orginal=read_data()
        df_orginal=df_orginal.dropna()
        df_orginal=df_orginal.drop_duplicates()
        
        text = "<span style='color:black'><b>Số lượng sản phẩm theo từng danh mục</b></span>"
        st.markdown(text, unsafe_allow_html=True)
        df_class=df_orginal['category'].value_counts()
        df_class=df_class.reset_index()
        df_class.rename(columns={'index': 'danh_muc_sp'}, inplace=True)
        df_class.rename(columns={'category': 'sum_of_products'}, inplace=True)
        st.dataframe(df_class)
        labels=df_class['danh_muc_sp']
        values=df_class['sum_of_products']
       # Tạo biểu đồ cột
        fig = go.Figure(data=[go.Bar(x=labels, y=values)])
        fig.update_layout(
            title={
                'text': "Số lượng sản phẩm theo từng danh mục",
                'x':0.5, # canh giữa trên trục x
                'y':0.9, # đặt tiêu đề trên cùng
                'xanchor': 'center', # canh giữa theo trục x
                'yanchor': 'top' # canh trên cùng theo trục y
            }
        )
        # Thiết lập kích thước biểu đồ
        fig.update_layout(width=600, height=500)
        # Hiển thị biểu đồ trong Streamlit
        st.plotly_chart(fig)

        text = "<span style='color:black'><b>Số lượng sản phẩm theo từng phân lớp</b></span>"
        st.markdown(text, unsafe_allow_html=True)
        data_class=df_orginal[['class','product_id']].groupby(['class']).count()
        data_class = data_class.reset_index(drop=False)
        data_class.rename(columns={'product_id': 'sum_of_products'}, inplace=True)
        st.dataframe(data_class)

        labels=['Negative','Postive']
        values=data_class['sum_of_products']
        # Vẽ biểu đồ hình tròn
        
        #fig, ax = plt.subplots()
        #ax.pie(values, labels=labels, autopct='%1.1f%%')
        #ax.axis('equal')
        #plt.title("Số lượng sản phẩm theo từng phân lớp Neg/Pos", fontsize=12,fontweight='bold', fontfamily='Tahoma')

        # Thiết lập kích thước biểu đồ
        
        #fig.set_size_inches(8, 5)

        # Hiển thị biểu đồ trong Streamlit

        #st.pyplot(fig)

        st.text_area('Minh hoạ nội dung 1 phản hồi của khách hàng:',df['comment'].iloc[0])

    with st.expander("**NHẬN XÉT CHUNG VỀ DỮ LIỆU**"):
        text1= "Số dòng dữ liệu giữa hai phân lớp bình luận Tiêu cực/Tích cực không đồng đều"
        text2= "Trong mỗi bình luận, văn bản text còn nhiều lỗi chính tả"
        text3= "Văn bản tồn tại nhiều dấu câu, khoảng trắng, ký tự đặc biệt, icon cảm xúc ..."
        text4= "Văn bản có nhiều stopwords có thể không phục vụ phân tích ..."
        
        bullet_list3=[text1,text2,text3,text4]

        format_bullet(bullet_list3)

        text = "<span style='color:red'><b>Do đó, cần thực hiện tiền xử lý bộ dữ liệu</b></span>"
        st.markdown(text, unsafe_allow_html=True)
    
    with st.expander("**TIỀN XỬ LÝ DỮ LIỆU**"):

        text6= "Có thể lựa chọn và áp dụng một vài phương pháp để cân bằng dữ liệu"
        text7= "Loại bỏ (hoặc thay thế) các từ viết sai chính tả, teencode, từ sai quy cách tiếng Việt, ..."
        text8= "Loại các icon, ký tự đặc biệt, links ..."
        text9= "Loại bỏ stopwords và thêm vào tập tin stopwords các từ phù hợp ..."
        
        bullet_list4=[text6,text7,text8,text9]
        format_bullet(bullet_list4)

        text = "Chỉ giữ lại 3 thuộc tính quan trọng như <span style='color:red'><b>rating, comment, class</b></span>"
        st.markdown(text, unsafe_allow_html=True)
        text = "<span style='color:red'><b>BỘ DỮ LIỆU SAU KHI ĐÃ XOÁ CÁC CỘT KHÔNG CẦN THIẾT: </b></span>"
        st.markdown(text, unsafe_allow_html=True)
        df=df.drop(['product_id', 'category', 'sub_category', 'user'],axis=1)
        review_data(df)
    
    with st.expander("**DỮ LIỆU SAU KHI TIỀN XỬ LÝ**"):
        text = "<span style='color:black'><b>DỮ LIỆU SAU TIỀN XỬ LÝ:</b></span>"
        st.markdown(text, unsafe_allow_html=True)
        review_data(df_cleaned)
        
        text = "<span style='color:red'><b>Kết quả cho thấy có sự mất cân bằng dữ liệu giữa hai phân lớp</b></span>"
        st.markdown(text, unsafe_allow_html=True)
    
    with st.expander("**TRỰC QUAN HOÁ DỮ LIỆU**"):
        
    
        st.write('WordCloud của bình luận tiêu cực')
        
        wc_neg=WordCloud(width=1000, height=600,
            background_color='black',
            max_words=500
        ).generate(str(df_neg['comment'].values))
        fig1, ax = plt.subplots()
        ax.imshow(wc_neg, interpolation='bilinear')
        ax.axis('off')
        st.pyplot(fig1)

        st.write('WordCloud của bình luận tích cực')
        
        wc_pos=WordCloud(width=1000, height=600,
            background_color='black',
            max_words=500
        ).generate(str(df_pos['comment'].values))
        fig2, ax = plt.subplots()
        ax.imshow(wc_pos, interpolation='bilinear')
        ax.axis('off')
        st.pyplot(fig2)

elif st.session_state.sidebar1_option == "PHƯƠNG PHÁP THỰC HIỆN":
    st.subheader('PHƯƠNG PHÁP THỰC HIỆN')
    
    with st.expander("**GIỚI THIỆU TỔNG QUAN PHƯƠNG PHÁP**"):

        text11= "Xác định yêu cầu bài toán"
        text12= "Thu thập dữ liệu và tiền xử lý dữ liệu"
        text13= "Phân tích khám phá dữ liệu"
        text14= "Xác định thuật toán Machine Learning sẽ áp dụng"
        text15= "Phân chia dữ liệu Train và dữ liệu Text"
        text16= "Xây dựng model"
        text17= "Huấn luyện model trên bộ dữ liệu"
        text18= "Đánh giá và lựa chọn mô hình phù hợp"
        text19 = "Phát triển ứng dụng trên Streamlit"

        
        bullet_list5=[text11,text12,text13,text14,text15,text16,text17,text18,text19]

        format_bullet(bullet_list5)

        st.image('step_by_step.png')
    
    with st.expander("**CÁC THUẬT TOÁN TRONG MACHINE LEARNING**"):
        text = "<span style='color:red'><b>KỸ THUẬT phân tích Text - TfidfVectorizer</b></span>"
        st.markdown(text, unsafe_allow_html=True)
        text11= "Tập TRAIN"
        text12= "Tập TEXT"
        text13="Phản hồi nhập vào COMMENTS"

        bullet_list6=[text11,text12,text13]
        format_bullet(bullet_list6)

        text = "<span style='color:red'><b>CÁC THUẬT TOÁN PHÂN LỚP TRONG MACHINE LEARNING</b></span>"
        st.markdown(text, unsafe_allow_html=True)
        data = [
        ['Naive Bayes','68 giây','84 %'],
        ['Logistic Regression','76 giây','86 %'],
        ['Decision Tree','155 giây','72 %'],
        ['K-means','198 giây', '81 %'],
        ['Random Forest','647 giây', '83 %']]

        data=pd.DataFrame(data,columns=['Thuật toán','Thời gian','Độ chính xác'])
        st.table(data)

        text = "<span style='color:red'><b>==> Chọn Logistic Regression dự đoán</b></span>"
        st.markdown(text, unsafe_allow_html=True)
        
        text = "<span style='color:red'><b>Kết quả cho thấy có sự mất cân bằng dữ liệu giữa hai phân lớp</b></span>"
        st.markdown(text, unsafe_allow_html=True)

        st.code('Số dòng dữ liệu thuộc lớp tiêu cực: '+str(len(df_neg)))
        st.code('Số dòng dữ liệu thuộc lớp tích cực: '+str(len(df_pos)))

        data = [
        ['Under-Sampling Data',['Random Undersampling']],
        ['Over-Sampling Data',['SMOTE','ADASYN']],
        ['Synthetic data generation',['GAN','VAE']]]
        data=pd.DataFrame(data,columns=['Phương pháp','Kỹ thuật'])
        st.table(data)

        text = "<span style='color:red'><b>Kết quả LOGISTIC REGRESSION tốt nhất với độ chính xác 86%</b></span>"
        st.markdown(text, unsafe_allow_html=True)
        
    with st.expander("**QUY TRÌNH THỰC HIỆN PROJECT**"):    
        st.image('overview.jpg')

elif st.session_state.sidebar1_option == "SENTIMENT ANALYSIS":
    st.subheader('PHÂN TÍCH CẢM XÚC TRONG PHẢN HỒI TỪ KHÁCH HÀNG')
    
    text = "<span style='color:red'><b>CHỌN DỮ LIỆU PHÂN TÍCH</b></span>"
    st.markdown(text, unsafe_allow_html=True)

    option = st.radio("**Tải tập tin hoặc Nhập bình luận**", options=("Tải tập tin", "Nhập bình luận","Nhập bằng giọng nói"))
    menu=['Tập tin CSV','Tập tin Excel','Tập tin TXT']
    
    if option=="Tải tập tin":
        choice = st.selectbox("**Chọn định dạng tập tin**",menu)
        
        if choice == "Tập tin Excel":
            uploaded_file = st.file_uploader("Vui lòng chọn tập tin **.XLSX**: ")
            if uploaded_file is not None:
                if uploaded_file.name.split('.')[-1] != 'xlsx':
                    st.write('File tải lên sai định dạng, vui lòng chọn đúng file excel')
                else:
                    df_upload = pd.read_excel(uploaded_file)
                    st.dataframe(df_upload)
                    list_result = []
                    for i in range(len(df_upload)):
                        comment = df_upload['review_text'][i]
                        comment = clean_text(comment)
                        comment = tfidf_model.transform([comment])
                        y_predict = LogisticRegression_model.predict(comment)
                        list_result.append(y_predict[0])

                    # apppend list result to dataframe
                    df_upload['sentiment'] = list_result
                    df_after_predict = df_upload.copy()
                    # change sentiment to string
                    y_class = {0: 'Phản hồi tiêu cực', 1: 'Phản hồi tích cực'}
                    df_after_predict['sentiment'] = [y_class[i] for i in df_after_predict.sentiment]

                    # show table result
                    st.subheader("Kết quả phân tích phản hồi:")
                    st.table(df_after_predict)

                    fig, ax = plt.subplots(figsize=(10, 5))

                    ax = sns.countplot(x='sentiment', data=df_after_predict)
                    ax.axis('equal')
                    plt.title("Thống kê phản hồi theo từng phân lớp Tích cực/ Tiêu cực", fontsize=12,fontweight='bold', fontfamily='Tahoma')
                    fig.set_size_inches(10,8)
                    st.pyplot(fig)

                    # download file excel
                    text = "<span style='color:red'><b>Tải tập tin .XLSX kết quả dự đoán</b></span>"
                    st.markdown(text, unsafe_allow_html=True)
                    output = io.BytesIO()
                    writer = pd.ExcelWriter(output)
                    df_after_predict.to_excel(writer, sheet_name='Prediction', index=False)
                    writer.save()
                    output.seek(0)

                    st.download_button('**TẢI KẾT QUẢ**', data=output, file_name='result_excel.xlsx',
                                    mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')

        elif choice == "Tập tin CSV" :
            uploaded_file = st.file_uploader("Vui lòng chọn tập tin **.CSV**: ")
            if uploaded_file is not None:
                if uploaded_file.name.split('.')[-1] != 'csv':
                    st.write('File tải lên sai định dạng, vui lòng chọn đúng file CSV')
                elif uploaded_file.name.split('.')[-1] == 'csv':
                    df_upload = pd.read_csv(uploaded_file)
                    df_upload.drop('Unnamed: 0', axis=1, inplace=True)
                    st.dataframe(df_upload)
                    list_result = []
                    for i in range(len(df_upload)):
                        comment = df_upload['review_text'][i]
                        comment = clean_text(comment)
                        comment = tfidf_model.transform([comment])
                        y_predict = LogisticRegression_model.predict(comment)
                        list_result.append(y_predict[0])

                    df_upload['sentiment'] = list_result
                    df_after_predict = df_upload.copy()
                    # change sentiment to string
                    y_class = {0: 'Phản hồi tiêu cực', 1: 'Phản hồi tích cực'}
                    df_after_predict['sentiment'] = [y_class[i] for i in df_after_predict.sentiment]

                    # show table result
                    st.subheader("Kết quả phân tích phản hồi:")
                    st.table(df_after_predict)

                    fig, ax = plt.subplots(figsize=(10, 7))

                    ax = sns.countplot(x='sentiment', data=df_after_predict)
                    ax.axis('equal')
                    plt.title("Thống kê phản hồi theo từng phân lớp Tích cực/ Tiêu cực", fontsize=12,fontweight='bold', fontfamily='Tahoma')
                    fig.set_size_inches(8, 5)
                    st.pyplot(fig)

                    # download file csv
                    text = "<span style='color:red'><b>Tải tập tin .CSV kết quả dự đoán</b></span>"
                    st.markdown(text, unsafe_allow_html=True)
                    output = io.BytesIO()
                    df_after_predict.to_csv(output, index=False)
                    output.seek(0)
                    st.download_button('**TẢI KẾT QUẢ**', data=output, file_name='result_csv.csv', mime='csv')
    
        elif choice == "Tập tin TXT" :
            uploaded_file = st.file_uploader("Vui lòng chọn tập tin **.TXT**: ")
            if uploaded_file is not None:
                if uploaded_file.name.split('.')[-1] != 'txt':
                    st.write('File tải lên sai định dạng, vui lòng chọn đúng file TXT')
                elif uploaded_file.name.split('.')[-1] == 'txt':
                    df_upload = pd.read_csv(uploaded_file,delimiter="\t", header=None,names=["review_text"])
                    st.dataframe(df_upload)
                    list_result = []
                    for i in range(len(df_upload)):
                        comment = df_upload['review_text'][i]
                        comment = clean_text(comment)
                        comment = tfidf_model.transform([comment])
                        y_predict = LogisticRegression_model.predict(comment)
                        list_result.append(y_predict[0])

                    df_upload['sentiment'] = list_result
                    df_after_predict = df_upload.copy()
                    # change sentiment to string
                    y_class = {0: 'Phản hồi tiêu cực', 1: 'Phản hồi tích cực'}
                    df_after_predict['sentiment'] = [y_class[i] for i in df_after_predict.sentiment]

                    # show table result
                    st.subheader("Kết quả phân tích phản hồi:")
                    st.table(df_after_predict)

                    fig, ax = plt.subplots(figsize=(10, 7))

                    ax = sns.countplot(x='sentiment', data=df_after_predict)
                    ax.axis('equal')
                    plt.title("Thống kê phản hồi theo từng phân lớp Tích cực/ Tiêu cực", fontsize=12,fontweight='bold', fontfamily='Tahoma')
                    st.pyplot(fig)

                    text = "<span style='color:red'><b>Tải tập tin .TXT kết quả dự đoán</b></span>"
                    st.markdown(text, unsafe_allow_html=True)
                    output = io.BytesIO()
                    df_after_predict.to_csv(output, index=False)
                    output.seek(0)
                    st.download_button('**TẢI KẾT QUẢ**', data=output, file_name='result_csv.txt', mime='text')

    elif option == 'Nhập bình luận':
        options = ['Giao hàng chậm', 'Giao hàng nhanh', 
                   'Phản hồi nhanh','Phản hồi chậm',
                   'Hỗ trợ tư vấn nhiệt tình','Tư vấn không nhiệt tình',
                    'Chất lượng hàng kém', 'Chất lượng hàng tốt', 
                    'Hoàn toàn khác hình', 'Sản phẩm giống hình',
                    'Không đúng hàng đã đặt','Đúng hàng đã đặt',
                    'Sản phẩm giống như mô tả','Sản phẩm không giống mô tả',
                    'Chăm sóc khách hàng nhiệt tình','Chăm sóc khách hàng kém',
                    'Rất hài lòng', 'Không hài lòng']

    
    
    # Hiển thị multiselect và text_area
        selected_options = st.multiselect('**Phản hồi nhanh:** ', options)
        input = st.text_area('**Ý kiến khác**')
        options_text = ', '.join(selected_options)
        
        comment = f'{options_text}, {input}'

        st.text_area('**Phản hồi của khách hàng**',value=comment)
        if st.button("**PHÂN TÍCH PHẢN HỒI**"):
            if comment != '':
                comment = clean_text(comment)
                comment = tfidf_model.transform([comment])
                y_predict = LogisticRegression_model.predict(comment)

                if y_predict[0] == 1:
                    st.write('Phản hồi là tích cực')
                else:
                    st.write('Phản hồi là tiêu cực')
            else:
                st.write('Nhập vào một bình luận')
    elif option == 'Nhập bằng giọng nói':
        
        i = 0
        k = 0
        if st.button("Bắt đầu nói",key="batdau"):
            while True:
                comment = hear()
                st.empty()
                st.text_area('**Phản hồi của khách hàng**',value=comment)
                if comment != '':
                    comment = clean_text(comment)
                    comment = tfidf_model.transform([comment])
                    y_predict = LogisticRegression_model.predict(comment)

                    if y_predict[0] == 1:
                        st.write('Phản hồi là tích cực')
                    else:
                        st.write('Phản hồi là tiêu cực')
                break
                #i = i +1
                #if st.button("Ngưng bình luận",key="kethuc_" + str(i)):
                #    break
                
elif st.session_state.sidebar1_option == "ĐÁNH GIÁ KẾT QUẢ":
    st.subheader("MỘT SỐ KẾT QUẢ")
    
    with st.expander("**ƯU ĐIỂM VÀ CÁC KẾT QUẢ ĐẠT ĐƯỢC**"):
        text15= "Hoàn thành các yêu cầu cơ bản của bài toán PHÂN TÍCH CẢM XÚC TỪ PHẢN HỒI"
        text16= "Thực hiện trên nhiều thuật toán MACHINE LEARNING và lựa chọn thuật toán phù hợp nhất với bộ dữ liệu"
        text17= "Vận dụng một số kỹ thuật khác: GridSearchCV - UnderSampling - OverSampling để lựa chọn mô hình phù hợp"
        text18= "Phân tích cảm xúc bằng cả 3 hình thức: tải file phản hồi - nhập văn bản/ giọng nói"
        text19= "Giao diện, bố cục và tốc độ xử lý đã cải thiện"
        
        bullet_list7=[text15,text16,text17,text18]
        format_bullet(bullet_list7)

    with st.expander("**NHƯỢC ĐIỂM VÀ CÁC VẤN ĐỀ CẦN HOÀN THIỆN**"):
        text20= "Bộ dữ liệu sau khi tiền xử lý còn chưa hoàn toàn tốt"
        text21= "Chưa vận dụng DeepLearning"
        text22= "Giao diện và tốc độ xử lý chưa nhanh"
        text23= "Chức năng nhập giọng nói còn chưa ổn định khi xử lý ngôn ngữ tiếng Việt"
        
        bullet_list8=[text20,text21,text22,text23]
        format_bullet(bullet_list8)
elif st.session_state.sidebar1_option == "HƯỚNG PHÁT TRIỂN":

    text30= "Xây dựng dữ liệu StopWord phù hợp hơn"
    text31= "Thực hiện dự đoán vận dụng các giải thuật từ DeepLearning"
    text32= "Xây dựng và thiết kế giao diện web chuyên nghiệp hơn"
    text33= "Cải thiện tốc độ xử lý bằng các kỹ thuật xử lý chuyên nghiệp hơn"
    text34= "Xây dựng bộ tiêu chí phản hồi để hạn chế các lỗi về nhập liệu, cũng như tăng độ chính xác cho rating"
    text35= "Áp dụng tiidf cho bộ từ vựng bao gồm từ đơn và từ ghép"
        
    bullet_list9=[text30,text31,text32,text33,text34,text35]
    format_bullet(bullet_list9)

    
        