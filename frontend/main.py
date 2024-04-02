
import streamlit as st
from streamlit_option_menu  import option_menu
import requests




m = st.markdown("""
    <style>
    div.stButton > button:first-child {
        background-color: #1c78bd;
        color:white;
        border: none;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 58px;
        cursor: pointer;
        width: 705px;
        height: 70px;
        
    }
    
    
    </style>""", unsafe_allow_html=True)


with st.sidebar:
    selected = option_menu(
            menu_title=None,  # required
            options=["Data haqqında məlumat", "Sual cavab sistemi","Model metrics"],  # required
            icons=["info-circle", "question-circle","calculator"],  # optional
            menu_icon="cast",  # optional
            default_index=0,  # optional
            #orientation="horizontal",
            styles={
                "container": {"padding": "0!important", "background-color": "#ffffff"},
                "icon": {"color": "#09446f", "font-size": "25px"},
                "nav-link": {
                    "font-size": "25px",
                    "text-align": "left",
                    "margin": "0px",
                    "--hover-color": "#a3c8e4",
                },
                "nav-link-selected": {"background-color": "#1c78bd"},
            },
        )

c1 = st.container()
c2 = st.container()
if selected=="Data haqqında məlumat":
    st.title(f'Data haqqında məlumat.')
    st.header('SQuAD Azerbaijani Dataset')

    st.subheader("Dataset")
    st.write("https://huggingface.co/datasets/vrashad/squad_azerbaijan/viewer/default/train")

    st.subheader('Description')
    st.write("This dataset is the Azerbaijani version of the Stanford Question Answering Dataset (SQuAD), automatically translated from the original English dataset. SQuAD is a prominent dataset in natural language processing, used for machine comprehension and question-answering tasks. It consists of questions based on a set of Wikipedia articles, where the answer to each question is a segment of text from the corresponding article.")

    st.subheader('Dataset Structure')
    st.subheader('Data Fields')
    st.write("- **id:** A unique identifier for each question-answer pair.")
    st.write("- **title:** The title of the Wikipedia article from which the context is extracted.")
    st.write("- **context:** A segment of text from the Wikipedia article that contains the information necessary to answer the question.")
    st.write("- **question:** The question posed, translated into Azerbaijani.")
    st.write("- **answers:** A list containing:")
    st.write("  - **text:** The segment of text that answers the question.")
    st.write("  - **answer_start:** The position of the answer's first character in the context.")

    st.subheader('Data Splits')
    st.write("The dataset is split into two subsets: train and test. The train subset is used for training models, while the test subset is for validating and testing them.")

    st.subheader('Licensing Information')
    st.write("This work is licensed under a Creative Commons Attribution Non-Commercial 2.0 Generic License (CC BY-NC 2.0). This license allows others to remix, tweak, and build upon this work non-commercially, as long as they credit the creator and license their new creations under the identical terms.")

    st.subheader('Citation')
    st.write("Please cite the following paper when using this dataset:")
    st.write("[Original SQuAD Paper Citation]")

    st.subheader('Acknowledgements')
    st.write("This dataset was created by [Valiyev Rashad], based on the Stanford Question Answering Dataset [https://rajpurkar.github.io/SQuAD-explorer/].")

    st.markdown("---")
    st.text("© 2024 Azərbaycan Sual Cavab ")

if selected=="Sual cavab sistemi":



    st.title(f'Sual cavab sisteminə xoş gəldiniz.')



    st.subheader('Mətni daxil edin:')
    user_context =st.text_area (' ',height=200)

    st.subheader('Sualınızı daxil edin:')
    user_question =st.text_area ('  ', height=200)


    #button_html = f'<span style="{button_style} font-size: 26px;">cavabla</span>'
    #button_html = f'<span style="{button_style}">Click me</span>'

    # button1 = st.write(
    #     f'<button style="{button_style}"> Cavabla</button>',
    #     unsafe_allow_html=True
    # )
    button1= st.button("cavabla")



    #button1=st.markdown(button_html, unsafe_allow_html=True)

    if button1:
        payload = {"context":user_context,"question":user_question}
        response=requests.post("http://api:8080/predict",json=payload)


        st.write(response.json()["asnwer"])
        st.subheader(f'SCORE:{response.json()["score"]}')


if selected=="Model metrics":
    response=requests.get("http://api:8080/metrics")
    st.subheader(f'F1 Score:{response.json()["f1 score"]}')
    st.subheader(f'BLEU Score:{response.json()["BLEU"]}')



