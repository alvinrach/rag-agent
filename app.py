import streamlit as st
import os
from beyondllm import source, llms, retrieve, generator
from beyondllm.embeddings import FineTuneEmbeddings
import shutil
import nltk
nltk.download('punkt')

st.title('Talk With Your CV!')
st.markdown("""
            In this web app, you can talk with your CV!

            It can be useful when you don't have enough time to answer all recruiters. Just ask them to talk to this RAG app :D

            Simply upload the CV, then you can ask anything about yourself!
            """)

if not os.path.exists('the_cv.pdf'):
    cv = st.file_uploader('Upload Your CV', type='.pdf')
    if cv:
        save_path = os.path.join('.', 'the_cv.pdf')
        with open(save_path, 'wb') as f:
            f.write(cv.getbuffer())

if os.path.exists('the_cv.pdf'):
    os.environ['GOOGLE_API_KEY'] = st.text_input('Your Gemini Key', key="token", type="password")

    if os.environ['GOOGLE_API_KEY']:
        path = "the_cv.pdf"
        data = source.fit(path, dtype="pdf", chunk_size=10000, chunk_overlap=0)

        llm = llms.GeminiModel()
        fine_tuned_model = FineTuneEmbeddings()

        if not os.path.exists("fintune"):
            embed_model = fine_tuned_model.train([path], "BAAI/bge-small-en-v1.5", llm, "fintune")

        embed_model = fine_tuned_model.load_model("fintune")
        retriever = retrieve.auto_retriever(data, embed_model, type="normal", top_k=4)

        the_question = st.text_input("Ask Something", key="ask", placeholder="Can you give summary about this person?")

        if the_question:
            pipeline = generator.Generate(question=the_question ,retriever=retriever,llm=llm)
            st.markdown(pipeline.call())

            if st.button("Reupload and Retrain CV"):
                if os.path.exists(path):
                    os.remove(path)
                if os.path.exists('train_dataset.json'):
                    os.remove('train_dataset.json')
                if os.path.exists('val_dataset.json'):
                    os.remove('val_dataset.json')
                if os.path.exists("fintune"):
                    shutil.rmtree("fintune")
                st.rerun()