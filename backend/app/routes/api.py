# Standard library imports
import json
import os
import logging
import pickle
from crypt import methods
import time
# Third-party library imports
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.manifold import TSNE
from sklearn.decomposition import KernelPCA

# Flask imports
from flask import (
    Blueprint,
    current_app,
    request,
    jsonify,
    send_from_directory,
    url_for,
    Response,
    stream_with_context
)

# LangChain imports
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
# from langchain_community.chat_models import ChatOpenAI
# from langchain_community.embeddings import OpenAIEmbeddings

# Local application imports
from app.dataService.llm_eval import llm_evaluate_deepeval
import app.dataService.summarize as summ
import base64

LOG = logging.getLogger(__name__)
api = Blueprint('api', __name__)

@api.route('/')
def index():
    print('main url!')
    return json.dumps('/')

@api.route('/files', methods=['GET'])
def get_files():
    files = []
    for filename in os.listdir(current_app.dataService.GV.data_dir):
        if filename.endswith('.pdf'):
            file_path = os.path.join(current_app.dataService.GV.data_dir, filename)
            with open(file_path, 'rb') as file:
                file_data = file.read()
                files.append({
                    "name": filename,
                    "url": request.host_url + 'uploads/' + filename,
                    "raw": "data:application/pdf;base64," + base64.b64encode(file_data).decode('utf-8')
                })
    return jsonify(files)

@api.route('/upload', methods=['POST'])
def upload():
    if 'file' in request.files:
        files = request.files.getlist('file')
        file_urls = []
        for file in files:
            if file:
                filename = file.filename
                # print(" : ", filename)
                file.save(os.path.join(current_app.dataService.GV.data_dir, filename))
                file_urls.append({
                    "url": request.host_url + '/api/uploads/' + filename
                })
        return jsonify(file_urls)
    else:
        return {"message": "No file part in the request"}, 400

@api.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(current_app.dataService.GV.data_dir,  (filename))

@api.route('/images/<filename>')
def serve_image(filename):
    # print("serve_image: ", current_app.dataService.GV.data_dir, "output",  (filename))
    return send_from_directory(current_app.dataService.GV.data_dir, "output/" +  (filename))

@api.route("/extract_meta_from_pdf", methods=["POST"])
def extract_meta_from_pdf():
    data = request.get_json()
    filenames = data["filenames"]
    filepaths = [os.path.join(current_app.dataService.GV.data_dir,  filename["name"]) for filename in filenames]
    # read cached meta info
    metapaths = [os.path.join(current_app.dataService.GV.meta_dir,  filename["name"].split(".")[0] + ".json") for filename in filenames]
    meta_infos = []
    for mpath in metapaths:
        with open(mpath, "r") as f:
            metainfo = json.load(f)
            meta_infos.append(metainfo)
    return meta_infos

@api.route("/extract_table_from_pdf", methods=["POST"])
def extract_table_from_pdf():
    data = request.get_json()
    filenames = data["filenames"]
    filepaths = [os.path.join(current_app.dataService.GV.table_dir,  filename["name"].split(".")[0] + ".json") for filename in filenames]
    ## precomputed table info
    tablInfos = []
    for filepath in filepaths:
        with open(filepath, "r") as f:
            tableInfo = json.load(f)
            tablInfos.append(tableInfo)
    return jsonify(tablInfos)


@api.route("/extract_figure_from_pdf", methods=["POST"])
def extract_figure_from_pdf():
    data = request.get_json()
    filenames = data["filenames"]

    # load preprocessed figure info
    filepaths = [
        os.path.join(
            current_app.dataService.GV.figure_dir,
            filename["name"].split(".")[0] + ".json",
        )
        for filename in filenames
    ]

    figure_infos = []
    for fpath in filepaths:
        with open(fpath, "r") as f:
            figs = json.load(f)
            # update figure_url to serve through the API
            for fig in figs:
                if "figure_url" in fig:
                    # handle absolute paths from any OS
                    figure_name = os.path.basename(fig["figure_url"])
                    fig["figure_url"] = url_for(
                        "api.serve_image",
                        filename=figure_name,
                        _external=True,
                    )
            figure_infos.append(figs)

    return jsonify(figure_infos)

@api.route('/qa', methods=["POST"])
def qa():
    data = request.json
    question = data["question"]
    filenames = [filename["name"] for filename in data["filenames"]]
    summary, ans = current_app.dataService.run_rag_qa(filenames, question, batch_size = 75)

    return jsonify({
        "summary": summary,
        "answer": ans
    })

@api.route('/summarize', methods=["POST"])
def summarize():
    data = request.json
    docs = [
        Document(
            page_content= f"paper title: {d['title']}; paper abstract: {d['abstract']}",
            metadata={"source": d['title']},
        )
        for d in data
    ]
    return jsonify(summ.summarize_docs(docs))


@api.route('/get_confidence_scores', methods=['POST'])
def get_eval_scores():
    data = request.json
    question = data['question']
    answer = str(data['answer'])
    return json.dumps(llm_evaluate_deepeval(metric=['answer_relevancy'], question=question, answer=answer, contexts=""))

if __name__ == '__main__':
    pass
