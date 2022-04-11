import pandas as pd
import string, re
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory, StopWordRemover, ArrayDictionary
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
# MultinomialNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score
# import library evaluation
from sklearn.metrics import f1_score, recall_score, precision_score, confusion_matrix, accuracy_score, classification_report
import pickle
import matplotlib.pyplot as plt
from collections import Counter
import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
from random import randint
import time

# Inisiasi Window
window = tk.Tk()

window.geometry('500x300')
# window.state('zoomed')
window.title("Sentimen Analisis Produk UMKM")

# Fungsi cleansing
def cleansing(data):
    # lower text
    data = data.lower()

    # hapus punctuation
    # remove mentions
    data = re.sub('@[A-Za-z0–9]+', '', data) # Menghapus @mentions
    data = re.sub('#', '', data) # Menghapus '#' hashtag
    data = re.sub('_', '', data) # Menghapus '_' hashtag
    data = re.sub('RT[\s]+', '', data) # Menghapus RT
    data = re.sub('https?:\/\/\S+', '', data) # Menghapus hyperlink
    data = re.sub(r"\d+", "", data) 

    # remove ASCII dan Unicode
    data = data.encode('ascii', 'ignore').decode('utf-8')
    data = re.sub(r'[^\x00-\x7f]', r'', data)

    # remove newline
    data = data.replace('\n', ' ')

    return data

def proses_sent(namafile):

    messagebox.showinfo("Pemberitahuan", "Mohon tunggu, proses akan berlangsung -+ 20 menit..")
    
    # membaca file
    df_review = pd.read_csv(namafile)
    print('Proses pembuatan model akan berlangsung..')
    time.sleep(2)
    print('Proses pengambilan data..')
    time.sleep(1)

    # tambahkan kolom header untuk file
    df_review.columns = ['alamat link', 'tanggal postingan', 'nama akun', 'komentar', 'tanggal komen', 'sentimen']
    print('Proses pemberian nama header..')
    time.sleep(1)

    # buat salinan baru file
    df_preprocessed = df_review.copy()

    # buang header file yang tidak akan dipakai
    df_preprocessed = df_preprocessed.drop(columns=['alamat link', 'tanggal postingan', 'nama akun', 'tanggal komen'])


    # proses resampling
    s_0 = df_preprocessed[df_preprocessed['sentimen']==0].sample(2500,replace=True)
    s_1 = df_preprocessed[df_preprocessed['sentimen']==1].sample(2500,replace=True)
    s_2 = df_preprocessed[df_preprocessed['sentimen']==2].sample(2500,replace=True)
    df_preprocessed = pd.concat([s_0, s_1, s_2])
    print('Proses resampling data..')
    time.sleep(1)

    # jalankan fungsi cleansing pada data kita
    review = []
    print('Proses cleansing data..')
    time.sleep(1)
    for index, row in df_preprocessed.iterrows():
        review.append(cleansing(row['komentar']))

    df_preprocessed['komentar'] = review

    # Inisiasi stemmer
    print('Inisiasi Stemmer..')
    time.sleep(1)
    # Inisiasi Stemmer sastrawi
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()

    # stemm data kita
    review = []
    print('Proses stemming..')
    time.sleep(1)
    for index, row in df_preprocessed.iterrows():
        review.append(stemmer.stem(row["komentar"]))

    df_preprocessed["komentar"] = review

    # inisiasi Stopword Sastrawi
    print('Inisiasi Stopword..')
    time.sleep(1)
    factory = StopWordRemoverFactory().get_stop_words()

    # tambah kata stopword
    sw = []
    swf = open('data/kebutuhan/sw.txt', 'r')
    for u in swf:
        sw.append(u)

    # Tambahkan Stopword Baru
    data = factory + sw
    dictionary = ArrayDictionary(data)
    stopword = StopWordRemover(dictionary)


    # jalankan stopword pada data kita
    review = []
    print('Proses stopword..')
    time.sleep(1)
    for index, row in df_preprocessed.iterrows():
        review.append(stopword.remove(row['komentar']))

    df_preprocessed['komentar'] = review

    # Train test split
    print('Proses pemisahan data latih dan data tes..')
    time.sleep(1)
    X_train, X_test, y_train, y_test = train_test_split(df_preprocessed['komentar'], df_preprocessed['sentimen'],
                                                        test_size=0.1, stratify=df_preprocessed['sentimen'], random_state=42)

    print('Inisiasi TF-IDF..')
    time.sleep(1)
    # Inisiasi TF-IDF Vectorizer
    vectorizer = TfidfVectorizer()

    # tes vectorizer pada dokumen kita
    X_train = vectorizer.fit_transform(X_train)
    X_test = vectorizer.transform(X_test)

    # Naive Bayes Classifier
    print('Inisiasi Naive Bayes Classifier..')
    time.sleep(1)
    modelnb = MultinomialNB()

    # cross validation
    score_val = cross_val_score(modelnb, X_train, y_train, cv=5)

    # lakukan prediksi pada dataset
    modelnb.fit(X_train,y_train)

    # lakukan prediksi pada dataset
    predict = modelnb.predict(X_test)
    # Menentukan probabilitas hasil prediksi
    prob = modelnb.predict_proba(X_test)


    # Nilai EVALUASI HASIL
    # f1 score
    f1 = f1_score(y_test, predict, average='macro')

    # accuracy score
    acs = accuracy_score(y_test, predict) 

    # precision score
    ps = precision_score(y_test, predict, average='macro')

    # recall score
    rs = recall_score(y_test, predict, average='macro')

    # Klasifikasi report
    cr = classification_report(predict, y_test)

    # Confusion Matriks
    neg, net, pos = confusion_matrix(y_test, predict)

    print('Proses pembuatan model telah SELESAI..')
    time.sleep(2)

    msg = messagebox.askokcancel("Simpan model ?","HASIL \nNilai F1 :"+ str(f1) +" \nNilai Accuracy : "+ str(acs) +"\nNilai Precission : "+ str(ps) +"\nNilai Recall : "+ str(rs) +".")

    if msg == 1:
        nm = randint(1, 26)
        with open('data/hasil_model/model_'+ str(nm) +'','wb') as f:
            pickle.dump(f1,f)
            pickle.dump(acs,f)
            pickle.dump(ps,f)
            pickle.dump(rs,f)
            pickle.dump(cr,f)
            pickle.dump(score_val,f)
            pickle.dump(neg,f)
            pickle.dump(net,f)
            pickle.dump(pos,f)
            pickle.dump(vectorizer,f)
            pickle.dump(modelnb,f)

        # simpan hasil ke file .txt
        f = open('data/hasil_model_txt/hasil_model_'+ str(nm) +'.txt', 'a')
        f.write(' ===================== Hasil Evaluasi Model Sentimen Analisis ===================== \n')
        f.write(' \n')
        f.write(' ===================== 1x === 2x ====== 3x ====== 4x ====== 5x \n')
        f.write(' \n')
        f.write(' Score Validation (5x) : '+ str(score_val) +'\n')
        f.write(' \n')
        f.write(' ===================== Hasil Confusion Matrix ===================== \n')
        f.write(' \n')
        f.write(' Negatif == Netral == Positif \n')
        f.write(' \n')
        f.write(' Positif : '+ str(pos) +'\n')
        f.write(' Netral : '+ str(net) +'\n')
        f.write(' Negatif : '+ str(neg) +'\n')
        f.write(' \n')
        f.write(' Nilai F1 scorenya adalah : '+ str(f1) +'\n')
        f.write(' Nilai Accuracy scorenya adalah : '+ str(acs) +'\n')
        f.write(' Nilai Precision scorenya adalah : '+ str(ps) +'\n')
        f.write(' Nilai Recall scorenya adalah : '+ str(rs) +'\n')
        f.write(' \n')
        f.write(' ===================== Hasil Laporan Klasifikasi ===================== \n')
        f.write(' \n')
        f.write(' \n')
        f.write( cr +' \n')
        f.write(' \n')
        f.write(' \n')
        f.close()

        messagebox.showinfo('Berhasil disimpan','model telah disimpan pada folder data/hasil_model/model_'+ str(nm) +' dan pada folder data/hasil_model_txt/hasil_model_'+ str(nm))

    else:
        messagebox.showinfo('model tidak disimpan','Hasil model tidak disimpan.')

def ambilFileDataset(event=None):
    filename = filedialog.askopenfilename(filetypes=[("CSV Template", "*.csv")])
    proses_sent(filename)

def simpan_evaluasi(data):
    komentar_jelek = data
    
    sim = input('simpan seluruh komentar negatif yang terkandung kata : ')
    dicari = []

    for fkn in komentar_jelek:
        if sim in fkn:
            dicari.append(fkn)

    # simpan ke txt
    from random import randrange
    nm = randrange(21)
    f = open('data/hasil_evaluasi/hasil_evaluasi_'+ str(nm) +'.txt', 'a', encoding="utf-8")
    no = 1
    for d in dicari:
        f.write('['+ str(no) +']. '+ str(d) +'\n')
        no+=1

    messagebox.showinfo('Evaluasi disimpan','Hasil evaluasi telah disimpan pada : data/hasil_evaluasi/hasil_evaluasi_'+ str(nm) +'.txt')

def load_model(namaModel,dataset):
    # load model
    with open(namaModel,'rb') as f:
        f1 = pickle.load(f)
        acs = pickle.load(f)
        ps = pickle.load(f)
        rs = pickle.load(f)
        cr = pickle.load(f)
        score_val = pickle.load(f)
        neg = pickle.load(f)
        net = pickle.load(f)
        pos = pickle.load(f)
        vectorizer = pickle.load(f)
        model = pickle.load(f)

    # membaca file
    df_review = pd.read_csv(dataset)
    df_review.columns = ['alamat link', 'tanggal postingan', 'nama akun', 'komentar', 'tanggal komen']

    # inisiasi Stopword Sastrawi
    factory = StopWordRemoverFactory().get_stop_words()

    # tambah kata stopword
    sw = []
    swf = open('data/kebutuhan/sw.txt', 'r')
    for u in swf:
        sw.append(u)

    # Tambahkan Stopword Baru
    data = factory + sw
    dictionary = ArrayDictionary(data)
    stopword = StopWordRemover(dictionary)
    # Akhir Inisiasi SW

    sentimen_pos = 0
    sentimen_net = 0
    sentimen_neg = 0

    # buat array berisi komentar jelek
    komentar_jelek = []
    for df in df_review['komentar']:
        komentar_bersih = cleansing(df)
        komen_sw = stopword.remove(komentar_bersih)
        transformed_komentar = vectorizer.transform([komen_sw])
        prediksi = model.predict(transformed_komentar)
        if prediksi == 0:
            sentimen_neg += 1
            komentar_jelek.append(komen_sw)
        elif prediksi == 1:
            sentimen_net += 1
        else:
            sentimen_pos += 1


    negatif = round(sentimen_neg / len(df_review['komentar']) * 100)
    netral = round(sentimen_net / len(df_review['komentar']) * 100)
    positif = round(sentimen_pos / len(df_review['komentar']) * 100)

    sentimen = ['Positif','Netral', 'Negatif']
    Nilai = [positif,netral,negatif]
    papap = ['Komentar positif : '+ str(sentimen_pos)+' data','Komentar netral : '+ str(sentimen_net)+' data','Komentar negatif : '+ str(sentimen_neg)+' data']
    mycolors = ["green", "yellow", "red"]
    plt.pie(Nilai,labels=sentimen,autopct='%1.2f%%')
    plt.pie(Nilai, labels = sentimen, colors = mycolors)
    patches, texts = plt.pie(Nilai, colors=mycolors, startangle=0)
    plt.legend(patches, papap, loc="best")
    plt.title('Hasil persentase sentimen dari : '+ str(len(df_review['komentar'])) +' data')
    plt.axis('equal')
    plt.tight_layout()
    plt.show()

    frekuensi_komen_neg = Counter(" ".join(komentar_jelek).split()).most_common(10)

    no = 1
    print('10 kata komentar negatif terbanyak : ')
    print('')
    for list_f in frekuensi_komen_neg:
        print('['+ str(no) +']. '+ str(list_f))
        no+=1

    print('================================')
    print('')
    print('[1]. Ya')
    print('[2]. tidak')
    se = input('Simpan evaluasi : ')
    se = se.lower()
    if se == 'ya':
        simpan_evaluasi(komentar_jelek)
        print('Evaluasi telah disimpan.')
    else:
        print('keluar.')


def ambilFileModel(event=None):
    namaModel = filedialog.askopenfilename()
    dataset = filedialog.askopenfilename(filetypes=[("CSV Template", "*.csv")])
    # print('Selected:', filename)
    load_model(namaModel,dataset)


# Isi

# label

# ===================== Frame awal =========================
lb1 = tk.Label(window, text='PILIH MENU', font=('Helvetica', 14, 'bold'))
lb1.pack(pady=40)

# tombol pilihan
bt1 = tk.Button(window, text='Buat model baru', width=30, command=ambilFileDataset)
bt1.pack(pady=5)  
bt2 = tk.Button(window, text='Prediksi dataset', width=30, command=ambilFileModel)
bt2.pack()

# label
lb2 = tk.Label(window, text='---------------', font=('Helvetica', 14))
lb2.pack()

bt3 = tk.Button(window, text='Keluar..', width=30, command=window.destroy)
bt3.pack()  
# Akhir isi
# ===================== Frame awal =========================

# Tutup window
window.mainloop()