from sklearn.preprocessing import StandardScaler
import csv,emoji,unicodedata,time,pandas,numpy,re,os,seaborn as sns,matplotlib.pyplot as plt,torch,scipy.sparse as sp
from sklearn.model_selection import PredefinedSplit
from sklearn.decomposition import TruncatedSVD
from transformers import AutoTokenizer,AutoModel
from nltk.corpus import stopwords
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split,RandomizedSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.metrics import f1_score,precision_score,recall_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report,accuracy_score,confusion_matrix
from scipy.sparse import hstack

normalization_map={"إ":"ا","أ":"ا","آ":"ا","ٱ":"ا","ى":"ي","ؤ":"ء","ئ":"ء","ة":"ه","گ":"ك","ڪ":"ك","ڬ":"ك","ڤ":"ف","ڧ":"ق","چ":"ج","پ":"ب","ڜ":"ش","ۂ":"ه","ۀ":"ه"}
normalization_table=str.maketrans(normalization_map)
stopWords=set(stopwords.words('arabic'))
ARABIC_LETTERS=("\u0621-\u064A""\u066E-\u066F""\u0671-\u06D3""\u06FA-\u06FC""\u0750-\u077F""\u08A0-\u08FF")
TASHKEEL="\u064B-\u065F\u0670\u06D6-\u06ED"
EMOJI_BLOCKS=("\U0001F1E0-\U0001F1FF""\U0001F300-\U0001FAFF""\U00002600-\U000026FF""\U00002700-\U000027BF")
BIDI_MARKS=r"[\u200e\u200f\u202a-\u202e\u2066-\u2069]"

EMOTICON_MAP={":-)":"ابتسامة",":)":"ابتسامة","=)":"ابتسامة","(:":"ابتسامة","^_^":"سعادة","^^":"سعادة","☆":"نجمة","♡":"قلب حب","☺":"ابتسامة","😊":"ابتسامة","☻":"ابتسامة",":-D":"ضحك",":D":"ضحك","=D":"ضحك","xD":"ضحك","XD":"ضحك",":-(":"حزن",":(":"حزن","=(":"حزن","):":"حزن","T_T":"بكاء","TT":"بكاء",":'(":"بكاء",":'-(":"بكاء",";-)":"غمزة",";)":"غمزة",":-P":"لسان",":P":"لسان",":-p":"لسان",":p":"لسان",":-O":"دهشة",":O":"دهشة",":-o":"دهشة",":o":"دهشة","-_-":"ملل","_-_":"ملل","=_=":"ملل",">:(":"غضب","D:<":"غضب","<3":"حب","</3":"قلب_مكسور"}
emojiKey=sorted(EMOTICON_MAP.keys(),key=len,reverse=True)
replaceEmoji=re.compile("|".join(re.escape(k) for k in emojiKey))
def _rep(m):
    return f" {EMOTICON_MAP[m.group(0)]} "

def removeNonArabic(text):
    text=unicodedata.normalize("NFKC",text)
    text=re.sub(BIDI_MARKS,"",text)
    text=text.replace("\u0640","")
    text=re.sub(f"[{TASHKEEL}]","",text)
    text=re.sub(r"\.{3,}|…{1,}"," تكملة ",text)
    text=re.sub(r"[؟\?]+"," سؤال ",text)
    text=re.sub(r"!+"," تعجب ",text)
    keep=ARABIC_LETTERS+EMOJI_BLOCKS+"\u200D\uFE0F "
    text=re.sub(f"[^{keep}]+"," ",text)
    text=re.sub(r"\s+"," ",text).strip()
    return text

def removeDuplicates(text):
    text=re.sub(r"\s+"," ",text).strip()
    text=re.sub(r"(.)\1{2,}",r"\1\1",text)
    return text

def addSpacesBetweenSpecificWords(text):
    text=re.sub(r"ة(?=\S)","ة ",text)
    text=re.sub(r"ء(?!ة)(?=\S)","ء ",text)
    text=re.sub(r"(?<!\S)ب\s+(\S)",r"ب\1",text)
    text=re.sub(r"(?<!\S)ل\s+(\S)",r"ل\1",text)
    return text

def preprocessingTextForTF_IDF(text):
    text=emoji.demojize(text,delimiters=("ايموجي "," "),language='ar')
    text=replaceEmoji.sub(_rep,text)
    text=removeNonArabic(text)
    text=addSpacesBetweenSpecificWords(text)
    text=text.translate(normalization_table)
    text=removeDuplicates(text)
    return text

def preprocessForTransformer(text):
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(BIDI_MARKS, "", text)
    text = text.replace("\u0640", "")
    text = re.sub(r"(https?://\S+|www\.\S+)", " رابط ", text)
    text = re.sub(r"\b[\w\.-]+@[\w\.-]+\.\w+\b", " بريد ", text)
    text = re.sub(r"@\w+", " حساب ", text)
    text = emoji.demojize(text, delimiters=("ايموجي ", " "), language='ar')
    text = replaceEmoji.sub(_rep, text)
    keep_for_transformer = ARABIC_LETTERS + EMOJI_BLOCKS + "\u200D\uFE0F !?.,؛،:"
    text = re.sub(f"[^{keep_for_transformer}]+", " ", text)
    text = re.sub(f"[{TASHKEEL}]", "", text)
    text=addSpacesBetweenSpecificWords(text)
    text = text.translate(normalization_table)
    text=removeDuplicates(text)
    return text

def meanPool(last_hidden,attention_mask):
    mask=attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
    sum=(last_hidden*mask).sum(dim=1)
    count=mask.sum(dim=1).clamp(min=1e-9)
    return sum/count

def evaluateModel(name,model,X_train,X_test,y_train,y_test,save_cm_name,fit_kwargs=None):
    print("\n"+"="*60+name+"="*60)
    if fit_kwargs is None:
        fit_kwargs={}
    model.fit(X_train,y_train,**fit_kwargs)
    y_pred=model.predict(X_test)
    acc=accuracy_score(y_test,y_pred)
    prec=precision_score(y_test,y_pred,average="macro",zero_division=0)
    rec=recall_score(y_test,y_pred,average="macro",zero_division=0)
    f1=f1_score(y_test,y_pred,average="macro",zero_division=0)
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-Score : {f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test,y_pred,digits=4,zero_division=0))
    labels=sorted(set(y_test.tolist()))
    cm=confusion_matrix(y_test,y_pred,labels=labels)
    plt.figure(figsize=(7,6))
    sns.heatmap(cm,annot=True,fmt="d",xticklabels=labels,yticklabels=labels)
    plt.title(f"Confusion Matrix: {name}")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(save_cm_name)
    plt.show()
    print("Saved:",save_cm_name)
    return y_pred,{"accuracy":acc,"precision":prec,"recall":rec,"f1_score":f1}

def printLeaderboard(rows):
    dataFrame=pandas.DataFrame(rows).sort_values("f1_score",ascending=False)
    with pandas.option_context("display.width",240,"display.max_columns",None):
        print("\n"+"="*80)
        print("PERFORMANCE LEADERBOARD (sorted by F1-Score)")
        print("="*80)
        print(dataFrame.to_string(index=False,float_format=lambda x:f"{x:.4f}"))
    return dataFrame

def plotPerformanceGraphs(rows):
    dataFrame=pandas.DataFrame(rows).sort_values("f1_score",ascending=False)
    fig1=plt.figure(figsize=(14,6))
    ax1=plt.subplot(1,2,1)
    metrics=['accuracy','precision','recall','f1_score']
    x=numpy.arange(len(dataFrame))
    width=0.2
    colors=['#3498DB','#E74C3C','#2ECC71','#F39C12']
    for i,metric in enumerate(metrics):
        bars=ax1.bar(x+i*width,dataFrame[metric],width,label=metric.replace('_',' ').title(),color=colors[i])
        for bar in bars:
            height=bar.get_height()
            ax1.text(bar.get_x()+bar.get_width()/2.,height,f'{height:.3f}',
                    ha='center',va='bottom',fontsize=8)
    ax1.set_xlabel('Models',fontsize=12,fontweight='bold')
    ax1.set_ylabel('Score',fontsize=12,fontweight='bold')
    ax1.set_title('Model Performance Metrics Comparison',fontsize=14,fontweight='bold')
    ax1.set_xticks(x+1.5*width)
    ax1.set_xticklabels(dataFrame['model'],rotation=15,ha='right')
    ax1.legend(loc='lower right')
    ax1.grid(axis='y',alpha=0.3)
    ax1.set_ylim(0,1.1)
    ax2=plt.subplot(1,2,2)
    colors_time=['#9B59B6','#1ABC9C','#E67E22']
    bars=ax2.barh(dataFrame['model'],dataFrame['time_seconds'],color=colors_time)
    ax2.set_xlabel('Time (seconds)',fontsize=12,fontweight='bold')
    ax2.set_title('Training Time Comparison',fontsize=14,fontweight='bold')
    ax2.grid(axis='x',alpha=0.3)
    for i,bar in enumerate(bars):
        width=bar.get_width()
        minutes=int(width//60)
        seconds=int(width%60)
        time_str=f"{minutes}m{seconds}s" if minutes>0 else f"{seconds}s"
        ax2.text(width,bar.get_y()+bar.get_height()/2,f' {time_str}',
                va='center',fontsize=10,fontweight='bold')
    plt.tight_layout()
    plt.savefig('Figure_1_Performance_Metrics.png',dpi=300,bbox_inches='tight')
    plt.show()
    print("Saved: Figure_1_Performance_Metrics.png")
    fig2=plt.figure(figsize=(14,6))
    ax3=plt.subplot(1,2,1)
    x_pos=numpy.arange(len(dataFrame))
    ax3.plot(x_pos,dataFrame['accuracy'],'o-',label='Accuracy',linewidth=2.5,markersize=10,color='#3498DB')
    ax3.plot(x_pos,dataFrame['precision'],'s-',label='Precision',linewidth=2.5,markersize=10,color='#E74C3C')
    ax3.plot(x_pos,dataFrame['recall'],'^-',label='Recall',linewidth=2.5,markersize=10,color='#2ECC71')
    ax3.plot(x_pos,dataFrame['f1_score'],'d-',label='F1-Score',linewidth=2.5,markersize=10,color='#F39C12')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(dataFrame['model'],rotation=15,ha='right')
    ax3.set_ylabel('Score',fontsize=12,fontweight='bold')
    ax3.set_title('Performance Metrics Trend',fontsize=14,fontweight='bold')
    ax3.legend(loc='best',fontsize=10)
    ax3.grid(True,alpha=0.3)
    ax3.set_ylim(0,1)
    for i,row in dataFrame.iterrows():
        ax3.text(list(dataFrame.index).index(i),row['f1_score']+0.02,f"{row['f1_score']:.3f}",
                ha='center',fontsize=8,fontweight='bold')
    ax4=plt.subplot(1,2,2)
    colors_f1=['#27AE60' if i==0 else '#3498DB' if i==1 else '#95A5A6' for i in range(len(dataFrame))]
    bars=ax4.barh(dataFrame['model'],dataFrame['f1_score'],color=colors_f1)
    ax4.set_xlabel('F1-Score',fontsize=12,fontweight='bold')
    ax4.set_title('F1-Score Ranking',fontsize=14,fontweight='bold')
    ax4.set_xlim(0,1)
    ax4.grid(axis='x',alpha=0.3)
    for i,(idx,row) in enumerate(dataFrame.iterrows()):
        ax4.text(row['f1_score'],i,f" {row['f1_score']:.4f}",
                va='center',fontsize=11,fontweight='bold')
        rank_text=['1st','2nd','3rd'][i] if i<3 else f"{i+1}th"
        ax4.text(0.02,i,rank_text,va='center',fontsize=9,color='white',fontweight='bold')
    plt.tight_layout()
    plt.savefig('Figure_2_Metrics_Analysis.png',dpi=300,bbox_inches='tight')
    plt.show()
    print("Saved: Figure_2_Metrics_Analysis.png")
    fig3,ax5=plt.subplots(figsize=(12,6))
    ax5.axis('tight')
    ax5.axis('off')
    table_data=[]
    for idx,row in dataFrame.iterrows():
        minutes=int(row['time_seconds']//60)
        seconds=int(row['time_seconds']%60)
        time_str=f"{minutes}m {seconds}s" if minutes>0 else f"{seconds}s"
        table_data.append([row['model'],f"{row['accuracy']:.4f}",f"{row['precision']:.4f}",f"{row['recall']:.4f}",f"{row['f1_score']:.4f}",time_str])
    table=ax5.table(cellText=table_data,colLabels=['Model','Accuracy','Precision','Recall','F1-Score','Time'],cellLoc='center',loc='center',colWidths=[0.3,0.14,0.14,0.14,0.14,0.14])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1,3)
    for i in range(6):
        table[(0,i)].set_facecolor('#34495E')
        table[(0,i)].set_text_props(weight='bold',color='white',size=12)
    for i in range(1,len(table_data)+1):
        if i==1:
            for j in range(6):
                table[(i,j)].set_facecolor('#D5F4E6')
        elif i%2==0:
            for j in range(6):
                table[(i,j)].set_facecolor('#ECF0F1')
        table[(i,4)].set_text_props(weight='bold',color='#E74C3C')
    ax5.set_title('Performance Summary Table',fontsize=16,fontweight='bold',pad=20)
    plt.tight_layout()
    plt.savefig('Figure_3_Summary_Table.png',dpi=300,bbox_inches='tight')
    plt.show()
    print("Saved: Figure_3_Summary_Table.png")

def get_transformer_embeddings(texts,model_name="UBC-NLP/MARBERT",max_length=64,batch_size=32,cache_path=None):
    if cache_path is not None and os.path.exists(cache_path):
        return numpy.load(cache_path)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer=AutoTokenizer.from_pretrained(model_name)
    model=AutoModel.from_pretrained(model_name).to(device)
    model.eval()
    all_vecs=[]
    t0=time.perf_counter()
    with torch.no_grad():
        for i in range(0,len(texts),batch_size):
            batch=texts[i:i+batch_size]
            enc=tokenizer(list(batch),padding=True,truncation=True,max_length=max_length,return_tensors="pt")
            enc={k:v.to(device) for k,v in enc.items()}
            out=model(**enc)
            vec=meanPool(out.last_hidden_state,enc["attention_mask"])
            all_vecs.append(vec.detach().cpu().numpy())
    X=numpy.vstack(all_vecs).astype(numpy.float32)
    seconds=time.perf_counter()-t0
    print(f"\n[Transformer embeddings] model={model_name} | device={device.type} | samples={len(texts)} | time={seconds:.2f}s")
    if cache_path is not None:
        numpy.save(cache_path,X)
    return X

def nb_val_f1(alpha,prior=None):
    if prior is None:
        nb=MultinomialNB(alpha=float(alpha),fit_prior=True)
    else:
        nb=MultinomialNB(alpha=float(alpha),fit_prior=False,class_prior=prior)
    nb.fit(mixTrain,labelTrainText)
    pred=nb.predict(mixValidation)
    return f1_score(labelValidationText,pred,average="macro",zero_division=0)

def zoom_best_alpha(alphaMin=1e-4,alphaHigh=1.0,numOfPoints=15,max_rounds=5,tol=1e-3,min_width=1e-3):
    bestTempAlpha=alphaMin
    bestFScore=-1.0
    tempbestScore=None
    for _ in range(max_rounds):
        grid=numpy.linspace(alphaMin,alphaHigh,numOfPoints)
        scores=numpy.array([float(nb_val_f1(a,None)) for a in grid],dtype=numpy.float64)
        idx=int(scores.argmax())
        if scores[idx]>bestFScore:bestFScore=float(scores[idx])
        bestTempAlpha=float(grid[idx])
        if tempbestScore is not None and abs(bestTempAlpha-tempbestScore)<=tol:break
        tempbestScore=bestTempAlpha
        li=max(0,idx-1)
        ri=min(numOfPoints-1,idx+1)
        nlow=float(grid[li])
        nhigh=float(grid[ri])
        if nhigh-nlow<min_width:break
        if nlow==nhigh:
            step=(alphaHigh-alphaMin)/(numOfPoints-1)
            nlow=max(1e-4,bestTempAlpha-step)
            nhigh=min(1.0,bestTempAlpha+step)
        alphaMin,alphaHigh=nlow,nhigh
    return bestTempAlpha,bestFScore

def zoom_best_s(sMin=0.0,sMax=1.0,numOfPoints=15,maxItterations=4):
    bestS=0.0
    bestFScore=-1.0
    for _ in range(maxItterations):
        grid=numpy.linspace(sMin,sMax,numOfPoints)
        scores=[]
        for s in grid:
            pri=(1-s)*trainPrior+s*uniformPrior
            scores.append(float(nb_val_f1(bestAlpha,pri)))
        scores=numpy.array(scores,dtype=numpy.float64)
        idx=int(scores.argmax())
        if scores[idx]>bestFScore:bestFScore=float(scores[idx])
        bestS=float(grid[idx])
        li=max(0,idx-1)
        ri=min(numOfPoints-1,idx+1)
        nlow=float(grid[li])
        nhigh=float(grid[ri])
        if nlow==nhigh:
            step=(sMax-sMin)/(numOfPoints-1)
            nlow=max(0.0,bestS-step)
            nhigh=min(1.0,bestS+step)
        sMin,sMax=nlow,nhigh
    return bestS,bestFScore

filePath=input("Enter path to DataSet.txt: ")
if os.path.exists(filePath):
    dataFrame=pandas.read_csv(filePath,sep="\t",names=["text","sentiment"],header=None,engine="python",quoting=csv.QUOTE_NONE)
    dataFrame["sentiment"]=dataFrame["sentiment"].replace({"OBJ":"NEUTRAL"})
    print("Total Records: "+str(len(dataFrame)))
    print("\nClass Distribution:\n",dataFrame["sentiment"].value_counts())
    plt.figure(figsize=(7,5))
    sns.countplot(data=dataFrame,x="sentiment",hue="sentiment",palette="viridis",legend=False)
    plt.title("Figure 1: Class Distribution")
    plt.savefig("1_distribution.png")
    plt.show()
    dataFrame["tfidf_text"]=dataFrame["text"].apply(preprocessingTextForTF_IDF)
    dataFrame["tr_text"]=dataFrame["text"].apply(preprocessForTransformer)
    dataFrame[["text","tfidf_text","tr_text","sentiment"]].to_excel("Preprocessing_Comparison.xlsx",index=False)
    TFIDFText=dataFrame["tfidf_text"].astype(str).to_numpy()
    transformerText=dataFrame["tr_text"].astype(str).to_numpy()
    labelOfText=dataFrame["sentiment"].astype(str).to_numpy()
    TFIDFTrainText,TFIDFTempText,transformerTrainText,transformerTempText,labelTrainText,labelTempText=train_test_split(
        TFIDFText,transformerText,labelOfText,test_size=0.4,stratify=labelOfText,random_state=42
    )
    TFIDFValidationText,TFIDFTestText,transformerValidationText,transformerTestText,labelValidationText,labelTestText=train_test_split(
        TFIDFTempText,transformerTempText,labelTempText,test_size=0.5,stratify=labelTempText,random_state=42
    )
    print("\nSplit sizes:")
    print("Train:"+str(len(labelTrainText))+"| Validation:"+str(len(labelValidationText))+"| Test:"+str(len(labelTestText)))
    TFIDFChar=TfidfVectorizer(analyzer="char",ngram_range=(3,5),min_df=2,max_df=0.9)
    TFIDFWord=TfidfVectorizer(analyzer="word",ngram_range=(1,3),min_df=2,max_df=0.9)
    TFIDFTrainChar=TFIDFChar.fit_transform(TFIDFTrainText)
    TFIDFValidationChar=TFIDFChar.transform(TFIDFValidationText)
    TFIDFTestChar=TFIDFChar.transform(TFIDFTestText)
    TFIDFTrainWord=TFIDFWord.fit_transform(TFIDFTrainText)
    TFIDFValidationWord=TFIDFWord.transform(TFIDFValidationText)
    TFIDFTestWord=TFIDFWord.transform(TFIDFTestText)
    mixTrain=sp.csr_matrix(hstack([TFIDFTrainChar,TFIDFTrainWord]))
    mixValidation=sp.csr_matrix(hstack([TFIDFValidationChar,TFIDFValidationWord]))
    mixTest=sp.csr_matrix(hstack([TFIDFTestChar,TFIDFTestWord]))
    rows=[]
    classes=numpy.array(["NEG","NEUTRAL","POS"])
    # NAIVE BAYES 
    naiveBayesTimer=time.perf_counter()
    print("\n"+"="*80)
    print("NAIVE BAYES HYPERPARAMETER TUNING")
    print("="*80)
    bestAlpha,_=zoom_best_alpha(alphaMin=1e-4,alphaHigh=1.0,numOfPoints=15)
    counts=numpy.array([(labelTrainText==c).sum() for c in classes],dtype=numpy.float64)
    trainPrior=counts/counts.sum()
    uniformPrior=numpy.ones(len(classes),dtype=numpy.float64)/len(classes)
    bestS,_=zoom_best_s(sMin=0.0,sMax=1.0,numOfPoints=15)
    bestPriorName=f"mix_s={numpy.round(bestS,6)}"
    bestPriors=(1-bestS)*trainPrior+bestS*uniformPrior
    print("BEST_ALPHA =",bestAlpha)
    print("BEST_PRIOR_NAME =",bestPriorName)
    print("BEST_PRIORS =",bestPriors)

    mixTrainAndValidation=sp.csr_matrix(sp.vstack([mixTrain,mixValidation]))
    labelsTrainAndValidation=numpy.concatenate([labelTrainText,labelValidationText])
    
    nb=MultinomialNB(alpha=bestAlpha,fit_prior=False,class_prior=bestPriors)
    _,m_nb=evaluateModel(
        f"Naive Bayes MultinomialNB (TF-IDF)",
        nb,mixTrainAndValidation,mixTest,labelsTrainAndValidation,labelTestText,"CM_NB_TFIDF.png"
    )
    naiveBayesTimerTotalTime=time.perf_counter()-naiveBayesTimer
    print(f"\nNAIVE BAYES TOTAL TIME: {int(naiveBayesTimerTotalTime//60)}m {int(naiveBayesTimerTotalTime%60)}s ({naiveBayesTimerTotalTime:.2f}s)")
    
    rows.append({"model":"Naive Bayes","accuracy":m_nb["accuracy"],"precision":m_nb["precision"],"recall":m_nb["recall"],"f1_score":m_nb["f1_score"],"time_seconds":naiveBayesTimerTotalTime})
    # MLP
    MLPTimer=time.perf_counter()
    print("\n"+"="*80)
    print("MLP NEURAL NETWORK HYPERPARAMETER TUNING")
    print("="*80)
    TRANSFORMER_MODEL="UBC-NLP/MARBERT"
    MAX_LEN=96
    BATCH=32
    safe_name=TRANSFORMER_MODEL.replace("/","_")
    transformerTrainTextEmbeedings=get_transformer_embeddings(transformerTrainText,model_name=TRANSFORMER_MODEL,max_length=MAX_LEN,batch_size=BATCH,cache_path=f"cache_{safe_name}_train_{MAX_LEN}.npy" )
    transformerValidationTextEmbeedings=get_transformer_embeddings(transformerValidationText,model_name=TRANSFORMER_MODEL,max_length=MAX_LEN,batch_size=BATCH,cache_path=f"cache_{safe_name}_val_{MAX_LEN}.npy")
    transformerTestTextEmbeedings=get_transformer_embeddings(transformerTestText,model_name=TRANSFORMER_MODEL,max_length=MAX_LEN,batch_size=BATCH,cache_path=f"cache_{safe_name}_test_{MAX_LEN}.npy")
    kSVD=250
    svd=TruncatedSVD(n_components=kSVD,algorithm="randomized",n_iter=7,random_state=42)
    SVDTrain=svd.fit_transform(mixTrain).astype(numpy.float32)
    SVDValidation=svd.transform(mixValidation).astype(numpy.float32)
    SVDTest=svd.transform(mixTest).astype(numpy.float32)
    scaledTFIDF=StandardScaler()
    scaledTFIDFTrain=scaledTFIDF.fit_transform(SVDTrain).astype(numpy.float32)
    scaledTFIDFValidation=scaledTFIDF.transform(SVDValidation).astype(numpy.float32)
    scaledTFIDFTest=scaledTFIDF.transform(SVDTest).astype(numpy.float32)
    scaledTransformer=StandardScaler()
    scaledTransformerTrain=scaledTransformer.fit_transform(transformerTrainTextEmbeedings).astype(numpy.float32)
    scaledTransformerValidation=scaledTransformer.transform(transformerValidationTextEmbeedings).astype(numpy.float32)
    scaledTransformerTest=scaledTransformer.transform(transformerTestTextEmbeedings).astype(numpy.float32)
    mixTrainScaled=numpy.hstack([scaledTFIDFTrain,scaledTransformerTrain]).astype(numpy.float32)
    mixValidationScaled=numpy.hstack([scaledTFIDFValidation,scaledTransformerValidation]).astype(numpy.float32)
    mixTestScaled=numpy.hstack([scaledTFIDFTest,scaledTransformerTest]).astype(numpy.float32)
    labelEncoderMLP = LabelEncoder()
    labelTrainEncoder = labelEncoderMLP.fit_transform(labelTrainText)
    labelValidationEncoder = labelEncoderMLP.transform(labelValidationText)
    MLPTrainAndValidation = numpy.vstack([mixTrainScaled, mixValidationScaled])
    MLPLabel = numpy.concatenate([labelTrainEncoder, labelValidationEncoder])
    test_fold = numpy.concatenate([numpy.full(len(labelTrainText), -1),numpy.full(len(labelValidationText), 0)])
    predefinedSplit = PredefinedSplit(test_fold)
    mlp_param_grid = {
        "hidden_layer_sizes": [(512,), (256,), (128,), (512, 256), (256, 128), (128, 64), (512, 256, 128), (256, 128, 64)],
        "alpha": [0.00001, 0.0001, 0.001, 0.01],
        "learning_rate_init": [0.0001, 0.001, 0.01],
        "activation": ['relu', 'tanh'],
        "solver": ['adam'],
        "batch_size": [32, 64, 128],
        "max_iter": [1000],
        "early_stopping": [True],
        "validation_fraction": [0.1],
        "n_iter_no_change": [20]
    }
    mlp_search = RandomizedSearchCV(
        MLPClassifier(random_state=42),
        mlp_param_grid,
        n_iter=20,
        scoring="f1_macro",
        cv=predefinedSplit,
        random_state=42,
        n_jobs=-1,
        verbose=1
    )
    print("\nSearching for best MLP hyperparameters...")
    mlp_search.fit(MLPTrainAndValidation, MLPLabel)
    best_mlp_params = mlp_search.best_params_
    print("Best MLP params:", best_mlp_params)
    print(f"Best CV F1 score: {mlp_search.best_score_:.4f}")
    X_trainval_fused = numpy.vstack([mixTrainScaled, mixValidationScaled]).astype(numpy.float32)
    y_trainval_enc = labelEncoderMLP.transform(labelTrainText.tolist() + labelValidationText.tolist())
    best_mlp: MLPClassifier = mlp_search.best_estimator_ # type: ignore
    best_mlp.fit(X_trainval_fused, y_trainval_enc)
    y_pred_mlp_int = best_mlp.predict(mixTestScaled)
    y_pred_mlp = labelEncoderMLP.inverse_transform(y_pred_mlp_int)
    acc = accuracy_score(labelTestText, y_pred_mlp)
    prec = precision_score(labelTestText, y_pred_mlp, average="macro", zero_division=0)
    rec = recall_score(labelTestText, y_pred_mlp, average="macro", zero_division=0)
    f1 = f1_score(labelTestText, y_pred_mlp, average="macro", zero_division=0)
    print("\n" + "="*60)
    print(f"Neural Network MLP")
    print("="*60)
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-Score : {f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(labelTestText, y_pred_mlp, digits=4, zero_division=0))
    labels = sorted(set(labelTestText.tolist()))
    cm = confusion_matrix(labelTestText, y_pred_mlp, labels=labels)
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=labels, yticklabels=labels)
    plt.title("Confusion Matrix: Neural Network MLP")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig("CM_MLP_FUSED.png")
    plt.show()
    print("Saved: CM_MLP_FUSED.png")
    mlp_total_time=time.perf_counter()-MLPTimer
    print(f"\nMLP TOTAL TIME: {int(mlp_total_time//60)}m {int(mlp_total_time%60)}s ({mlp_total_time:.2f}s)")
    rows.append({"model":"Neural Network","accuracy":acc,"precision":prec,"recall":rec,"f1_score":f1,"time_seconds":mlp_total_time})
    # ==================== RANDOM FOREST TUNING ====================
    randomForestStartTime = time.perf_counter()
    print("\n" + "="*80)
    print("BALANCED RANDOM FOREST HYPERPARAMETER TUNING")
    print("="*80)
    kGrid = [100, 150, 200, 250, 300, 350, 400]
    bestK = None
    bestF1 = -1.0
    bestPack = None
    maxK = max(kGrid)
    SVDAll = TruncatedSVD(n_components=maxK, algorithm="randomized", n_iter=7, random_state=42)
    trainSVDAll = SVDAll.fit_transform(mixTrainScaled).astype(numpy.float32)
    validationSVDAll = SVDAll.transform(mixValidationScaled).astype(numpy.float32)
    testSVDAll = SVDAll.transform(mixTestScaled).astype(numpy.float32)
    scaledTFIDFAll = StandardScaler()
    trainSVDAll = scaledTFIDFAll.fit_transform(trainSVDAll).astype(numpy.float32)
    validationSVDAll = scaledTFIDFAll.transform(validationSVDAll).astype(numpy.float32)
    testSVDAll = scaledTFIDFAll.transform(testSVDAll).astype(numpy.float32)
    trainMixRF = numpy.hstack([trainSVDAll[:, :maxK], scaledTransformerTrain]).astype(numpy.float32)
    validationMixRF = numpy.hstack([validationSVDAll[:, :maxK], scaledTransformerValidation]).astype(numpy.float32)
    textSearchRF = numpy.vstack([trainMixRF, validationMixRF])
    labelSearchRF = numpy.concatenate([labelTrainText, labelValidationText])
    test_fold_rf = numpy.concatenate([
        numpy.full(len(labelTrainText), -1),
        numpy.full(len(labelValidationText), 0)
    ])
    ps_rf = PredefinedSplit(test_fold_rf)
    RFParameter = {
        "n_estimators": [300, 500, 700, 1000, 1500],
        "max_depth": [20, 30, 40, 50, None],
        "min_samples_split": [2, 5, 10, 15],
        "max_samples": [0.7, 0.8, 0.9, 1.0],
        "bootstrap": [True],
        "min_samples_leaf": [1, 2, 4, 8],
        "max_features": ["sqrt", "log2"],
        "replacement": [True, False],
        "sampling_strategy": ['auto', 'majority', 'all']
    }
    print(f"\n--- Starting Hyperparameter Tuning using k={maxK} (BalancedRandomForestClassifier) ---")
    rfBase = BalancedRandomForestClassifier(random_state=42, n_jobs=-1)
    rfSearch = RandomizedSearchCV(
        rfBase,
        param_distributions=RFParameter,
        n_iter=15,
        scoring="f1_macro",
        cv=ps_rf,
        random_state=42,
        n_jobs=-1,
        verbose=1
    )
    rfSearch.fit(textSearchRF, labelSearchRF)
    bestParameters = rfSearch.best_params_
    print("Best RF params found:", bestParameters)
    print(f"Best CV F1 score: {rfSearch.best_score_:.4f}")
    print("\n--- Evaluating k_grid with fixed best parameters ---")
    for k in kGrid:
        k = int(k)
        X_train_k = numpy.hstack([trainSVDAll[:, :k], scaledTransformerTrain]).astype(numpy.float32)
        X_val_k = numpy.hstack([validationSVDAll[:, :k], scaledTransformerValidation]).astype(numpy.float32)
        X_test_k = numpy.hstack([testSVDAll[:, :k], scaledTransformerTest]).astype(numpy.float32)
        model = BalancedRandomForestClassifier(**bestParameters, random_state=42, n_jobs=-1)
        model.fit(X_train_k, labelTrainText)
        pred_val = model.predict(X_val_k)
        val_f1 = float(f1_score(labelValidationText, pred_val, average="macro", zero_division=0))
        if val_f1 > bestF1:
            bestF1 = val_f1
            bestK = k
            bestPack = (X_train_k, X_val_k, X_test_k)
        print(f"k={k} | Val F1={val_f1:.4f} | Best so far: k={bestK} (F1={bestF1:.4f})")
    if bestPack is None:
        raise ValueError("No best_k found")
    X_train_best, X_val_best, X_test_best = bestPack
    X_trainval_best = numpy.vstack([X_train_best, X_val_best]).astype(numpy.float32)
    y_trainval = numpy.concatenate([labelTrainText, labelValidationText])
    rf_final = BalancedRandomForestClassifier(**bestParameters, random_state=42, n_jobs=-1)
    _, m_rf = evaluateModel(
        f"Balanced Random Forest",
        rf_final,
        X_trainval_best,
        X_test_best,
        y_trainval,
        labelTestText,
        "CM_BRF_FUSED.png"
    )
    rf_total_time = time.perf_counter() - randomForestStartTime
    print(f"\nRANDOM FOREST TOTAL TIME: {int(rf_total_time//60)}m {int(rf_total_time%60)}s ({rf_total_time:.2f}s)")
    rows.append({
        "model": "Balanced Random Forest",
        "accuracy": m_rf["accuracy"],
        "precision": m_rf["precision"],
        "recall": m_rf["recall"],
        "f1_score": m_rf["f1_score"],
        "time_seconds": rf_total_time
    })
    printLeaderboard(rows)
    plotPerformanceGraphs(rows)
else:
    print("File not found")