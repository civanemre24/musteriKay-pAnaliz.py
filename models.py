import pandas as pd
import matplotlib as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split



#veri setini yükleyelim


dosya_yolu = r'C:\Users\civan\Desktop\pratic\musteriKayıpAnaliz.py\Costmer-Data.csv'
df=pd.read_csv(dosya_yolu)
print(df.shape,df.columns,df.info(5))



#veri setinini temizleme

print(df.isnull().sum())
print(df.dtypes)

# Boşluk karakterlerini (gizli kayıp veriyi) NaN (Not a Number) ile değiştir
df['TotalCharges'] = df['TotalCharges'].replace(' ', np.nan)

df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
print(df.dtypes)

#1.Total-Charges sütununun medyan degerini hesaplıyalım
total_charges_median = df['TotalCharges'].median()
print(f"TotalCharges MEDYAN DEGERİ :{total_charges_median}")

#2.kayıp(NaN) verileri hesaplalnan medyan ile dolduralım.
#inplace=True ile degişklilkleri dogrudan df üzerine aktarıyoruz.
df['TotalCharges'] = df['TotalCharges'].fillna(total_charges_median)

#kayıp verileri kontrol edelim
print("\nGüncel kayıp veri sayısı :")
print(df.isnull().sum())

#degişkenlerden biri olan churn dagılınama bakarak analize devam edicez.
print("Churn Dagılımı:")
print(df['Churn'].value_counts())
print("\nOranlar:")
print(df['Churn'].value_counts(normalize=True))


#şimdi ise sütünları kategorik ve numerik olarak ayıralım yani sayı yazanlar bir tarafa yazı yazanlar bir tarafa
#customerID  benzersiz oldugu için yani kişye özel modellemde kulanmıycaz.
#şimdi sadece kategorik(yani yazı olanları) tipindeki sütınları listeleyelim.
kategorik_sutunlar = df.select_dtypes(include=['object']).columns.tolist()

print("Kategorik Sütunlar:")
print(kategorik_sutunlar)

#tüm sütünalar için (unique degerleri) görelimki hagisne hangi kodlama tipini uygulaycagmızı karar verelim.

print("\nTüm kategorik sütunların eşsiz Degerleri:")
for col in kategorik_sutunlar:
    print(f"____{col}")
    print(df[col].unique())


#şimdi customerID yi silicez çünkü her kulanıcıda benzersiz oldugu için modele katkı saglamaz.
df.drop('customerID', axis = 1, inplace =True)

#şimdi iki degerli sütunları kodyalım(yes/no,famele/male gibi...)
#tüm ikli(binary) degerli sütunları listeleyelim
binary_cols = ['gender','Partner' ,'Dependents','PhoneService','PaperlessBilling','Churn']

#Churn için daha önce ypmıştık o yüzden şimdi diger 4 sütun içim yes/no'yu 1/0 yapalım.
df['gender'] = df['gender'].replace({'Female':0 ,'Male':1})

#diger yes/no sütunları içinde yanısını yapalım
for col in ['Partner','Dependents','PhoneService','PaperlessBilling']:
    df[col] = df[col].replace({'Yes' :1, 'No' :0})

#son kalan çok degerli sütunar içinde aynı işlemi yapalım
ohe_cols = ['MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 
            'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 
            'Contract', 'PaymentMethod']

df=pd.get_dummies(df, columns=ohe_cols, drop_first=True)

print("\nOne-Hot Encoding Yapıldı. Yeni Sütun Sayısı:", df.shape[1])
print("Yeni Sütun Başlıklarının İlk 10 tanesi:", df.columns[:10].tolist())



df['Churn'] = df['Churn'].replace({'Yes': 1, 'No': 0}) 

# 1. X ve y'yi ayırma (Şimdi y sayısal olacak)
X = df.drop('Churn', axis=1)
y = df['Churn'] 

# 2. Eğitim ve test seti olark ayıralım
X_train, X_test ,y_train,y_test = train_test_split(
    X ,y, test_size=0.3, random_state=42, stratify=y
)



#şimdi model egitmeye geçebilirz.
#veri setini egtim ve test olrak ayıralım.

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
#1.X ve y'yi ayırma
X = df.drop('Churn',axis=1)  #Churn dışındaki tüm sütunalr X
y =df['Churn']               #Churn sütunu y (hedef)

#2.Egtim ve test seti olark ayıralım 70 egtim 30 test olarak yapalım
# test_size=0.3 : %30'u test seti için ayırır.
# random_state=42 : Sonuçların tekrar çalıştırıldığında aynı kalmasını sağlar.

X_train, X_test ,y_train,y_test = train_test_split(
    X ,y, test_size=0.3, random_state=42,stratify=y
    )
print("fToplam veri sayısı:df.shape[0]}")
print("---")

print(f"Eğitim seti boyutu (satır):%70 = {X_train.shape[0]}")
print(f"test seti boyutu (satır):%30 ={X_test.shape[0]}")
print(f"Toplam Sütun Sayısı: {X_train.shape[1]}") # Sütun sayısı hala aynı
print("---")


#sayısal sütunları scaler ile ölçeklendirelim
Scaler = StandardScaler()

#scaleri sadece egtim verisi (X_train üzerinde) egitiyoruz

X_train_scaled = Scaler.fit_transform(X_train)

#egitmiş oldugumuz scalerı test setine uygulayalım.

X_test_scaled = Scaler.transform(X_test)

print("Ölçeklendirme tamamlandı.")





from sklearn.linear_model import LogisticRegression,LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

#1.Modeli tanımlama ve egitme
log_model = LogisticRegression(random_state=42)

#Modeli, ölçeklendirilmiş egtim verileriyle egitiyoruz: Bu, öodelin "ögrenme" anıdır.
log_model.fit(X_train_scaled, y_train)

print("Lojistik Regresyon Modeli Başarıyla Egitildi.")

#2.test seti üzerinde tahmin yapalım.
y_pred= log_model.predict(X_test_scaled)

#3.modelin performansını degerlendirelim.
accuracy= accuracy_score(y_test, y_pred)
print(f"\nModel Dogruluk (Accuracy) Puanı: {accuracy:.4f}")

print("\nKarışıklık Matrisi (Confusion Matrix):")
print(confusion_matrix(y_test, y_pred))

print("\nSınıflandırma Raporu:")
print(classification_report(y_test, y_pred))

from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Modeli tanımlama
log_model_smote = LogisticRegression(random_state=42)

# Modeli SMOTE uygulanmış veriyle eğitme
# y_train_smote ve X_train_smote kullanılıyor.
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train)

log_model_smote.fit(X_train_smote, y_train_smote)

# Test seti üzerinde tahmin yapma (X_test_scaled değişmedi)
y_pred_smote = log_model_smote.predict(X_test_scaled)

# Performansı değerlendirme
print("\n--- SMOTE Uygulanmış Lojistik Regresyon Performansı ---")
print(f"Model Doğruluk (Accuracy) Puanı: {accuracy_score(y_test, y_pred_smote):.4f}")
print("\nKarışıklık Matrisi:")
print(confusion_matrix(y_test, y_pred_smote))
print("\nSınıflandırma Raporu:")
print(classification_report(y_test, y_pred_smote))


#smote ile yapay veri setleri oluşturuyoruz Recal degerini arttırmak için.

smote = SMOTE(random_state=42)

X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train)
print(f"Orijinal Eğitim Verisi Boyutu: {X_train_scaled.shape}")
print(f"SMOTE Sonrası Eğitim Verisi Boyutu: {X_train_smote.shape}")
print("\nSMOTE Sonrası Churn Dağılımı:")
print(y_train_smote.value_counts())

#xgboost kullanarak modeli daha da iyleştirecez xgboosttu smote ile oluşturdugmuz veri üzerinde uygulayacagız.
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

xgb_model = XGBClassifier(
    random_state = 42,
    use_label_encoder = False,
    eval_metric = 'logloss',
    n_estimators=100
)
# Modeli SMOTE uygulanmış veriyle eğitme
xgb_model.fit(X_train_smote, y_train_smote)
y_pred_xgb = xgb_model.predict(X_test_scaled)

print("\n--- SMOTE Uygulanmış XGBoost Performansı ---")
print(f"Model Doğruluk (Accuracy) Puanı: {accuracy_score(y_test, y_pred_xgb):.4f}")
print("\nKarışıklık Matrisi:")
print(confusion_matrix(y_test, y_pred_xgb))
print("\nSınıflandırma Raporu:")
print(classification_report(y_test, y_pred_xgb))

#şimdi Lojistik regrasyon eşigini ayarlayarak modelin performanısını iyileştircez.
y_proba = log_model_smote.predict_proba(X_test_scaled)[:, 1]

from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
precision , recall, thresholds = precision_recall_curve(y_test, y_proba)

#Grafik çizimi
plt.figure(figsize=(8, 6))
plt.plot(thresholds, precision[:-1], label='Precision')
plt.plot(thresholds, recall[:-1], label='Recall')
plt.title('Eşiklere göre Recall ve Precision Değişimi')
plt.xlabel('Eşik Değerleri')
plt.ylabel('Değerler')
plt.legend()
plt.grid(True)
plt.show()

ideal_threshold_index = np.where(recall >= 0.70)[0][-1]
ideal_threshold = thresholds[ideal_threshold_index]

print(f"Recall >= 0.70 iken en uygun eşik (threshold): {ideal_threshold:.4f}")
print(f"Bu eşikteki Precision: {precision[ideal_threshold_index]:.4f}")
print(f"Bu eşikteki Recall: {recall[ideal_threshold_index]:.4f}")

#Şimdi yeni eşik degeri ile tahmin yapıcaz.
y_pred_tuned = (y_proba >= ideal_threshold).astype(int)

print("\n--- Eşik Ayarı Sonrası Lojistik Regresyon Performansı ---")
print(f"Yeni Eşik: {ideal_threshold:.4f}")
print(f"Model Doğruluk (Accuracy) Puanı: {accuracy_score(y_test, y_pred_tuned):.4f}")
print("\nKarışıklık Matrisi:")
print(confusion_matrix(y_test, y_pred_tuned))
print("\nSınıflandırma Raporu:")
print(classification_report(y_test, y_pred_tuned))

import joblib
Scaler = StandardScaler()
joblib.dump(log_model_smote, 'en_iyi_churn_modeli.pkl')
joblib.dump(Scaler, 'scaler_objesi.pkl')