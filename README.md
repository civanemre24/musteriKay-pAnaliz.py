# 📈 Müşteri Kaybı (Churn) Tahmin Modeli: Telco Müşteri Analizi

Bu proje, bir telekomünikasyon şirketinin müşteri kaybını tahmin etmek ve şirket için en maliyetli olan ayrılma olaylarını en yüksek oranda yakalamak amacıyla geliştirilmiş bir Makine Öğrenimi çözümüdür.

## 🎯 Temel Hedef ve İş Sonucu

Amacımız, genel doğruluktan (Accuracy) ziyade, **ayrılma riski yüksek müşterileri yakalama oranı (Recall)** üzerine odaklanmaktır. Şirketin hedeflediği minimum yakalama oranı %70 olarak belirlenmiştir.

- **Seçilen Model:** SMOTE + Eşik Ayarlı Lojistik Regresyon
- **Recall (Müşteriyi Yakalama):** %70
- **Precision (Kesinlik):** %56

Bu optimize edilmiş model, şirketin hedeflediği %70 yakalama oranına ulaşırken, gereksiz kampanya maliyetini azaltarak iş hedeflerine en uygun çözümü sunmaktadır.

## 🛠️ Metodoloji ve Kullanılan Teknikler

1.  **Veri Hazırlığı:** Gizli kayıp veriler median ile dolduruldu; kategorik veriler **One-Hot Encoding** ile sayısal hale getirildi.
2.  **Dengesizlik Giderme:** Eğitim verisine **SMOTE** uygulanarak Churn sınıfları dengelendi.
3.  **Model Optimizasyonu:** Lojistik Regresyon modeli, iş hedefine uygun olarak $0.6187$ eşiği ile ayarlandı.

## 💾 Proje İçeriği

- `musteriAnaliz.py`: Tüm veri işleme, model eğitimi ve değerlendirme kodlarını içerir.
- `data.csv`: Orijinal veri seti.
- `models.py`: Eğitilmiş model ve ölçekleyicilerin bulunduğu alandır.
