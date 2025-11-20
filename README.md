# 📈 Müşteri Kaybı (Churn) Tahmin Modeli: Telco Müşteri Analizi

Bu proje, bir telekomünikasyon şirketinin müşteri kaybını tahmin etmek ve şirket için en maliyetli olan ayrılma olaylarını en yüksek oranda yakalamak amacıyla geliştirilmiş bir Makine Öğrenimi çözümüdür.

## 🎯 TEMEL HEDEF VE İŞ SONUCU

Amacımız, genel doğruluktan (Accuracy) ziyade, **ayrılma riski yüksek müşterileri yakalama oranı (Recall)** üzerine odaklanmaktır. Şirketin hedeflediği minimum yakalama oranı %70 olarak belirlenmiştir.

- **Seçilen Model:** SMOTE + Eşik Ayarlı Lojistik Regresyon
- **Recall (Müşteriyi Yakalama):** **%70**
- **Precision (Kesinlik):** **%56**

Bu optimize edilmiş model, şirketin hedeflediği %70 yakalama oranına ulaşırken, gereksiz kampanya maliyetini azaltarak iş hedeflerine en uygun çözümü sunmaktadır.

---

## 🛠️ METODOLOJİ VE KULLANILAN TEKNİKLER

1.  **Veri Hazırlığı:** Gizli kayıp veriler median ile dolduruldu; kategorik veriler **One-Hot Encoding** ile sayısal hale getirildi.
2.  **Dengesizlik Giderme:** Eğitim verisine **SMOTE** uygulanarak Churn sınıfları dengelendi.
3.  **Model Optimizasyonu:** Lojistik Regresyon modeli, iş hedefine uygun olarak $\mathbf{0.6187}$ eşiği ile ayarlandı.

---

## 🔬 MODEL PERFORMANS ANALİZİ

Final modelin performansı, **Churn (Sınıf 1)** sınıfının doğru tahmin edilmesine odaklanmıştır:

* **Final Recall:** $\mathbf{0.70}$ (Gerçekten ayrılanların %70'i doğru yakalanmıştır.)
* **Final Precision:** $\mathbf{0.56}$ (Yanlış alarm maliyeti düşürülmüştür.)
* **Eşik Ayarının Rolü:** Tahmin eşiğinin $\mathbf{0.5}$'ten $\mathbf{0.6187}$'ye yükseltilmesi, $\text{Recall}$'ın %70'te tutulmasını sağlarken, $\text{Precision}$'ın $0.51$'den $\mathbf{0.56}$'ya çıkmasını sağlamıştır.

| Metrik | Sınıf 0 (Ayrılmayan) | Sınıf 1 (Ayrılan / Churn) | Genel |
| :--- | :--- | :--- | :--- |
| **Precision** | $0.88$ | $0.56$ | $0.80$ |
| **Recall** | $0.80$ | $0.70$ | $0.77$ |
| **Accuracy** | - | - | **$0.7747$** |

---


