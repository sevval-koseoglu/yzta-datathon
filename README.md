# YZTA 5.0 Datathon - Bilişsel Performans Skoru Tahmini

Bu repo, YZTA 5.0 Datathon kapsamında `bilissel_performans_skoru` hedef değişkenini tahmin etmek için hazırlanmış modelleme çalışmasını içerir.

## Problem

Verilen uyku, sağlık, yaşam tarzı ve demografik değişkenlerden bilişsel performans skoru tahmin edilmektedir.

- Problem tipi: Regresyon
- Metrik: RMSE
- Model: CatBoost Regressor
- Public leaderboard skoru: 1.20556

## Dosya yapısı

```text
notebooks/
  YZTA_Datathon_CatBoost_Clean.ipynb
requirements.txt
.gitignore
README.md
```

## Gerekli veri dosyaları

Veri dosyaları repoya eklenmemiştir. Notebook'u çalıştırmadan önce aşağıdaki dosyaları lokal çalışma dizinine veya Kaggle input alanına ekleyin:

```text
train.csv
test_x.csv
sample_submission.csv
```

## Çalıştırma

1. Gerekli kütüphaneleri kurun.
2. Notebook'u açın.
3. Hücreleri sırayla çalıştırın.
4. Oluşan `catboost_submission.csv` dosyasını Kaggle'a yükleyin.
