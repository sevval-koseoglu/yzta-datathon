import numpy as np
def add_features(df):
    df = df.copy()
    eps = 1e-6

    df["ulke_norm"] = df["ulke"].replace({
        "South Korea": "Guney Kore",
        "Spain": "Ispanya",
        "Sweden": "Isvec",
        "Mexico": "Meksika",
        "Netherlands": "Hollanda",
    })
    df["meslek_norm"] = df["meslek"].replace({"Lawyer": "Avukat"})

    df["rem_derin_toplam"] = df["rem_yuzdesi"] + df["derin_uyku_yuzdesi"]
    df["hafif_uyku_yuzdesi"] = 100 - df["rem_yuzdesi"] - df["derin_uyku_yuzdesi"]
    df["rem_derin_oran"] = df["rem_yuzdesi"] / (df["derin_uyku_yuzdesi"] + eps)
    df["derin_rem_oran"] = df["derin_uyku_yuzdesi"] / (df["rem_yuzdesi"] + eps)
    df["uyku_kalitesi_raw"] = 0.45 * df["rem_yuzdesi"] + 0.55 * df["derin_uyku_yuzdesi"]
    df["uyku_bolunme_skoru"] = df["uykuya_dalma_suresi_dk"] + 10 * df["gecelik_uyanma_sayisi"]
    df["iyi_uyku_indeksi"] = (
        df["rem_yuzdesi"]
        + df["derin_uyku_yuzdesi"]
        - 0.25 * df["uykuya_dalma_suresi_dk"]
        - 2 * df["gecelik_uyanma_sayisi"]
    )
    df["uyku_verimlilik_proxy"] = (
        df["rem_derin_toplam"]
        - 0.10 * df["hafif_uyku_yuzdesi"]
        - 2 * df["gecelik_uyanma_sayisi"]
    )

    df["kafein_ekran_toplam"] = df["uyku_oncesi_kafein_mg"] / 50 + df["uyku_oncesi_ekran_suresi_dk"] / 60
    df["kafein_ekran_carpim"] = df["uyku_oncesi_kafein_mg"] * df["uyku_oncesi_ekran_suresi_dk"]
    df["aktivite_calisma_oran"] = df["gunluk_adim_sayisi"] / (df["gunluk_calisma_saati"] + 1)
    df["adim_bin_proxy"] = df["gunluk_adim_sayisi"] / 1000
    df["sekerleme_saat"] = df["sekerleme_suresi_dk"] / 60

    df["stres_calisma"] = df["stres_skoru"] * df["gunluk_calisma_saati"]
    df["stres_uyku_bolunme"] = df["stres_skoru"] * df["gecelik_uyanma_sayisi"]
    df["stres_kafein"] = df["stres_skoru"] * df["uyku_oncesi_kafein_mg"]
    df["stres_ekran"] = df["stres_skoru"] * df["uyku_oncesi_ekran_suresi_dk"]
    df["nabiz_stres"] = df["dinlenik_nabiz_bpm"] * df["stres_skoru"]
    df["saglik_risk_indeksi"] = df["stres_skoru"] + df["vucut_kitle_indeksi"] / 10 + df["dinlenik_nabiz_bpm"] / 20
    df["bmi_nabiz"] = df["vucut_kitle_indeksi"] * df["dinlenik_nabiz_bpm"]
    df["yas_stres"] = df["yas"] * df["stres_skoru"]
    df["yas_nabiz"] = df["yas"] * df["dinlenik_nabiz_bpm"]

    df["hafta_sonu_mu"] = (df["gun_tipi"] == "Hafta sonu").astype(int)
    df["sicaklik_idealden_sapma"] = (df["oda_sicakligi_celsius"] - 20).abs()
    df["sicaklik_idealden_sapma2"] = df["sicaklik_idealden_sapma"] ** 2
    df["hafta_sonu_farki_abs"] = df["hafta_sonu_uyku_farki_saat"].abs()
    df["hafta_sonu_farki2"] = df["hafta_sonu_uyku_farki_saat"] ** 2

    for col in [
        "stres_skoru",
        "uyku_oncesi_ekran_suresi_dk",
        "uyku_oncesi_kafein_mg",
        "gunluk_calisma_saati",
        "gecelik_uyanma_sayisi",
        "uykuya_dalma_suresi_dk",
    ]:
        df[f"{col}_sq"] = df[col] ** 2
        df[f"{col}_log1p"] = np.log1p(df[col])

    df["kronotip_gun_tipi"] = df["kronotip"].astype(str) + "_" + df["gun_tipi"].astype(str)
    df["ruh_kronotip"] = df["ruh_sagligi_durumu"].astype(str) + "_" + df["kronotip"].astype(str)
    df["meslek_gun_tipi"] = df["meslek_norm"].astype(str) + "_" + df["gun_tipi"].astype(str)
    df["ulke_mevsim"] = df["ulke_norm"].astype(str) + "_" + df["mevsim"].astype(str)
    df["cinsiyet_kronotip"] = df["cinsiyet"].astype(str) + "_" + df["kronotip"].astype(str)

    for col in ["meslek", "vucut_kitle_indeksi", "uyku_oncesi_kafein_mg", "stres_skoru", "kronotip", "ruh_sagligi_durumu"]:
        df[f"{col}_missing"] = df[col].isna().astype(int)

    return df
