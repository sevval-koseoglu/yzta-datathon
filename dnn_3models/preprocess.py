import pandas as pd
import matplotlib.pyplot as plt

from add_feature import add_features

def preprocess_data(document):
    # Load the dataset
    df = pd.read_csv(document)

    print(df.head())
    print(df.info())
    print(df.describe())

    # Check for missing values
    print(df.isnull().sum())
    #delete if null
    df = df.dropna()

    # Check for duplicates
    print(df.duplicated().sum())
    # Remove duplicates
    df = df.drop_duplicates()

    
    
    # Remove outliers using IQR f
    #bunu hangi columlar için yapacağımızı bilemedim, o yüzden yorum satırı yaptım

    # Q1 = df['column_name'].quantile(0.25)
    # Q3 = df['column_name'].quantile(0.75)
    # IQR = Q3 - Q1
    # lower_bound = Q1 - 1.5 * IQR
    # upper_bound = Q3 + 1.5 * IQR
    # df = df[(df['column_name'] >= lower_bound) & (df['column_name'] <= upper_bound)]
    #aşağıdaki gibi görselleştirme yaparak hangi columnlarda outlier olduğunu görebiliriz
      
    # plt.boxplot(df['column_name'])
    # plt.title('Boxplot of column_name')
    # plt.show()

    # Encode categorical variables
    df = pd.get_dummies(df, columns=["cinsiyet","meslek","ulke","kronotip","ruh_sagligi_durumu","mevsim","gun_tipi"])

    #check string columns
    print(df.select_dtypes(include=['object']).columns)

    #drop id column if exists
    df = df.drop('id', axis=1)

    # Convert string columns to numeric using label encoding
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = le.fit_transform(df[col])

    return df

def preprocess_and_add_data(document):
    df = pd.read_csv(document)

    print(df.head())
    print(df.info())
    print(df.describe())

    # Check for missing values
    print(df.isnull().sum())   


    # 1) toplam_kaliteli_uyku
    df["toplam_kaliteli_uyku"] = df["rem_yuzdesi"] + df["derin_uyku_yuzdesi"]

    # 2) kafein_ekran_etkisi
    df["kafein_ekran_etkisi"] = (
        df["uyku_oncesi_kafein_mg"] * df["uyku_oncesi_ekran_suresi_dk"]
    )

    # 3) kronotip × gun_tipi interaction
    df["kronotip_gun_tipi"] = (
        df["kronotip"].astype(str) + "_" + df["gun_tipi"].astype(str)
    )
    df = add_features(df)
    cat_cols = df.select_dtypes(include="object").columns.tolist()
    # Kategorik interaction'ı modele uygun hale getirmek için one-hot encode et
    df = pd.get_dummies(df, columns=cat_cols)
    print(df.select_dtypes(include=['object']).columns)
    #drop id column if exists
    df = df.drop('id', axis=1)
    print(df.head())
    print(df.columns)

    # Convert string columns to numeric using label encoding
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = le.fit_transform(df[col])

    return df
