from flask import Flask, render_template, request
from catboost import CatBoostClassifier, Pool
import numpy as np

app = Flask(__name__)

model = CatBoostClassifier()
model.load_model('model/catboost-x6-tuning.cbm')

cat_features = [
    'akreditasi_sekolah', 'jenis_kelamin', 'prov',
    'pres1_jenjang', 'pres2_jenjang', 'pres3_jenjang',
    'juara_pres1', 'juara_pres2', 'juara_pres3',
    'pil1', 'pil2'
]

bobot_jenjang = {
    'Kabupaten/Kota': 0.1,
    'Propinsi': 0.2,
    'Nasional': 0.3,
    'Internasional': 0.4
}

bobot_juara_mapping = {
    'Juara 1': 0.5,
    'Juara 2': 0.3,
    'Juara 3': 0.2,
    'Medali Emas': 0.5,
    'Medali Perak': 0.3,
    'Medali Perunggu': 0.2
}

subject_mapping = {
    'matematika': 'Matematika',
    'fisika': 'Fisika',
    'kimia': 'Kimia',
    'biologi': 'Biologi',
    'bhsindonesia': 'BhsIndonesia',
    'bhsinggris': 'BhsInggris'
}

mapel_pendukung = {
    "FISIKA": ["Fisika", "Matematika"],
    "MATEMATIKA": ["Matematika", "Fisika"],
    "STATISTIKA": ["Matematika", "Fisika"],
    "KIMIA": ["Kimia", "Fisika"],
    "BIOLOGI": ["Biologi", "Kimia"],
    "SAINS AKTUARIA": ["Matematika", "Ekonomi"],
    "SAINS ANALITIK DAN INSTRUMENTASI KIMIA": ["Kimia", "Fisika"],
    "TEKNIK MESIN": ["Fisika", "Matematika"],
    "TEKNIK KIMIA": ["Fisika", "Kimia"],
    "TEKNIK FISIKA": ["Fisika", "Matematika"],
    "TEKNIK INDUSTRI": ["Fisika", "Matematika"],
    "TEKNIK MATERIAL": ["Fisika", "Kimia"],
    "TEKNIK PANGAN": ["Fisika", "Kimia"],
    "TEKNIK SIPIL": ["Fisika", "Matematika"],
    "ARSITEKTUR": ["Matematika", "Fisika"],
    "TEKNIK LINGKUNGAN": ["Matematika", "Kimia"],
    "PERENCANAAN WILAYAH DAN KOTA": ["Matematika", "Fisika"],
    "TEKNIK GEOMATIKA": ["Matematika", "Fisika"],
    "TEKNIK GEOFISIKA": ["Matematika", "Fisika"],
    "TEKNIK PERKAPALAN": ["Matematika", "Fisika"],
    "TEKNIK SISTEM PERKAPALAN": ["Matematika", "Fisika"],
    "TEK. SIST PERKAPALAN (GLR GANDA ITS-JERMAN)": ["Matematika", "Fisika"],
    "TEKNIK KELAUTAN": ["Matematika", "Fisika"],
    "TEKNIK TRANSPORTASI LAUT": ["Matematika", "Fisika"],
    "TEKNIK LEPAS PANTAI": ["Matematika", "Fisika"],
    "TEKNIK ELEKTRO": ["Matematika", "Fisika"],
    "TEKNIK BIOMEDIK": ["Matematika", "Fisika"],
    "TEKNIK KOMPUTER": ["Matematika", "Fisika"],
    "TEKNIK INFORMATIKA": ["Matematika", "Fisika"],
    "SISTEM INFORMASI": ["Matematika", "Fisika"],
    "TEKNOLOGI INFORMASI": ["Matematika", "Fisika"],
    "TEKNIK TELEKOMUNIKASI": ["Matematika", "Fisika"],
    "INOVASI DIGITAL": ["Matematika", "Fisika"],
    "DESAIN PRODUK INDUSTRI": ["Matematika"],
    "DESAIN INTERIOR": ["Matematika"],
    "DESAIN KOMUNIKASI VISUAL": ["Matematika"],
    "MANAJEMEN BISNIS": ["Matematika"],
    "STUDI PEMBANGUNAN": ["Matematika"],
    "TEKNOLOGI REKAYASA KONSTRUKSI BANGUNAN AIR": ["Matematika", "Fisika"],
    "TEKNOLOGI REKAYASA MANUFAKTUR": ["Matematika", "Fisika"],
    "TEKNOLOGI REKAYASA KONVERSI ENERGI": ["Matematika", "Fisika"],
    "TEKNOLOGI REKAYASA OTOMASI": ["Matematika", "Fisika"],
    "TEKNOLOGI REKAYASA KIMIA INDUSTRI": ["Fisika", "Kimia"],
    "TEKNOLOGI REKAYASA INSTRUMENTASI": ["Matematika", "Fisika"],
    "STATISTIKA BISNIS": ["Matematika"],
    "TEKNOLOGI KEDOKTERAN": ["Matematika", "Fisika"]
}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    features = {}

    for key in request.form:
        val = request.form[key]
        if key.startswith("kls") and any(sub in key for sub in subject_mapping):
            prefix = key[:key.rindex('_') + 1]
            subjek = key[key.rindex('_') + 1:]
            mapel = subject_mapping.get(subjek)
            if mapel:
                features[prefix + mapel] = float(val) if val else 0.0
        elif key in ['pres1_tahun', 'pres2_tahun', 'pres3_tahun']:
            features[key] = float(val) if val else 0.0
        else:
            features[key] = val

    for i in range(1, 4):
        features[f'pres{i}_jenjang'] = features.get(f'pres{i}_jenjang', '')
        features[f'juara_pres{i}'] = features.get(f'juara_pres{i}', '')

    nilai_cols = [f'kls{kls}_sem{sem}_{mapel}' for kls in [10, 11, 12] for sem in [1, 2] for mapel in subject_mapping.values()]
    nilai_values = [features.get(col) for col in nilai_cols if features.get(col) is not None]

    features["jumlah_nilai"] = sum(nilai_values)
    features["average_nilai"] = sum(nilai_values) / len(nilai_values) if nilai_values else 0

    for i in range(1, 4):
        jenjang = features.get(f'pres{i}_jenjang', '')
        features[f'bobot_jenjang_{i}'] = bobot_jenjang.get(jenjang, 0.0)

    for i in range(1, 4):
        juara = features.get(f'juara_pres{i}', '')
        features[f'bobot_juara_pres{i}'] = bobot_juara_mapping.get(juara, 0.0)

    features["bobot_juara_total"] = sum([features.get(f'bobot_juara_pres{i}', 0) for i in range(1, 4)])

    features["skor_prestasi_total"] = 0.0
    for i in range(1, 4):
        skor = features[f'bobot_jenjang_{i}'] * features[f'bobot_juara_pres{i}']
        if skor > 0:
            features["skor_prestasi_total"] += skor

    def hitung_rata_mapel_pendukung(pilihan, nilai_columns, features):
        jurusan = str(pilihan).strip().upper()
        mapel_list = mapel_pendukung.get(jurusan, [])
        nilai_total = 0
        count = 0
        for mapel in mapel_list:
            kolom_mapel = [col for col in nilai_columns if col.lower().endswith(mapel.lower())]
            for kol in kolom_mapel:
                nilai_total += features.get(kol, 0)
            count += len(kolom_mapel)
        if len(mapel_list) == 2:
            return nilai_total / 10
        elif len(mapel_list) == 1:
            return nilai_total / 5
        else:
            return 0

    features["bobot_mapel_pendukung1"] = hitung_rata_mapel_pendukung(features.get("pil1", ""), nilai_cols, features)
    features["bobot_mapel_pendukung2"] = hitung_rata_mapel_pendukung(features.get("pil2", ""), nilai_cols, features)

    pil1_valid = features.get("pil1", "").strip() != "" and features["pil1"] != "Tidak Ada Pilihan 1"
    pil2_valid = features.get("pil2", "").strip() != "" and features["pil2"] != "Tidak Ada Pilihan 2"

    if pil1_valid and pil2_valid:
        features["nilai_mapel_pendukung"] = (
            features["bobot_mapel_pendukung1"] + features["bobot_mapel_pendukung2"]
        ) / 2
    elif pil1_valid:
        features["nilai_mapel_pendukung"] = features["bobot_mapel_pendukung1"]
    elif pil2_valid:
        features["nilai_mapel_pendukung"] = features["bobot_mapel_pendukung2"]
    else:
        features["nilai_mapel_pendukung"] = 0

    features["skor_total_1"] = (
        0.5 * features["average_nilai"] +
        0.2 * features["skor_prestasi_total"] * 100 +
        0.3 * features["nilai_mapel_pendukung"]
    )

    fitur_nama = model.feature_names_
    input_data = []
    for f in fitur_nama:
        if f in cat_features:
            input_data.append(str(features.get(f, "")))
        else:
            input_data.append(float(features.get(f, 0)))

    cat_feature_indices = [i for i, name in enumerate(fitur_nama) if name in cat_features]

    print("\n=== DEBUG Fitur Input ===")
    for k, v in features.items():
        print(f"{k}: {v}")
    print("Input ke model:", input_data)
    print("Index fitur kategorikal:", cat_feature_indices)
    print("=========================\n")

    try:
        pool = Pool(data=[input_data], cat_features=cat_feature_indices)
        hasil = model.predict(pool)
        print(f"Hasil prediksi model: {hasil[0]}")
        label = "Lulus" if hasil[0] == 1 else "Tidak Lulus"
        return render_template('result.html', prediction=label)
    except Exception as e:
        print(f"❌ Error saat prediksi: {e}")
        return render_template('result.html', prediction=f"Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)