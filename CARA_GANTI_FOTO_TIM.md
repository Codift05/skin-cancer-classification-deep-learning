# Cara Mengganti Foto Tim di Aplikasi

## Lokasi Foto Tim

Foto tim placeholder berada di file `app/app.py` sekitar baris 310-320.

## Instruksi Penggantian

### Opsi 1: Gunakan URL Gambar Online

Jika foto tim sudah di-upload ke internet (Google Drive, Imgur, dll):

```python
# Ganti bagian ini di app.py:
st.markdown("""
<div class='team-photo' style='background: url("URL_FOTO_ANDA_DI_SINI") center/cover;'>
    <div style='background: rgba(0,0,0,0.5); padding: 2rem; border-radius: 12px;'>
        <h3>Tim Pengembang</h3>
        <p>Nama Anggota Tim</p>
    </div>
</div>
""", unsafe_allow_html=True)
```

### Opsi 2: Gunakan Gambar Lokal

1. Buat folder `assets` di root project:
   ```
   mkdir assets
   ```

2. Copy foto tim Anda ke folder `assets`:
   ```
   assets/team_photo.jpg
   ```

3. Update code di `app.py`:

```python
from PIL import Image

# Di bagian team photo:
team_photo_path = Path(__file__).parent.parent / "assets" / "team_photo.jpg"

if team_photo_path.exists():
    team_image = Image.open(team_photo_path)
    st.image(team_image, use_column_width=True)
else:
    # Tampilkan placeholder
    st.markdown("""
    <div class='team-photo'>
        <svg style='width: 80px; height: 80px; margin-bottom: 1rem;' viewBox='0 0 24 24' fill='white'>
            <path d='M16 17V19H2V17S2 13 9 13 16 17 16 17M12.5 7.5A3.5 3.5 0 1 0 9 11A3.5 3.5 0 0 0 12.5 7.5M15.94 13A5.32 5.32 0 0 1 18 17V19H22V17S22 13.37 15.94 13M15 4A3.39 3.39 0 0 0 13.07 4.59A5 5 0 0 1 13.07 10.41A3.39 3.39 0 0 0 15 11A3.5 3.5 0 0 0 15 4Z'/>
        </svg>
        <h3>Team Photo</h3>
        <p>Masukkan foto tim Anda di sini</p>
    </div>
    """, unsafe_allow_html=True)
```

### Opsi 3: Gunakan Base64 (Embed langsung)

Jika ingin embed foto langsung dalam code:

1. Convert foto ke base64:
   ```python
   import base64
   
   with open("team_photo.jpg", "rb") as img_file:
       b64_string = base64.b64encode(img_file.read()).decode()
   
   print(f"data:image/jpeg;base64,{b64_string}")
   ```

2. Gunakan hasilnya di CSS:
   ```python
   st.markdown("""
   <div style='background: url("data:image/jpeg;base64,BASE64_STRING_ANDA") center/cover;
                height: 300px; border-radius: 12px;'>
   </div>
   """, unsafe_allow_html=True)
   ```

## Contoh Code Lengkap (Recommended)

Tambahkan di `app.py` setelah placeholder team photo:

```python
# Team Photo Section
st.markdown("<br>", unsafe_allow_html=True)

# Coba load foto dari folder assets
team_photo_path = Path(__file__).parent.parent / "assets" / "team_photo.jpg"

if team_photo_path.exists():
    # Jika ada foto, tampilkan
    st.markdown("""
    <div class='card' style='text-align: center;'>
        <h3 style='color: #2c3e50; margin-bottom: 1rem;'>Tim Pengembang</h3>
    </div>
    """, unsafe_allow_html=True)
    
    team_image = Image.open(team_photo_path)
    st.image(team_image, use_column_width=True, caption="Tim Pengembang Skin Cancer Classifier")
    
    # Informasi tim (opsional)
    st.markdown("""
    <div class='card' style='text-align: center;'>
        <h4>Anggota Tim:</h4>
        <p>Nama 1 - NIM 1</p>
        <p>Nama 2 - NIM 2</p>
        <p>Nama 3 - NIM 3</p>
    </div>
    """, unsafe_allow_html=True)
else:
    # Placeholder jika belum ada foto
    st.markdown("""
    <div class='team-photo'>
        <svg style='width: 80px; height: 80px; margin-bottom: 1rem;' viewBox='0 0 24 24' fill='white' xmlns='http://www.w3.org/2000/svg'>
            <path d='M16 17V19H2V17S2 13 9 13 16 17 16 17M12.5 7.5A3.5 3.5 0 1 0 9 11A3.5 3.5 0 0 0 12.5 7.5M15.94 13A5.32 5.32 0 0 1 18 17V19H22V17S22 13.37 15.94 13M15 4A3.39 3.39 0 0 0 13.07 4.59A5 5 0 0 1 13.07 10.41A3.39 3.39 0 0 0 15 11A3.5 3.5 0 0 0 15 4Z'/>
        </svg>
        <h3>Team Photo</h3>
        <p>Masukkan foto tim Anda di sini</p>
        <p style='font-size: 0.9rem; opacity: 0.8;'>
            1. Buat folder 'assets' di root project<br>
            2. Copy foto Anda sebagai 'team_photo.jpg'<br>
            3. Refresh aplikasi
        </p>
    </div>
    """, unsafe_allow_html=True)
```

## Tips Desain Foto

1. **Ukuran Optimal**: 1200x600 pixels (landscape) atau 800x800 (square)
2. **Format**: JPG atau PNG
3. **Ukuran File**: Maksimal 2MB untuk loading cepat
4. **Komposisi**: Pastikan wajah tim terlihat jelas
5. **Background**: Gunakan background profesional atau sesuai tema

## Troubleshooting

**Foto tidak muncul?**
- Cek path file sudah benar
- Pastikan nama file sesuai (case-sensitive)
- Cek permission folder

**Foto terlalu besar/kecil?**
- Gunakan parameter `width` di `st.image()`:
  ```python
  st.image(team_image, width=600)
  ```

**Ingin styling custom?**
- Gunakan CSS di markdown untuk border, shadow, dll:
  ```python
  st.markdown("""
  <style>
  .team-image {
      border-radius: 12px;
      box-shadow: 0 4px 12px rgba(0,0,0,0.15);
  }
  </style>
  """, unsafe_allow_html=True)
  ```

---

**Selamat mencoba! Jika ada pertanyaan, silakan tanya.**
