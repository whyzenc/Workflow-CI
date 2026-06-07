# Workflow-CI — Melbourne Housing Price Prediction
**Nama:** Whenny Zenica  
**Dataset:** Melbourne Housing Dataset  
**Model:** Linear Regression (Baseline)

---

## Struktur Repository

```
Workflow-CI/
├── .github/
│   └── workflows/
│       └── ci.yml                  ← GitHub Actions CI workflow
└── MLProject/
    ├── modelling.py                ← Script training model
    ├── conda.yaml                  ← Environment dependencies
    ├── MLProject                   ← MLflow Project definition
    └── melb_preprocessed.csv       ← Dataset (upload manual)
```

---

## Setup Sebelum Push

### 1. Tambahkan Dataset
Letakkan file `melb_preprocessed.csv` di dalam folder `MLProject/`.

### 2. Tambahkan GitHub Secrets
Buka **Settings → Secrets and variables → Actions → New repository secret**, lalu tambahkan:

| Secret Name          | Nilai                        |
|----------------------|------------------------------|
| `DOCKERHUB_USERNAME` | Username Docker Hub kamu     |
| `DOCKERHUB_TOKEN`    | Access Token dari Docker Hub |

> Cara buat Docker Hub Token: Login Docker Hub → Account Settings → Security → New Access Token

---

## Cara Menjalankan Workflow

### Otomatis
Workflow berjalan otomatis setiap **push ke branch `main`**.

### Manual (Trigger)
1. Buka tab **Actions** di GitHub
2. Pilih workflow **CI - MLflow Training & Docker Build**
3. Klik **Run workflow**

---

## Output

- ✅ **Artefak MLflow** tersimpan di GitHub Actions (tab Artifacts, retention 30 hari)
- ✅ **Docker Image** ter-push ke Docker Hub:  
  `docker pull <DOCKERHUB_USERNAME>/melbourne-housing-whenny:latest`

---

## Docker Hub
> Tautan: `https://hub.docker.com/r/<DOCKERHUB_USERNAME>/melbourne-housing-whenny`
