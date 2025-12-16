README – Etapa 5: Configurarea și Antrenarea Modelului RN
Disciplina: Rețele Neuronale  
Instituție: POLITEHNICA București – FIIR  
Student: [Nume Prenume]  
Link Repository GitHub: https://dictionary.cambridge.org/dictionary/french-english/complet  
Data predării: [Data]
________________________________________
Scopul Etapei 5
Această etapă corespunde punctului 6. Configurarea și antrenarea modelului RN din lista de 9 etape.
Obiectiv principal: Antrenarea efectivă a modelului CNN-LSTM Autoencoder definit în Etapa 4 pe datele de comportament normal, evaluarea performanței de detectare a anomaliilor (prin Eroarea de Reconstrucție) și integrarea modelului antrenat în aplicația completă de supraveghere.
Pornire obligatorie: Arhitectura completă și funcțională din Etapa 4: State Machine definit, cele 3 module funcționale, și minimum 40% date originale în dataset.
________________________________________
PREREQUISITE – Verificare Etapa 4 (OBLIGATORIU)
Înainte de a începe Etapa 5, verificați că aveți din Etapa 4:
•	[ ] State Machine definit și documentat în docs/state_machine.*
•	[ ] Contribuție ≥40% date originale în data/generated/ (verificabil)
•	[ ] Modul 1 (Data Logging) funcțional - produce secvențe .npy video
•	[ ] Modul 2 (RN) cu arhitectură definită dar NEANTRENATĂ (models/untrained_model.pt sau .h5)
•	[ ] Modul 3 (UI/Web Service) funcțional cu model dummy
•	[ ] Tabelul "Nevoie → Soluție → Modul" complet în README Etapa 4
** Dacă oricare din punctele de mai sus lipsește → reveniți la Etapa 4 înainte de a continua.**
________________________________________
Pregătire Date pentru Antrenare
Dacă ați adăugat date noi în Etapa 4 (contribuția de 40%):
TREBUIE să refaceți preprocesarea pe dataset-ul COMBINAT:
Bash
# 1. Combinare date vechi (Etapa 3) + noi (Etapa 4)
python src/preprocessing/combine_datasets.py

# 2. Refacere preprocesare COMPLETĂ (Citire video, Creare secvente, salvare .npy)
python data_processing.py 

# 3. Split: Secvențele normale se împart în train/validation. Secvențele anormale sunt test set.
python src/preprocessing/data_splitter.py --train_ratio 0.7 --val_ratio 0.15 --test_ratio 0.15 
# Proporții split: 70% train (normal) / 15% validation (normal) / 15% test (anormal + normal)
** ATENȚIE - Folosiți ACEIAȘI parametri de preprocesare:**
•	Același scaler (dacă este folosit la normalizare)
•	Aceiași proporții split: 70% train / 15% validation / 15% test
•	Același random_state=42 pentru reproducibilitate
________________________________________
Cerințe Structurate pe 3 Niveluri
Nivel 1 – Obligatoriu pentru Toți (70% din punctaj)
Completați TOATE punctele următoare:
1.	Antrenare model Autoencoder definit în Etapa 4 pe setul final de date NORMAL (≥40% originale).
2.	Minimum 10 epoci, batch size 8–32.
3.	Împărțire stratificată a datelor normale în train/validation: 70% / 15%. Setul de test (anomalii + normal) este 15%.
4.	Tabel justificare hiperparametri (vezi secțiunea de mai jos - OBLIGATORIU).
5.	Metrici calculate pe test set (după stabilirea pragului $\theta$ pe val. set):
   - Acuratețe ≥ 65%
   - F1-score (macro) ≥ 0.60
6.	Salvare model antrenat în models/trained_model.pt (PyTorch) sau .h5 (Keras).
7.	Integrare în UI din Etapa 4:
   - UI trebuie să încarce modelul ANTRENAT.
   - Inferență REALĂ demonstrată (calcul Eroare Reconstrucție și Decizie Alertă).
   - Screenshot în docs/screenshots/inference_real.png.
Tabel Hiperparametri și Justificări (OBLIGATORIU - Nivel 1)
Hiperparametru	Valoare Aleasă	Justificare
Learning rate	0.001	Valoare standard pentru Adam optimizer. Asigură o convergență stabilă fără oscilații mari ale Loss-ului pe setul de antrenare normală.
Batch size	16	Compromis între viteză și memorie GPU, având în vedere că input-ul este constituit din secvențe video (tensor 5D: Batch x Timp x Canale x H x W).
Number of epochs	50	Număr suficient pentru a atinge convergența. Va fi controlat de un mecanism de Early Stopping (Nivel 2).
Optimizer	Adam	Algoritm adaptiv, eficient pentru modelarea complexă a seriilor temporale (LSTM) în cadrul Autoencoder-ului.
Loss function	MSE (Mean Squared Error)	OBLIGATORIU: Măsoară diferența pătratică (eroarea de reconstrucție) între secvența de intrare (cadre normale) și secvența de ieșire.
Activation functions	ReLU (hidden), Sigmoid (output)	Sigmoid în stratul final al Decodorului pentru a re-scala valorile pixelilor reconstruiți în intervalul standard [0, 1].
Justificare detaliată batch size (exemplu):
Am ales batch_size=16 pentru a gestiona eficient tensorii de intrare mari specifici datelor video, unde fiecare sample este o secvență de 16 cadre 128x128. Acest batch size asigură o utilizare rezonabilă a memoriei GPU și menține stabilitatea gradientului în timpul antrenării pe comportamentul normal.
________________________________________
Nivel 2 – Recomandat (85-90% din punctaj)
Includeți TOATE cerințele Nivel 1 + următoarele:
1.	Early Stopping - oprirea antrenării dacă val_loss nu scade în 5 epoci consecutive.
2.	Learning Rate Scheduler - ReduceLROnPlateau sau StepLR.
3.	Augmentări relevante domeniu: (aplicate pe setul de antrenare) Variații de iluminare (brightness), zgomot specific camerelor CCTV.
4.	Grafic Loss (MSE) și Val_Loss în funcție de epoci salvat în docs/loss_curve.png.
5.	Analiză erori context industrial (vezi secțiunea dedicată mai jos - OBLIGATORIU Nivel 2).
Indicatori țintă Nivel 2:
•	Acuratețe ≥ 75% (după aplicarea pragului $\theta$)
•	F1-score (macro) ≥ 0.70 (după aplicarea pragului $\theta$)
________________________________________
Nivel 3 – Bonus (până la 100%)
Punctaj bonus per activitate:
Activitate	 Livrabil
Comparare 2+ arhitecturi diferite	Tabel comparativ + justificare alegere finală în README
Export ONNX/TFLite + benchmark latență	Fișier models/final_model.onnx + demonstrație <50ms
Confusion Matrix + analiză 5 exemple greșite	docs/confusion_matrix.png + analiză în README
________________________________________
Verificare Consistență cu State Machine (Etapa 4)
Antrenarea și inferența trebuie să respecte fluxul din State Machine-ul vostru definit în Etapa 4.
Stare din Etapa 4	Implementare în Etapa 5 (Autoencoder)
PREPROCESS	Aplicare redimensionare/normalizare la secvența de 16 cadre.
RN_INFERENCE	Forward pass cu model ANTRENAT. Ieșirea este secvența reconstruită.
CALCUL SCOR	Se calculează $Loss_{Reconstructie}$ (MSE) între secvența intrare și cea reconstruită.
THRESHOLD_CHECK	Comparare $Loss_{Reconstructie}$ cu pragul $\theta$ (stabilit pe baza setului de validare).
ALERT	Trigger în UI bazat pe predicția modelului real (Loss > $\theta$).
________________________________________
Analiză Erori în Context Industrial (OBLIGATORIU Nivel 2)
1. Pe ce clase greșește cel mai mult modelul?
Context Anomaly Detection: Modelul Autoencoder confundă comportamentul Normal cu cel Anomal. Eroarea majoră este False Negatives (FN), unde modelul clasifică o anomalie (luptă/cădere) ca fiind normală, deoarece eroarea sa de reconstrucție este mică. Confusion Matrix arată că această eroare este concentrată pe anomaliile de tip 'cădere ușoară' sau 'obiect abandonat' (cazuri cu mișcare redusă).
2. Ce caracteristici ale datelor cauzează erori?
Modelul eșuează când apar variații BRUȘTE de iluminare (ex: o mașină mare trece rapid prin cadru), care determină o Eroare de Reconstrucție mare, chiar dacă nu există o anomalie de comportament (rezultând un False Positive). De asemenea, anomaliile subtile sau cele care au loc la marginea câmpului vizual sunt ratate (False Negatives), din cauza rezoluției scăzute din acea zonă.
3. Ce implicații are pentru aplicația industrială?
FALSE NEGATIVES (FN - anomalia reală este ratată): CRITIC → Risc major de securitate (incidentul de luptă/cădere nu este semnalat).
FALSE POSITIVES (FP - alarmă falsă): ACCEPTABIL → Cost suplimentar (operatorul verifică inutil), dar este de preferat unui FN.
Prioritatea sistemului este Minimizarea False Negatives (Sensibilitatea sau Recall) pentru a nu rata evenimente critice, chiar dacă acest lucru implică un număr mai mare de alarme false. Soluția este ajustarea pragului $\theta$.
4. Ce măsuri corective propuneți?
Măsuri corective:
1.	Calibrare Prag $\theta$: Ajustarea pragului $\theta$ (din main_training_inference.py) pentru a favoriza Recall-ul (detecția anomaliilor) în detrimentul preciziei, reducând riscul de FN.
2.	Date de Antrenare mai Robuste: Colectare/Generare 500+ secvențe de comportament 'normal' care includ variații de iluminare (augmentare la nivel de frame) pentru a învăța modelul să le ignore.
3.	Filtrare Pre-Procesare: Implementarea unui filtru simplu pentru a reduce variațiile globale de luminozitate (ex: egalizare adaptivă) înainte de a intra în RN.
4.	Re-antrenare cu parametri optimizați: Testarea altor optimizatori sau a unui Learning Rate mai mic pentru o convergență mai fină.
________________________________________
Structura Repository-ului la Finalul Etapei 5
proiect-rn-[prenume-nume]/
├── README.md
├── etapa3_analiza_date.md
├── etapa4_arhitectura_sia.md
├── etapa5_antrenare_model.md      # ← ACEST FIȘIER (completat)
│
├── docs/
│   ├── state_machine.png              
│   ├── loss_curve.png                 # NOU - Grafic antrenare
│   ├── confusion_matrix.png           # (opțional - Nivel 3)
│   └── screenshots/
│       ├── inference_real.png         # NOU - OBLIGATORIU
│       └── ui_demo.png
│
├── data/                               
│   ├── raw/
│   ├── generated/
│   ├── processed/
│   ├── train/
│   ├── validation/
│   └── test/
│
├── src/
│   ├── data_acquisition/
│   ├── preprocessing/
│   │   └── combine_datasets.py        
│   ├── neural_network/
│   │   ├── model.py
│   │   ├── train.py                   # NOU
│   │   └── evaluate.py                # NOU
│   └── app/
│       └── main.py                    # ACTUALIZAT
│
├── models/
│   ├── untrained_model.h5
│   └── trained_model.h5               # NOU - OBLIGATORIU
│
├── results/                            # NOU
│   ├── training_history.csv           # OBLIGATORIU
│   └── test_metrics.json              # OBLIGATORIU
│
├── config/
└── requirements.txt

