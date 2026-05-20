# Projet-Multi
.
├── Part1_preprocessing.py   # Conversion YCbCr + sous-échantillonnage 4:2:0
├── Part2_Iframe.py          # Codage intra-frame (DCT + quantification)
├── Part3_Pframe.py          # Codage inter-frame (estimation de mouvement)
├── Part4_entropy.py         # Codage entropique (zlib + format binaire)
├── Part5_evaluation.py      # Métriques, graphiques et visualisation pipeline
├── encoded_video.bin        # Fichier binaire compressé (sortie)
├── pipeline_visualisation.png
├── plot_compression_vs_qf.png
├── plot_compression_vs_gop.png
├── plot_psnr_per_frame.png
├── README.md
└── report.pdf

// pip install numpy scipy opencv-python matplotlib

//execute

python Part5_evaluation.py --frames my_Frames --output output 

// OR

python Part5_evaluation.py --frames my_Frames/ --output output/ --gop 8 --qf 1.0 --window 8

//Lancer le test synthétique (sans frames réelles)

python Part5_evaluation.py