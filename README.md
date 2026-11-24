# Activité pratique n°1 — Mise en œuvre du framework PyTorch

**Étudiante :** Amal Dalouch — Groupe G1  

**Lien GitHub :**https://github.com/Amal123-cey/tp1-pytorch-bases.git

Notebook Jupyter exécuté de l’activité pratique sur les bases de PyTorch : tenseurs, autograd, boucle d’apprentissage, puis quatre exercices (salaires, Iris, MNIST avec un MLP, MNIST avec un CNN).

## Contenu

| Fichier | Rôle |
|---|---|
| `TP1_PyTorch_Complet.ipynb` | Notebook de travail, avec explications et sorties |
| `Activité pratique Bases de Pytorch.ipynb` | Support d’origine, non modifié |
| `mnist_mlp.pth` | Poids du MLP MNIST après 5 époques |
| `mnist_cnn.pth` | Poids du CNN MNIST après 3 époques |
| `classifier_state_dict.pt` | Poids du classifieur de la démonstration |
| `data/` | Cache MNIST téléchargé par torchvision |

## Lancer le notebook

Windows, CPU. CUDA est utilisé automatiquement s’il est disponible.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch torchvision matplotlib scikit-learn ipykernel nbconvert jupyter
.\.venv\Scripts\python.exe -m ipykernel install --sys-prefix --name=tp1pytorch --display-name="Python (TP1)"
.\.venv\Scripts\python.exe -m jupyter notebook TP1_PyTorch_Complet.ipynb
```

Pour réexécuter toutes les cellules :

```powershell
.\.venv\Scripts\python.exe -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=1800 --ExecutePreprocessor.kernel_name=tp1pytorch TP1_PyTorch_Complet.ipynb
```

Les chargeurs de données utilisent `num_workers=0`. Le dossier `.venv` ne doit pas être versionné.

## Résultats de l’exécution CPU

- Régression des salaires : MAE de test 381,64 DH. Prédiction pour 7 ans et une expertise de 4 : 14 201,94 DH (valeur sans bruit : 14 200 DH).
- Iris : 100 % pour le réseau 4 → 16 → 3, 90 % pour `nn.Linear(4, 3)`, sur 30 fleurs de test. La fleur `[5.1, 3.5, 1.4, 0.2]` est prédite setosa.
- MLP MNIST, 5 époques, 10 000 images de test : accuracy 97,41 %, précision macro 97,41 %, rappel macro 97,40 %, F1 macro 97,40 %.
- CNN MNIST, 3 époques, dropout 0,25, mêmes 10 000 images : accuracy 98,93 %.

Le MLP et le CNN ne diffèrent pas seulement par l’architecture : le nombre d’époques et le prétraitement ne sont pas les mêmes.
