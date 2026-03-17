# Rapport de projet : Classification de crédit par SVM
**Arthur & Thomas**

---

## Contexte et objectif

L'objectif de ce projet est de prédire si une demande de crédit sera accordée ou non (`LoanApproved` : 0 ou 1), à partir d'un jeu de données issu de Kaggle ([Credit Risk and Loan Default Analysis Dataset](https://www.kaggle.com/datasets/algozee/credit-risk-and-loan-default-analysis-dataset/data)).

Il s'agit d'un problème de **classification binaire supervisée**.

---

## 1. Nettoyage des données

### Ce qui a été fait

Le jeu de données brut contenait plusieurs types d'anomalies que nous avons traitées méthodiquement :

**Observations incohérentes**
- Des individus présentaient un écart entre leur âge et leurs années d'expérience inférieur à 16 ans (ex. : 35 ans d'âge pour 30 ans d'expérience), ce qui est impossible. Ces lignes ont été supprimées.
- Des montants de prêt négatifs (`LoanAmount < 0`) ont également été retirés car ils n'ont pas de sens.

**Valeurs manquantes**
- Les lignes avec **au moins 2 valeurs manquantes** (représentant > 20 % des colonnes d'une observation) ont été supprimées, soit 12 lignes, ce qui correspond à moins de 1% du jeu de données.
- Les lignes avec **une seule valeur manquante** ont été conservées. Les valeurs ont été imputées :
  - `Income` et `CreditScore` → remplacement par la **moyenne**
  - `Education` → remplacement par le **mode**

**Revenus négatifs**
- Quelques observations présentaient un `Income` négatif. Ces cas ont été conservés : un revenu annuel net négatif reste plausible (dettes, déficits).

**Valeurs aberrantes**
- Les outliers détectés via boxplots ont été **conservés** volontairement, les données étant synthétiques et les valeurs extrêmes potentiellement informatives.

> **Limite identifiée :** les données sont générées artificiellement, ce qui explique les incohérences (âge/expérience, prêts négatifs). Cela réduit la portée réelle des conclusions.

---

## 2. Exploration et visualisation

### Ce qui a été observé

Trois paires de variables ont été visualisées en nuages de points avec la cible `LoanApproved` comme couleur :

| Paire de variables | Observation |
|---|---|
| `Income` vs `LoanAmount` | Légère séparation entre les deux groupes |
| `CreditScore` vs `LoanAmount` | Séparation visible |
| `CreditScore` vs `Income` | **Meilleure séparation** entre les deux groupes |

Dans chaque graphique, les moyennes par groupe (droites noires) sont systématiquement décalées par rapport aux moyennes globales (droites rouges) pour le groupe `LoanApproved = 1`, confirmant que les deux classes ont des profils distincts.

**Conclusion :** la combinaison `CreditScore` + `Income` est la plus discriminante visuellement. Ces deux features ont donc été retenues pour visualiser les frontières de décision des modèles.

---

## 3. Encodage et préparation des variables

- Les variables qualitatives (`Education`, `Gender`, `City`, `EmploymentType`) ont été encodées en **one-hot encoding** via `pd.get_dummies` (avec suppression de la première modalité pour éviter la multicolinéarité).
- Une **matrice de corrélation de Spearman** a été calculée sur les variables numériques : **aucune forte corrélation entre variables explicatives** n'a été détectée.

**Séparation train/test :** 80 % / 20 %, avec `shuffle=True` et `random_state=42`.

**Standardisation :** `StandardScaler` appliqué sur le train, puis transformé sur le test (pas de fuite de données).

---

## 4. Gestion du déséquilibre des classes

La variable cible `LoanApproved` était **déséquilibrée** : la classe 0 (crédit refusé) était nettement majoritaire. Pour corriger cela, un **Under-Sampling aléatoire** (`RandomUnderSampler`) a été appliqué sur le jeu d'entraînement uniquement, afin de rééquilibrer les deux classes.

---

## 5. Modélisation — Ce qui a été essayé

### Quatre noyaux SVM testés avec GridSearchCV (optimisation du F1, CV = 5)

| Noyau | Hyperparamètres testés | Meilleur F1 CV |
|---|---|---|
| **Polynomial** | C ∈ {0.1, 1, 10, 50}, degree ∈ {2,3,4,5}, coef0 ∈ {1,2,3} | — |
| **Linéaire** | C ∈ {0.01, 0.1, 1, 10, 50} | — |
| **Sigmoïde** | C, gamma, coef0 | — |
| **RBF** | C ∈ {0.1, 1, 10, 50, 100}, gamma ∈ {scale, auto, 0.01, 0.1, 1, 3} | **0.870** ✅ |

### Random Forest Classifier (comparatif)

Un `RandomForestRegressor` avait d'abord été entraîné sur l'ensemble des features (R² = 0.764), mais cette approche régressive n'était pas adaptée à un problème binaire. Un **`RandomForestClassifier`** a donc été entraîné en remplacement, avec des résultats nettement supérieurs :

| Métrique | Score |
|---|---|
| **Accuracy** | **97.9 %** |
| **Recall** | **94.5 %** |

---

## 6. Résultats du meilleur modèle — SVM RBF

**Paramètres retenus :** `C=1`, `gamma='auto'`

### Matrice de confusion (sur le jeu de test, 656 observations)

|  | Prédit 0 | Prédit 1 |
|---|---|---|
| **Réel 0** | 449 (VP) | 62 (FP) |
| **Réel 1** | 8 (FN) | 137 (VP) |

### Métriques

| Métrique | Train | Test |
|---|---|---|
| **Accuracy** | ~X % | **89,3 %** |
| **Recall classe 0** | — | **87,9 %** |
| **Recall classe 1** | — | **94,5 %** |
| **Précision classe 1** | — | **68,8 %** |

La très faible différence entre les scores train et test indique **l'absence de surapprentissage**.

### Interprétation

- Le modèle identifie 94,5 % des crédits effectivement accordés (très bon recall sur la classe 1).
- Il génère 62 faux positifs : des crédits prédits comme accordés alors qu'ils ne l'étaient pas.
- Dans un contexte métier bancaire, le **recall sur la classe 1** est souvent prioritaire (ne pas manquer un bon dossier), ce qui est bien géré par ce modèle.

---

## 7. Interprétabilité du modèle

### Importance par permutation (SVM RBF)

| Variable | Importance (baisse du F1) |
|---|---|
| `CreditScore` | ~0.36 (**1ère**) |
| `Unemployed` | ~0.24 |
| `Income` | ~0.10 |
| Autres variables | ~0 (impact négligeable) |

Le modèle repose principalement sur **3 variables** liées à la situation financière : score de crédit, statut d'emploi et revenu.

### Partial Dependence Plots

Des PDP ont été tracés pour toutes les features afin de visualiser l'effet marginal de chaque variable sur la prédiction, indépendamment des autres.

### SHAP (sur le Random Forest)

Un graphique `waterfall` a été produit pour une observation individuelle. Pour cet exemple :

- `CreditScore` contribue à **+0.29** sur la prédiction
- `Unemployed` contribue à **+0.12**
- `Income` contribue à **+0.07**

La prédiction finale est de **1** (crédit accordé), expliquée à l'essentiel par un bon score de crédit et une situation économique favorable.

> **Note :** LIME a aussi été initialisé dans le code, mais les analyses SHAP ont finalement été réalisées via `TreeExplainer` sur le Random Forest (les SVM ne supportant pas nativement SHAP).

---

## 8. Ce qui n'a pas marché / Limites

- **LIME** a été initialisé mais non exploité jusqu'au bout dans le rapport final.
- Le **Random Forest Regressor** a d'abord été utilisé comme comparatif (R² = 0.764), mais l'approche régressive n'était pas adaptée à un problème binaire. Un `RandomForestClassifier` a été entraîné en remplacement, obtenant 97.9 % d'accuracy, ce qui surpasse le SVM RBF sur cette métrique.
- Les données étant **synthétiques**, les résultats ne sont pas généralisables à des données réelles de crédit.
- Le **under-sampling** réduit la taille du jeu d'entraînement, ce qui peut limiter la capacité du modèle à apprendre des cas rares.

---

## 9. Conclusion

Le **SVM à noyau RBF** avec `C=1` et `gamma='auto'` est le meilleur modèle SVM de ce projet, avec un F1 de **0.870** en cross-validation et une accuracy de **89.3 %** sur le jeu de test. Cependant, le **Random Forest Classifier** obtient de meilleures performances globales avec **97.9 % d'accuracy** et **94.5 % de recall**, tout en étant plus interprétable via SHAP. Les trois variables les plus importantes restent `CreditScore`, `Unemployed` et `Income`, ce qui est cohérent avec la logique métier de l'octroi de crédit.
