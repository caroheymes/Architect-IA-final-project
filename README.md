# Architect-IA-final-project

## questions & bugs
le code de chargement et des cartes se trouve dans https://colab.research.google.com/drive/1JgLCTAt8Ur-atRvAyvvfpzPs1NCLCvWK?usp=sharing section**ICI**
[KRO]  : source des données vitesses `https://data.grandlyon.com/geoserver/metropole-de-lyon/ows?SERVICE=WFS&VERSION=2.0.0&request=GetFeature&typename=metropole-de-lyon:pvo_patrimoine_voirie.pvotrafic&outputFormat=application/json&SRSNAME=EPSG:2154&startIndex=0&sortby=gid`
### gaffe à la projection à utiliser pour l'extraction : impérativement RGF93 / Lambert-93 - France (EPSG:2154)
* hypothèse : on droppe toutes les lignes où properties_est_a_jour est nan : ok ?
* chargement des données de vitesse par script python à intégrer dans airflow. La transformation prend pas loin de 2 minutes à voir si on fait un chargement en base postgresL dans airflow puis un nettoyage dans un autre DAG
* utile/inutile de droppe les properties_vitesse à null (économise 900 lignes)
* comment on gère `properties_sens`, `properties_etat`  pour définir les sens de circulation ? (comparer sur quai jules courmont et sur avenue de saxe pour comprendre

TODO :
* [bug/ correctif] la carte met bien 4/5 minutes à être générée (450 Mo)  si la vitesse des capteurs est identique par `properties_libelle` alors grouper les line string  pour faire des multiplolygones. Après analyse, Ca se confirme 600 polygones ont la même vitesse sur 900 donc faire des multipolygones avec !!!
