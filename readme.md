- Tutti i dati sono salvati nella directory `data` in formato `csv`.
- I dati relativia gli ordini sono memorizzati come singole righe. Nel codice vengono invece utilizzati sotto forma di matrice binaria *clienti x prodotti* e per ogni giorno viene utilizzata una matrice diversa contenente tutti gli ordini della giornata.
- Per il plot dell'albero di classificazione è necessario http://www.graphviz.org/Download.php e il modulo `pydotplus`
- La generazione dei dati con dataset_creator.py è tanto lunga. Una matrice degli ordini di dimensione 100x1000 richiede circa 50 secondi. 1000x1000 richiede più di 10 minuti. Genero qundi 1000 clienti, 1000 prodotti e il modello clienti/prodotti (se necessario) e poi lancio 10 job con 100 clienti e 1000 prodotti
