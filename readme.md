- Tutti i dati sono salvati nella directory `data` in formato `csv`.
- I dati relativi agli ordini sono memorizzati come singole righe. Nel codice vengono invece utilizzati sotto forma di matrice binaria *clienti x prodotti* e per ogni giorno viene utilizzata una matrice diversa contenente tutti gli ordini della giornata.
- Per il plot dell'albero di classificazione è necessario http://www.graphviz.org/Download.php e il modulo `pydotplus`
- La generazione dei dati con dataset_creator.py è tanto lunga. Una matrice degli ordini di dimensione 100x1000 richiede circa 50 secondi. 1000x1000 richiede più di 10 minuti. Genero qundi 1000 clienti, 1000 prodotti e il modello clienti/prodotti (se necessario) e poi lancio 10 job con 100 clienti e 1000 prodotti.


## Utilizzo di dataset creator

JSON da utilizzare:
```
{
  "prefix": "test",
  "clients_count": 100,
  "client_resturant_ratio": 0.6,
  "client_gdo_ratio":0.2,
  "client_regular_ratio":0.5,
  "client_consumption_ratio": 0.5,
  "products_count": 10,
  "product_seasonal_ratio":0.2,
  "starting_day": "2017-04-03",
  "days_count": 20,
  "part_size": 5
  "part_size": 25 /* clients_count deve essere divisibile per part_size */
}
```

Creazione di un data set in blocco

```
$ pyton dataset_creator.py -f <pathToJSON>
```

Creazione di un data set parziale

```
$ pyton dataset_creator.py -f <pathToJSON> -s # Genera clienti, prodotti e modello di probabilità
$ pyton dataset_creator.py -f <pathToJSON> -p 0 # per ogni parte
$ pyton dataset_creator.py -f <pathToJSON> -p 1
... 
$ pyton dataset_creator.py -f <pathToJSON> -p n-1 # n = clients_count / part_size
$ pyton dataset_creator.py -f <pathToJSON> -r # ricombina gli ordini in un unico file
```

