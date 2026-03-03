# REQUISITI:

Python \>= 3.10\
chainlit==2.9.6\
chromadb==1.5.2\
sentence-transformers==5.2.3\
groq==1.0.0\
pypdf

# DA FARE PRIMA DELL'AVVIO:

Copia .env.example\
Rinominalo in .env\
Inserisci la tua chiave Groq

# AVVIO:

chainlit run main.py

# FILE DI PROVA:

Introduzione_alla_teoria_dei_nodi.pdf

# DOMANDE D'ESEMPIO (relative al file di prova):

Cos'è il gruppo di un nodo?\
Cosa sai dei nodi torici?\
Tutti i nodi hanno una superficie di Seifert?

# STRUTTURA GENERALE:

Il chatbot possiede un sistema di Retrieval-Augmented Generation (RAG)
che procede nelle seguenti fasi:

-   **Ingestion:** il documento viene letto, normalizzato e suddiviso in
    chunk.\
-   **Indexing:** i chunk vengono trasformati in embeddings e salvati
    nel vector store.\
-   **Retrieval:** la domanda viene convertita in embedding e
    confrontata con i chunk per trovare i risultati più rilevanti.\
-   **Augmentation:** i chunk recuperati vengono concatenati e inseriti
    nel prompt come contesto documentale.\
-   **Generation:** l'LLM genera la risposta utilizzando esclusivamente
    il contesto fornito.

# PIPELINE:

## Al caricamento del file:

Upload del file → lettura del testo → normalizzazione → suddivisione in
chunk → calcolo embeddings (locale) → salvataggio in ChromaDB con
metrica cosine

## Alla domanda dell'utente:

Domanda utente → embedding della query → retrieval top-k su Chroma →
filtro di qualità (soglia distanza) → selezione e limitazione del
contesto → inserimento del contesto nel prompt (augmentation) → chiamata
al LLM via Groq → risposta
