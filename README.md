#ChatBot Lucio
##Progetto del corso di Text Mining - Prof. Flora Amato
###ChatBot con base di conoscenza riguardante la Camera dei Deputati Italiana

Il file "Text Mining - ChatBot Documentation.pdf" presenta l'intera documentazione del proggetto.
Per quanto riguarda la parte applicativa, essa è suddivisa in due parti:

	- **chatbot** contiene il notebook con il il codice relativo al chatbot, il quale può essere eseguito su piattafrome come GoogleColab.
		Per l'esecuzione è necessario caricare la cartella faiss_db contenente il database vettoriale della base di conoscenza.
		E' inoltre possibile eseguire il tutto in locale con il file dashboard_locale.py. 
		Essendoci la libreria 'stramlit' è necessario eseguire quest'ultimo con il seguente comando:
		'''
		streamlit run dashboard_locale.py
		'''
	- **embedding** è tutta la parte che ci permette di creare la nostro database, partendo dai documenti di tipo .docx
