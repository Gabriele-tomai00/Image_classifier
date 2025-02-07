
import os
from PIL import Image

# Funzione per leggere le immagini e creare una lista di etichette
def carica_immagini_e_etichette(cartella_principale):
    etichette = []
    immagini = []
    
    # Scorri le sottocartelle nella cartella principale
    for cartella in os.listdir(cartella_principale):
        cartella_path = os.path.join(cartella_principale, cartella)
        
        # Verifica che sia una cartella
        if os.path.isdir(cartella_path):
            # Scorri le immagini all'interno della cartella
            for file in os.listdir(cartella_path):
                file_path = os.path.join(cartella_path, file)
                
                # Verifica che il file sia un'immagine (controlla l'estensione)
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                    try:
                        # Apre l'immagine per verificarne la validità
                        img = Image.open(file_path)
                        img.verify()  # Verifica se l'immagine è valida
                        
                        # Aggiungi l'immagine e la sua etichetta (nome della cartella)
                        immagini.append(file_path)
                        etichette.append(cartella)
                    except (IOError, SyntaxError):
                        # Se l'immagine non è valida, la ignora
                        continue
    
    return immagini, etichette




