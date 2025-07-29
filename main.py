from functions import procesar_excel

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Uso: python main.py archivo.xlsx")
    else:
        archivo_excel = sys.argv[1]
        procesar_excel(archivo_excel)
 