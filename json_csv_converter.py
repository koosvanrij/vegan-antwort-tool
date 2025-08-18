#!/usr/bin/env python3
"""
JSON-CSV Konverter
Konvertiert zwischen JSON und CSV Formaten für einfache Bearbeitung in Apple Numbers
"""

import json
import csv
import sys
import os
from pathlib import Path


def json_to_csv(json_file, csv_file=None):
    """
    Konvertiert JSON zu CSV
    
    Args:
        json_file (str): Pfad zur JSON-Datei
        csv_file (str): Pfad zur CSV-Datei (optional, wird automatisch generiert wenn nicht angegeben)
    """
    try:
        # JSON-Datei laden
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # CSV-Dateiname generieren wenn nicht angegeben
        if csv_file is None:
            csv_file = json_file.replace('.json', '.csv')
        
        # Überprüfen ob data eine Liste ist
        if isinstance(data, list) and len(data) > 0:
            # Feldnamen aus dem ersten Objekt extrahieren
            fieldnames = list(data[0].keys())
            
            # CSV-Datei schreiben
            with open(csv_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(data)
                
        elif isinstance(data, dict):
            # Falls es ein einzelnes Objekt ist, in Liste umwandeln
            fieldnames = list(data.keys())
            
            with open(csv_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerow(data)
        else:
            raise ValueError("JSON muss ein Objekt oder eine Liste von Objekten sein")
            
        print(f"✅ JSON zu CSV konvertiert: {json_file} → {csv_file}")
        
    except FileNotFoundError:
        print(f"❌ Fehler: Datei {json_file} nicht gefunden")
    except json.JSONDecodeError as e:
        print(f"❌ Fehler beim Lesen der JSON-Datei: {e}")
    except Exception as e:
        print(f"❌ Unerwarteter Fehler: {e}")


def csv_to_json(csv_file, json_file=None):
    """
    Konvertiert CSV zu JSON
    
    Args:
        csv_file (str): Pfad zur CSV-Datei
        json_file (str): Pfad zur JSON-Datei (optional, wird automatisch generiert wenn nicht angegeben)
    """
    try:
        # JSON-Dateiname generieren wenn nicht angegeben
        if json_file is None:
            json_file = csv_file.replace('.csv', '.json')
        
        data = []
        
        # CSV-Datei lesen
        with open(csv_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Leere Werte als None behandeln oder entfernen je nach Bedarf
                cleaned_row = {k: (v if v.strip() else None) for k, v in row.items()}
                data.append(cleaned_row)
        
        # JSON-Datei schreiben
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            
        print(f"✅ CSV zu JSON konvertiert: {csv_file} → {json_file}")
        
    except FileNotFoundError:
        print(f"❌ Fehler: Datei {csv_file} nicht gefunden")
    except Exception as e:
        print(f"❌ Unerwarteter Fehler: {e}")


def show_help():
    """Zeigt die Hilfe an"""
    help_text = """
JSON-CSV Konverter

Verwendung:
    python converter.py json2csv <json_datei> [csv_datei]
    python converter.py csv2json <csv_datei> [json_datei]
    
Beispiele:
    python converter.py json2csv fragen.json
    python converter.py json2csv fragen.json fragen_export.csv
    python converter.py csv2json fragen.csv
    python converter.py csv2json fragen_bearbeitet.csv neue_fragen.json

Optionen:
    -h, --help    Diese Hilfe anzeigen
    """
    print(help_text)


def main():
    """Hauptfunktion"""
    if len(sys.argv) < 2 or sys.argv[1] in ['-h', '--help']:
        show_help()
        return
    
    command = sys.argv[1].lower()
    
    if command == 'json2csv':
        if len(sys.argv) < 3:
            print("❌ Fehler: JSON-Datei muss angegeben werden")
            show_help()
            return
            
        json_file = sys.argv[2]
        csv_file = sys.argv[3] if len(sys.argv) > 3 else None
        json_to_csv(json_file, csv_file)
        
    elif command == 'csv2json':
        if len(sys.argv) < 3:
            print("❌ Fehler: CSV-Datei muss angegeben werden")
            show_help()
            return
            
        csv_file = sys.argv[2]
        json_file = sys.argv[3] if len(sys.argv) > 3 else None
        csv_to_json(csv_file, json_file)
        
    else:
        print(f"❌ Unbekannter Befehl: {command}")
        show_help()


if __name__ == "__main__":
    main()


# Zusätzliche Hilfsfunktionen für interaktive Nutzung

def quick_json_to_csv(json_file):
    """Schnelle JSON zu CSV Konvertierung"""
    json_to_csv(json_file)

def quick_csv_to_json(csv_file):
    """Schnelle CSV zu JSON Konvertierung"""
    csv_to_json(csv_file)

# Beispiel für die Nutzung in PyCharm:
# quick_json_to_csv('meine_fragen.json')
# quick_csv_to_json('meine_fragen.csv')